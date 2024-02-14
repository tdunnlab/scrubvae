
class HierarchicalResidualEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        ch=64,
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        init_dilation=None,
    ):
        super(HierarchicalResidualEncoder, self).__init__()
        if init_dilation is None:
            dilation = torch.ones(4)
        else:
            dilation = init_dilation * 2 ** torch.arange(4)

        self.conv_in = nn.Conv1d(in_channels, ch, 7, 1, 3)
        self.activation = nn.Tanh() if activation == "tanh" else nn.PReLU()

        self.res_layers = nn.Sequential(
            ResidualBlock(ch, 2 * ch, kernel, activation, dilation=dilation[0].item()),
            ResidualBlock(
                2 * ch, 4 * ch, kernel, activation, dilation=dilation[1].item()
            ),
            ResidualBlock(
                4 * ch, 8 * ch, kernel, activation, dilation=dilation[2].item()
            ),
            ResidualBlock(
                8 * ch, 16 * ch, kernel, activation, dilation=dilation[3].item()
            ),
        )
        self.flatten = nn.Flatten()

        sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
        flatten_dim1 = find_latent_dim(window, kernel, 2, dilation[:2]) * 4 * ch
        flatten_dim2 = find_latent_dim(window, kernel, 4, dilation) * 16 * ch

        self.fc_mu1 = nn.Linear(flatten_dim1, z_dim)
        self.fc_sigma1 = nn.Sequential(
            nn.Linear(flatten_dim1, sig_dim), CholeskyL(z_dim, is_diag)
        )

        self.fc_mu2 = nn.Linear(flatten_dim2, z_dim)
        self.fc_sigma2 = nn.Sequential(
            nn.Linear(flatten_dim2, sig_dim), CholeskyL(z_dim, is_diag)
        )

    def forward(self, x):
        x = self.activation(self.conv_in(x))
        x = self.res_layers[:2](x)
        mu1 = self.fc_mu1(self.flatten(x))
        sigma1 = self.fc_sigma1(self.flatten(x))

        x = self.res_layers[2:](x)
        x = self.flatten(x)
        mu2 = self.fc_mu2(x)
        sigma2 = self.fc_sigma2(x)
        return mu1, mu2, sigma1, sigma2


class HierarchicalResidualDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        ch=64,
        kernel=5,
        z_dim=64,
        window=200,
        activation="prelu",
        invariant_dim=0,
        init_dilation=None,
    ):
        super(HierarchicalResidualDecoder, self).__init__()
        self.invariant_dim = invariant_dim
        if init_dilation is None:
            dilation = torch.ones(4)
        else:
            dilation = init_dilation * 2 ** torch.arange(4)
        # L_o = window-1)
        flatten_dim = find_latent_dim(window, kernel, 4, dilation)
        self.fc2_in = nn.Linear(z_dim + invariant_dim, flatten_dim * 16 * ch)
        self.unflatten2 = nn.Unflatten(1, (16 * ch, -1))

        shallow_dim = find_out_dim(
            flatten_dim,
            kernel,
            2,
            dilation[2:],
        )

        self.fc1_in = nn.Linear(z_dim, shallow_dim * 4 * ch)
        self.unflatten1 = nn.Unflatten(1, (4 * ch, -1))

        self.res_layers = nn.Sequential(
            ResidualBlockTranspose(
                ch * 16,
                ch * 8,
                kernel,
                activation=activation,
                dilation=dilation[3].item(),
            ),
            ResidualBlockTranspose(
                ch * 8,
                ch * 4,
                kernel,
                activation=activation,
                dilation=dilation[2].item(),
            ),
            ResidualBlockTranspose(
                ch * 4,
                ch * 2,
                kernel,
                activation=activation,
                dilation=dilation[1].item(),
            ),
            ResidualBlockTranspose(
                ch * 2,
                ch,
                kernel,
                activation=activation,
                dilation=dilation[0].item(),
            ),
        )

        l_out = find_out_dim(flatten_dim, kernel, 4, dilation)

        final_kernel = window - l_out + 7
        print("Final ConvOut Kernel: {}".format(final_kernel))
        self.conv_out = nn.ConvTranspose1d(ch, out_channels, final_kernel, 1, 3)
        # self.activation = nn.Tanh()

    def forward(self, x_shallow, x_deep):
        x = self.unflatten2(self.fc2_in(x_deep))
        # x = self.activation(self.conv_in(x))
        x = self.res_layers[:2](x)
        x += self.unflatten1(self.fc1_in(x_shallow))
        x = self.res_layers[2:](x)
        x = torch.tanh(self.conv_out(x))
        return x


class HResVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        ch=64,
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        invariant_dim=0,
        init_dilation=None,
    ):
        super(HResVAE, self).__init__()
        self.in_channels = in_channels
        self.window = window
        self.is_diag = is_diag
        self.invariant_dim = invariant_dim
        self.encoder = HierarchicalResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim // 2,
            window=window,
            activation=activation,
            is_diag=is_diag,
        )
        self.decoder = HierarchicalResidualDecoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim // 2,
            window=window,
            activation=activation,
            invariant_dim=invariant_dim,
        )

    def sampling(self, mu, L):
        eps = torch.randn_like(mu)
        return torch.matmul(L, eps[..., None]).squeeze().add_(mu)

    def forward(self, x, invariant=None):
        in_shape = x.shape
        mu1, mu2, L1, L2 = self.encoder(
            x.moveaxis(1, -1).view(-1, self.in_channels, self.window)
        )
        z1 = self.sampling(mu1, L1)
        z2 = self.sampling(mu2, L2)

        if invariant is not None:
            z2 = torch.cat((z2, invariant), dim=-1)

        x_hat = self.decoder(z1, z2).moveaxis(-1, 1).reshape(in_shape)
        return x_hat, (mu1, mu2), (L1, L2)
