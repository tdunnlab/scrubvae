import torch.nn as nn
import torch.nn.functional as F
import torch


def find_latent_dim(
    window_size: int, kernel: int, num_layers: int, dilation=torch.ones(4)
):
    stride = 1 if any(dilation > 1) else 2
    layer_out = (
        lambda l_in, dil: (l_in + 2 * (kernel // 2) - dil * (kernel - 1) - 1) / stride
        + 1
    )

    l_out = window_size
    for i in range(num_layers):
        l_out = layer_out(l_out, dilation[i])

    return int(l_out)


def find_out_dim(latent_dim: int, kernel: int, num_layers: int, dilation=torch.ones(4)):
    stride = 1 if any(dilation > 1) else 2
    layer_out = (
        lambda l_in, dil: (l_in - 1) * stride
        - 2 * (kernel // 2)
        + dil * (kernel - 1)
        + 1
    )
    l_out = latent_dim
    for i in range(num_layers):
        l_out = layer_out(l_out, dilation[-i])

    return int(l_out)


class CholeskyL(nn.Module):
    def __init__(
        self,
        z_dim: torch.Tensor,
        is_diag: bool,
    ):
        """
        Reshapes x input into lower triangle positive definite matrix, L.
        Such that LL^T = \Sigma

        Option of only creating diagonal L
        """
        super(CholeskyL, self).__init__()
        self.z_dim = z_dim
        self.is_diag = is_diag
        # embed the elements into the matrix
        if is_diag:
            self.idxs = torch.arange(z_dim)[None, :].repeat(2, 1)
        else:
            self.idxs = torch.tril_indices(z_dim, z_dim)

    def forward(self, x):
        L = torch.zeros(x.shape[0], self.z_dim, self.z_dim, device=x.device)
        L[:, self.idxs[0], self.idxs[1]] = x
        # apply softplus to the diagonal entries to guarantee the resulting
        # matrix is positive definite
        new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
        L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
        # reshape y_hat so we can concatenate it to L
        return L


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=5, activation="prelu", dilation=1
    ):
        stride = 1 if dilation > 1 else 2

        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels // 2,
                kernel,
                stride,
                kernel // 2,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels // 2, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
            nn.Conv1d(
                out_channels // 2,
                out_channels,
                kernel,
                1,
                kernel // 2,
                dilation=1,
            ),
        )

        self.skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride,
            kernel // 2,
            dilation=dilation,
        )

        self.add = nn.Sequential(
            nn.BatchNorm1d(out_channels, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.residual(x)
        return self.add(x + skip)


class ResidualBlockTranspose(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=5,
        scale_factor=2,
        activation="prelu",
        dilation=1,
    ):
        super(ResidualBlockTranspose, self).__init__()

        stride = 1 if dilation > 1 else 2

        self.residual = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel,
                1,
                kernel // 2,
                dilation=1,
            ),
            nn.BatchNorm1d(in_channels // 2, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
            nn.ConvTranspose1d(
                in_channels // 2,
                out_channels,
                kernel,
                stride,
                kernel // 2,
                dilation=dilation,
            ),
        )

        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False),
            nn.Conv1d(
                in_channels,
                out_channels,
                (kernel + 1),
                1,
                kernel // 2,
                dilation=dilation,
            ),
        )

        self.add = nn.Sequential(
            nn.BatchNorm1d(out_channels, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.residual(x)
        return self.add(x + skip)


class ResidualEncoder(nn.Module):
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
        super(ResidualEncoder, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, ch, 7, 1, 3)
        self.activation = nn.Tanh() if activation == "tanh" else nn.PReLU()

        if init_dilation is None:
            dilation = torch.ones(4, dtype=int)
        else:
            dilation = init_dilation * 2 ** torch.arange(4)

        self.res_layers = nn.Sequential(
            ResidualBlock(ch, 2 * ch, kernel, activation, dilation[0].item()),
            ResidualBlock(2 * ch, 4 * ch, kernel, activation, dilation[1].item()),
            ResidualBlock(4 * ch, 8 * ch, kernel, activation, dilation[2].item()),
            ResidualBlock(8 * ch, 16 * ch, kernel, activation, dilation[3].item()),
        )
        self.flatten = nn.Flatten()

        flatten_dim = find_latent_dim(window, kernel, 4, dilation) * 16 * ch
        self.fc_mu = nn.Linear(flatten_dim, z_dim)
        sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
        self.fc_sigma = nn.Sequential(
            nn.Linear(flatten_dim, sig_dim), CholeskyL(z_dim, is_diag)
        )

    def forward(self, x):
        x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        ch=64,
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        invariant_dim=0,
        init_dilation=None,
    ):
        super(ResidualDecoder, self).__init__()
        self.invariant_dim = invariant_dim

        if init_dilation is None:
            dilation = torch.ones(4, dtype=int)
        else:
            dilation = init_dilation * 2 ** torch.arange(4)

        flatten_dim = find_latent_dim(window, kernel, 4, dilation) * 16 * ch
        self.fc_in = nn.Linear(z_dim + invariant_dim, flatten_dim)
        self.unflatten = nn.Unflatten(1, (16 * ch, -1))
        # self.conv_in = nn.ConvTranspose1d(int(flatten_dim[0]), ch*16, 3, 1, 1)

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

        l_out = find_out_dim(find_latent_dim(window, kernel, 4), kernel, 4)

        final_kernel = window - l_out + 7
        print("Final ConvOut Kernel: {}".format(final_kernel))
        self.conv_out = nn.ConvTranspose1d(ch, out_channels, final_kernel, 1, 3)
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.unflatten(self.fc_in(x))
        # x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        x = torch.tanh(self.conv_out(x))
        return x


class ResVAE(nn.Module):
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
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
    ):
        super(ResVAE, self).__init__()
        self.in_channels = in_channels
        self.ch=ch
        self.window = window
        self.is_diag = is_diag
        self.invariant_dim = invariant_dim
        self.kinematic_tree = kinematic_tree
        self.register_buffer("arena_size", arena_size)
        # self.arena_size = arena_size
        self.disentangle_keys = disentangle_keys
        self.encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            init_dilation=init_dilation,
        )
        self.decoder = ResidualDecoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            invariant_dim=invariant_dim,
            init_dilation=init_dilation,
        )
        if disentangle is not None:
            self.disentangle = nn.ModuleDict(disentangle)
        else:
            self.disentangle = nn.ModuleDict()

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def sampling(self, mu, L):
        eps = torch.randn_like(mu)
        return torch.matmul(L, eps[..., None]).squeeze().add_(mu)

    def encode(self, data):
        if self.arena_size is not None:
            # self.arena_size.to("cuda" if data["root"].is_cuda else "cpu")
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        mu, L = self.encoder(
            x_in.moveaxis(1, -1).view(-1, self.in_channels, self.window)
        )
        return mu, L

    def decode(self, mu):
        data_o = {}
        x_hat = self.decoder(mu).moveaxis(-1, 1)

        if self.arena_size is not None:
            x6d = x_hat[..., :-3]
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:]).reshape(
                mu.shape[0], self.window, 3
            )
        else:
            x6d = x_hat
            data_o["root"] = torch.zeros(mu.shape[0], self.window, 3).to(mu.device)

        data_o["x6d"] = x6d.reshape(mu.shape[0], self.window, -1, 6)

        return data_o

    def forward(self, data):
        if self.arena_size is not None:
            # self.arena_size.to("cuda" if data["root"].is_cuda else "cpu")
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        in_shape = x_in.shape
        data_o = {}
        data_o["mu"], data_o["L"] = self.encoder(
            x_in.moveaxis(1, -1).view(-1, self.in_channels, self.window)
        )
        z = self.sampling(data_o["mu"], data_o["L"]) if self.training else data_o["mu"]

        if self.invariant_dim > 0:
            z = torch.cat([z] + [data[k] for k in self.disentangle_keys], dim=-1)

        # Runni
        # if len(self.disentangle.keys()) > 0:
        data_o["disentangle"] = {
            k: dis(data_o["mu"]) for k, dis in self.disentangle.items()
        }

        x_hat = self.decoder(z).moveaxis(-1, 1).reshape(in_shape)

        if self.arena_size is not None:
            data_o["x6d"] = x_hat[..., :-3].reshape(data["x6d"].shape)
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:])

        return data_o


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
