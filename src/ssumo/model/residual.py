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
                bias=True,
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
                bias=True,
            ),
        )

        self.skip = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            stride,
            kernel // 2,
            dilation=dilation,
            bias=True,
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
                bias=True,
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
                bias=True,
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
                bias=True,
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
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        prior="gaussian",
        init_dilation=None,
    ):
        super(ResidualEncoder, self).__init__()
        self.prior = prior
        self.conv_in = nn.Conv1d(in_channels, ch[0], 7, 1, 3)
        self.activation = nn.Tanh() if activation == "tanh" else nn.PReLU()

        if init_dilation is None:
            dilation = torch.ones(len(ch) - 1, dtype=int)
        else:
            dilation = init_dilation * 2 ** torch.arange(len(ch) - 1)

        layers = []
        for i in range(len(ch) - 1):
            layers += [
                ResidualBlock(ch[i], ch[i + 1], kernel, activation, dilation[i].item())
            ]
        self.res_layers = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        flatten_dim = find_latent_dim(window, kernel, len(ch) - 1, dilation) * ch[-1]

        if prior == "gaussian":
            sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
            self.fc_mu = nn.Linear(flatten_dim, z_dim)
            self.fc_sigma = nn.Sequential(
                nn.Linear(flatten_dim, sig_dim), CholeskyL(z_dim, is_diag)
            )
        elif prior == "beta":
            self.fc_alpha = nn.Linear(flatten_dim, z_dim)
            self.fc_beta = nn.Linear(flatten_dim, z_dim)

    def forward(self, x):
        x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        x = self.flatten(x)
        if self.prior == "gaussian":
            mu = self.fc_mu(x)
            sigma = self.fc_sigma(x)
            return mu, sigma
        elif self.prior == "beta":
            alpha = F.softplus(self.fc_alpha(x)) + 1
            beta = F.softplus(self.fc_beta(x)) + 1
            return alpha, beta
        return 0


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        conditional_dim=0,
        init_dilation=None,
    ):
        super(ResidualDecoder, self).__init__()
        self.conditional_dim = conditional_dim

        if init_dilation is None:
            dilation = torch.ones(len(ch) - 1, dtype=int)
        else:
            dilation = init_dilation * 2 ** torch.arange(len(ch) - 1)

        flatten_dim = find_latent_dim(window, kernel, len(ch) - 1, dilation) * ch[-1]
        self.fc_in = nn.Linear(z_dim + conditional_dim, flatten_dim)
        self.unflatten = nn.Unflatten(1, (ch[-1], -1))
        # self.conv_in = nn.ConvTranspose1d(int(flatten_dim[0]), ch*16, 3, 1, 1)

        layers = []
        for i in range(1, len(ch)):
            layers += [
                ResidualBlockTranspose(
                    ch[-i],
                    ch[-i - 1],
                    kernel,
                    activation=activation,
                    dilation=dilation[-i].item(),
                )
            ]
        self.res_layers = nn.Sequential(*layers)

        l_out = find_out_dim(
            find_latent_dim(window, kernel, len(ch) - 1), kernel, len(ch) - 1
        )

        final_kernel = window - l_out + 7
        print("Final ConvOut Kernel: {}".format(final_kernel))
        self.conv_out = nn.ConvTranspose1d(ch[0], out_channels, final_kernel, 1, 3)
        # self.activation = nn.Tanh()

    def forward(self, x):
        x = self.unflatten(self.fc_in(x))
        # x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        x = torch.tanh(self.conv_out(x))
        return x


class VAE(nn.Module):
    def __init__(self, prior="gaussian"):
        super(VAE, self).__init__()
        self.prior = prior
        if prior == "gaussian":
            self.dist_params = ["mu","L"]
        elif prior == "beta":
            self.dist_params = ["alpha","beta"]
        return self

    def sampling(self, mu, L):
        """Reparameterization trick

        Parameters
        ----------
        mu : torch.tensor
            Batch_size x latent dimensions
        L : torch.tensor
            Batch_size x latent dimensions x latent dimensions. Lower triangular or diagonal matrix.
        """
        eps = torch.randn_like(mu)
        return torch.matmul(L, eps[..., None]).squeeze().add_(mu)

    def forward(self, data):
        data_o = self.encode(data)
        if self.prior == "gaussian":
            z = self.sampling(data_o["mu"], data_o["L"]) if self.training else data_o["mu"]
        elif self.prior == "beta":
            beta_dist = torch.distributions.Beta(data_o["alpha"], data_o["beta"])
            data_o["beta_dist"] = beta_dist
            z = beta_dist.rsample()*2-1
        data_o["z"] = z

        # Running disentangle
        data_o["disentangle"] = {}
        if "linear" in self.disentangle.keys():
            data_o["disentangle"]["linear"] = {
                k: model(data_o["mu"])
                for k, model in self.disentangle["linear"].items()
            }

        for method, module_dict in self.disentangle.items():
            if method == "linear":
                continue
            else:
                data_o["disentangle"][method] = {}
                for k, model in module_dict.items():
                    if "linear" in self.disentangle.keys():
                        latent = data_o["disentangle"]["linear"][k]["z_null"]
                    else:
                        latent = data_o["mu"]
                    data_o["disentangle"][method][k] = model(latent)

            # data_o["disentangle"][method] = {k: model(latent) for k, model in module_dict.items()}
        #     k: dis(data_o["mu"]) for k, dis in self.disentangle.items()
        # }

        data_o.update(self.decode(z, data))

        return data_o


class ResVAE(VAE):
    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        conditional_dim=0,
        init_dilation=None,
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
        conditional_keys=None,
        discrete_classes=None,
        prior="gaussian",
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.ch = ch
        self.window = window
        self.is_diag = is_diag
        self.conditional_dim = conditional_dim
        self.kinematic_tree = kinematic_tree
        self.register_buffer("arena_size", arena_size)
        self.disentangle_keys = disentangle_keys
        self.conditional_keys = conditional_keys
        self.discrete_classes = discrete_classes
        self.encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=prior,
            init_dilation=init_dilation,
        )
        self.decoder = ResidualDecoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            conditional_dim=conditional_dim,
            init_dilation=init_dilation,
        )
        if disentangle is not None:
            self.disentangle = nn.ModuleDict()
            for k, v in disentangle.items():
                self.disentangle[k] = nn.ModuleDict(v)
        else:
            self.disentangle = nn.ModuleDict(nn.ModuleDict())

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def encode(self, data):
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(
            x_in.moveaxis(1, -1).view(-1, self.in_channels, self.window)
        )

        if self.prior == "beta":
            data_o["mu"] = (data_o["alpha"]-1)/(data_o["alpha"] + data_o["beta"] - 2)*2-1
        return data_o

    def decode(self, z, data):
        data_o = {}
        if self.conditional_dim > 0:
            conditional_vars = [
                (
                    F.one_hot(data[k].ravel().long(), len(self.discrete_classes[k]))
                    if k in self.discrete_classes.keys()
                    else data[k]
                )
                for k in self.conditional_keys
            ]
            z = torch.cat(
                [z] + conditional_vars,
                dim=-1,
            )

        x_hat = self.decoder(z).moveaxis(-1, 1)

        if self.arena_size is None:
            x6d = x_hat
        else:
            x6d = x_hat[..., :-3]
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:]).reshape(
                z.shape[0], self.window, 3
            )

        data_o["x6d"] = x6d.reshape(z.shape[0], self.window, -1, 6)

        return data_o
