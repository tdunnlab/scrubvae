import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssumo.model.ResVAE import CholeskyL

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        z_dim=128,
        window=50,
        activation="gelu",
        n_heads=4,
        ff_size=512,
        n_layers=4,
        is_diag=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.activation = activation
        self.window = window

        self.pose_embedding = nn.Linear(in_channels, z_dim)
        self.pos_encoder = PositionalEncoding(z_dim, dropout=0.1)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=n_heads,
            dim_feedforward=ff_size,
            dropout=0.1,
            activation=activation,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=n_layers
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(z_dim * 50, z_dim)
        sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
        self.fc_sigma = nn.Sequential(
            nn.Linear(z_dim * 50, sig_dim), CholeskyL(z_dim, is_diag)
        )

    def forward(self, x):
        input = x.permute((2, 0, 1))
        input = self.pos_encoder(self.pose_embedding(input))
        input = self.transformer_encoder(input)
        out = self.flatten(input.permute((1, 0, 2)))
        mu = self.fc_mu(out)
        L = self.fc_sigma(out)
        return mu, L


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        out_channels,
        z_dim=128,
        window=50,
        activation="gelu",
        n_heads=4,
        ff_size=512,
        n_layers=4,
    ):
        super(TransformerDecoder, self).__init__()
        self.out_channels = out_channels
        self.window = window
        self.z_dim = z_dim
        self.activation = activation

        self.pos_encoder = PositionalEncoding(z_dim, dropout=0.1)
        transformer_layer = nn.TransformerDecoderLayer(
            d_model=z_dim,
            nhead=n_heads,
            dim_feedforward=ff_size,
            dropout=0.1,
            activation=activation,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            transformer_layer, num_layers=n_layers
        )

        self.fc_out = nn.Linear(z_dim, out_channels)

    def forward(self, z):
        time = torch.zeros(self.window, z.shape[0], self.z_dim, device=z.device)
        time = self.pos_encoder(time)
        out = self.transformer_decoder(tgt=time, memory=z[None,...])
        out = torch.tanh(self.fc_out(out))

        return out.permute(1,2,0)


class TransformerVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        z_dim=128,
        window=50,
        activation="gelu",
        n_heads=4,
        ff_size=512,
        n_layers=4,
        is_diag=False,
    ):
        super(TransformerVAE, self).__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.activation = activation
        self.window = window

        self.encoder = TransformerEncoder(
            in_channels = in_channels,
            z_dim=z_dim,
            window=window,
            activation=activation,
            n_heads=n_heads,
            ff_size=ff_size,
            n_layers=n_layers,
            is_diag=is_diag,
        )

        self.decoder = TransformerDecoder(
            out_channels = in_channels,
            z_dim=z_dim,
            window=window,
            activation=activation,
            n_heads=n_heads,
            ff_size=ff_size,
            n_layers=n_layers,
        )

    def reparameterize(self, mu, L):
        eps = torch.randn_like(mu)
        return torch.matmul(L, eps[..., None]).squeeze().add_(mu)

    def forward(self, x):
        mu, L = self.encoder(x)
        z = self.reparameterize(mu, L)
        x_hat = self.decoder(z)
        return x_hat, mu, L
