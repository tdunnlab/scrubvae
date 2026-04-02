import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scrubvae.model.residual import CholeskyL

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


# class ChannelInvariantEncoder(nn.Module):
#     def __init__(
#         self,
#         channel_encoder: Conv1DEncoder,
#         num_heads: int = 2,
#         query_size: int | List[int] = 8,
#         chan_mix_last: bool = False,
#         num_points: int | None = None,
#         latent_dim: int = 64,
#         use_be: bool = False,
#     ):
#         super().__init__()
#         self.channel_encoder = channel_encoder
#         self.data_dim = channel_encoder.in_channels
#         self.chan_out_dim = channel_encoder.out_channels
#         self.n_blocks = channel_encoder.n_ds

#         if isinstance(query_size, int):
#             query_size = [query_size] * self.n_blocks
#         self.query_size = query_size

#         attention_blocks = []
#         for i in range(self.n_blocks):
#             block = self.channel_encoder.backbone[i]
#             chan_out_dim = block.ds.out_channels
#             attention_blocks.append(
#                 SetAttentiveBlock(
#                     dim_in=chan_out_dim,
#                     dim_out=chan_out_dim,
#                     num_heads=num_heads,
#                     num_inds=query_size[i],
#                 )
#             )
#         self.attention_blocks = nn.Sequential(*attention_blocks)

#         self.latent_dim = latent_dim
#         self.chan_mix_last = chan_mix_last and num_points is not None
#         if self.chan_mix_last:
#             n_channels_out = num_points * self.chan_out_dim
#         else:
#             n_channels_out = query_size[-1] * self.chan_out_dim
#         self.fc = nn.Linear(n_channels_out, latent_dim)

#         self.use_be = use_be
#         if use_be:
#             self.be = nn.Embedding(6, channel_encoder.hidden_dim)

#     def _forward(self, x: torch.Tensor, pe_indices=None, **kwargs) -> torch.Tensor:
#         """Process sets of data points in a channel-invariant manner

#         Args:
#             x (torch.Tensor): [B, n_points, D, T], D is the data dimension (e.g., 3 for 3D Cartesian coordinates)

#         Returns:
#             out: torch.Tensor: [B, n_points, D_out, T']
#             hidden_feats: List[torch.Tensor], [B*T', query_size, D]
#         """
#         bs, n_points, D_in, T = x.shape
#         x = x.reshape(bs * n_points, D_in, T)

#         query_feats = []
#         for i in range(self.n_blocks):
#             x = self.channel_encoder.backbone[i](x)  # --> [B*N, D, T//ds]

#             x = x.reshape(bs, n_points, *x.shape[1:])  # --> [B, N, D, T//ds=T']
#             x = x.permute(0, 3, 1, 2).flatten(0, 1)  # --> [B*T', N, D]

#             x, h, attn1, attn2 = self.attention_blocks[i](x)

#             query_feats.append(
#                 h.reshape(bs, -1, *h.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
#             )  # --> [B*M, D, T']
#             x = (
#                 x.reshape(bs, -1, *x.shape[1:]).permute(0, 2, 3, 1).flatten(0, 1)
#             )  # [B*T', D, N] --> [B*N, D, T']

#             if i == 0 and self.use_be and pe_indices is not None:
#                 pe = self.be(pe_indices.to(x.device)).unsqueeze(-1)
#                 x = x.reshape(bs, -1, *x.shape[1:])
#                 x += pe
#                 x = x.flatten(0, 1)

#         out = x if self.chan_mix_last else query_feats[-1]
#         out = self.channel_encoder.out(out)
#         out = out.reshape(bs, -1, *out.shape[1:])  # [B, N, D_out, T']
#         out = out.flatten(1, 2)
#         return out

#     def forward(self, x: torch.Tensor, pe_indices=None, **kwargs):
#         out = self._forward(x, pe_indices, **kwargs)  # --> [B, N, D_out, T']

#         out = self.fc(out.permute(0, 2, 1))  # --> [B, T', D_latent]
#         out = out.permute(0, 2, 1)

#         return out
    

