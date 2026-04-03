from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


class RevIN(nn.Module):
    """
    Reversible instance normalization (minimal, KNF-style).
    Normalizes over time dimension (dim=1) by default.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False, axis=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.axis = axis
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self.register_buffer("_mean", torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer("_stdev", torch.ones(1, 1, num_features), persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = torch.mean(x, dim=self.axis, keepdim=True)
            self._stdev = torch.sqrt(torch.var(x, dim=self.axis, keepdim=True, unbiased=False) + self.eps)
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self._stdev + self._mean
        raise NotImplementedError(mode)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.0,
        use_instancenorm: bool = False,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_instancenorm:
            layers += [nn.InstanceNorm1d(hidden_dim)]
        layers += [nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            if use_instancenorm:
                layers += [nn.InstanceNorm1d(hidden_dim)]
            layers += [nn.ReLU(), nn.Dropout(dropout_rate)]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class KNFConfig:
    # input shaping
    input_dim: int = 1  # KNF "input_dim" (grouping factor), must divide seq_len

    # model core
    latent_dim: int = 64
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    encoder_num_layers: int = 5
    decoder_num_layers: int = 5

    # measurement functions
    num_poly: int = 3
    num_exp: int = 1
    num_sins: int = -1  # default: input_length//2 - 1

    # operators
    add_global_operator: bool = True
    add_control: bool = True
    control_hidden_dim: int = 64
    control_num_layers: int = 3

    # transformer for local operator
    num_heads: int = 1
    transformer_dim: int = 128
    transformer_num_layers: int = 3

    # normalization & regularization
    use_revin: bool = True
    use_instancenorm: bool = False
    regularize_rank: bool = False
    dropout_rate: float = 0.0


class KNF(nn.Module):
    """
    Deterministic Koopman Neural Forecaster (KNF), adapted from
    google-research/KNF for this repo.

    forward signature matches other models:
      forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, N]
    """

    def __init__(self, seq_len: int, pred_len: int, num_feats: int, cfg: KNFConfig):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.num_feats = int(num_feats)
        self.cfg = cfg

        if self.seq_len % self.cfg.input_dim != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by input_dim={self.cfg.input_dim}")

        self.input_length = self.seq_len
        self.L = self.input_length // self.cfg.input_dim  # number of grouped steps

        if self.cfg.num_sins == -1:
            num_sins = self.input_length // 2 - 1
        else:
            num_sins = self.cfg.num_sins
        # The original KNF indexing scheme requires that:
        # - embedding indices up to (num_poly + num_exp + 2*num_sins - 1) fit in latent_dim
        # - coefficient indices up to (3*num_sins + (num_poly+num_exp+num_sins-1)) fit in (latent_dim + 2*num_sins)
        # A sufficient condition is: num_sins <= (latent_dim - num_poly - num_exp)//2.
        max_num_sins = max(0, (int(self.cfg.latent_dim) - int(self.cfg.num_poly) - int(self.cfg.num_exp)) // 2)
        self.num_sins = int(min(int(num_sins), int(max_num_sins)))
        self.num_poly = int(self.cfg.num_poly)
        self.num_exp = int(self.cfg.num_exp)

        self.len_interas = len(list(itertools.combinations(np.arange(0, self.num_feats), 2))) if self.num_feats > 1 else 0

        if self.cfg.use_revin:
            self.normalizer = RevIN(num_features=self.num_feats, axis=1)

        # encoder learns coefficients for measurement functions
        enc_out_dim = (self.cfg.latent_dim + self.num_sins * 2) * self.cfg.input_dim * self.num_feats
        self.encoder = MLP(
            input_dim=self.cfg.input_dim * self.num_feats,
            output_dim=enc_out_dim,
            hidden_dim=self.cfg.encoder_hidden_dim,
            num_layers=self.cfg.encoder_num_layers,
            use_instancenorm=self.cfg.use_instancenorm,
            dropout_rate=self.cfg.dropout_rate,
        )

        # optional global operator
        embed_dim = self.cfg.latent_dim * self.num_feats + self.len_interas
        if self.cfg.add_global_operator:
            self.global_linear_transform = nn.Linear(embed_dim, embed_dim, bias=False)

        # local operator via transformer/attention
        d_model = self.input_length // self.cfg.input_dim
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.cfg.num_heads, dim_feedforward=self.cfg.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.transformer_num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.cfg.num_heads, batch_first=True)

        if self.cfg.add_control:
            self.control = MLP(
                input_dim=(self.input_length - self.cfg.input_dim) * self.num_feats,
                output_dim=embed_dim,
                hidden_dim=self.cfg.control_hidden_dim,
                num_layers=self.cfg.control_num_layers,
                use_instancenorm=self.cfg.use_instancenorm,
                dropout_rate=self.cfg.dropout_rate,
            )

        # decoder reconstructs observations
        self.decoder = MLP(
            input_dim=embed_dim,
            output_dim=self.num_feats,
            hidden_dim=self.cfg.decoder_hidden_dim,
            num_layers=self.cfg.decoder_num_layers,
            use_instancenorm=self.cfg.use_instancenorm,
            dropout_rate=self.cfg.dropout_rate,
        )

    def _single_forward(self, inps: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        inps: [B, L, input_dim*num_feats]
        returns: forecast [B, num_steps, num_feats]
        """
        B = inps.shape[0]

        encoder_outs = self.encoder(inps)
        encoder_outs = encoder_outs.reshape(B, self.L, (self.cfg.latent_dim + self.num_sins * 2), self.cfg.input_dim, self.num_feats)
        inps_g = inps.reshape(B, self.L, self.cfg.input_dim, self.num_feats)

        coefs = torch.einsum("blkdf, bldf -> blfk", encoder_outs, inps_g)  # [B,L,F,K]

        embedding = torch.zeros(B, self.L, self.num_feats, self.cfg.latent_dim, device=inps.device)
        for f in range(self.num_feats):
            # polynomials
            for i in range(self.num_poly):
                embedding[:, :, f, i] = coefs[:, :, f, i] ** (i + 1)
            # exponential
            for i in range(self.num_poly, self.num_poly + self.num_exp):
                embedding[:, :, f, i] = torch.exp(coefs[:, :, f, i])
            # sin/cos
            for i in range(self.num_poly + self.num_exp, self.num_poly + self.num_exp + self.num_sins):
                embedding[:, :, f, i] = coefs[:, :, f, self.num_sins * 2 + i] * torch.cos(coefs[:, :, f, i])
                embedding[:, :, f, self.num_sins + i] = coefs[:, :, f, self.num_sins * 3 + i] * torch.sin(
                    coefs[:, :, f, self.num_sins + i]
                )
            # remaining learned measurements
            embedding[:, :, f, self.num_poly + self.num_exp + self.num_sins * 2 :] = coefs[
                :, :, f, self.num_poly + self.num_exp + self.num_sins * 4 :
            ]

        embedding = embedding.reshape(B, self.L, -1)
        if self.num_feats > 1:
            inter_lsts = list(itertools.combinations(np.arange(0, self.num_feats), 2))
            embedding_inter = torch.zeros(B, self.L, len(inter_lsts), device=inps.device)
            for i, item in enumerate(inter_lsts):
                embedding_inter[..., i] = coefs[:, :, item[0], 0] * coefs[:, :, item[1], 0]
            embedding = torch.cat([embedding, embedding_inter], dim=-1)

        # local operator
        trans_out = self.transformer_encoder(embedding.transpose(1, 2))
        local_transform = self.attention(trans_out, trans_out, trans_out)[1]  # [B, E, E]

        # predict on lookback (for control)
        inp_embed_preds = []
        forw = embedding[:, :1]
        for i in range(self.L - 1):
            if self.cfg.add_global_operator:
                forw = self.global_linear_transform(forw)
            forw = torch.einsum("bnl, blh -> bnh", forw, local_transform)
            inp_embed_preds.append(forw)
        embed_preds = torch.cat(inp_embed_preds, dim=1) if inp_embed_preds else embedding[:, :0]
        inp_preds = self.decoder(embed_preds) if inp_embed_preds else torch.zeros(B, 0, self.num_feats, device=inps.device)

        # optional control adjustment
        if self.cfg.add_control and inp_embed_preds:
            pred_diff = inp_preds.reshape(B, -1) - inps_g[:, 1:].reshape(B, -1)
            linear_adj = self.control(pred_diff)
            linear_adj = torch.stack([torch.diagflat(linear_adj[i]) for i in range(B)])
        else:
            linear_adj = None

        # forward predictions
        forw_preds = []
        forward_iters = num_steps // self.cfg.input_dim
        if num_steps % self.cfg.input_dim > 0:
            forward_iters += 1
        for _ in range(forward_iters):
            if self.cfg.add_global_operator:
                forw = self.global_linear_transform(forw)
            if linear_adj is not None:
                forw = torch.einsum("bnl, blh -> bnh", forw, local_transform + linear_adj)
            else:
                forw = torch.einsum("bnl, blh -> bnh", forw, local_transform)
            forw_preds.append(forw)
        forw_preds = torch.cat(forw_preds, dim=1)
        forw_preds = self.decoder(forw_preds)
        forw_preds = forw_preds.reshape(B, -1, self.num_feats)[:, :num_steps]
        return forw_preds

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, T, N]
        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B,T,N], got {tuple(x_enc.shape)}")
        B, T, N = x_enc.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")
        if N != self.num_feats:
            raise ValueError(f"Expected num_feats={self.num_feats}, got {N}")

        inps = x_enc
        if self.cfg.use_revin:
            inps = self.normalizer(inps, mode="norm")

        # reshape to [B, L, input_dim*num_feats]
        inps = inps.reshape(B, -1, self.cfg.input_dim, self.num_feats).reshape(B, -1, self.cfg.input_dim * self.num_feats)
        preds = self._single_forward(inps, self.pred_len)

        if self.cfg.use_revin:
            preds = self.normalizer(preds, mode="denorm")
        return preds

