from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class KoopaConfig:
    enc_in: int
    seq_len: int
    pred_len: int

    # koopa hyper-params (from thuml/Koopa defaults)
    dynamic_dim: int = 128
    hidden_dim: int = 64
    hidden_layers: int = 2
    seg_len: int = 48
    num_blocks: int = 3
    alpha: float = 0.2
    multistep: bool = False
    dropout: float = 0.05

    # computed mask spectrum can be supplied by caller; if None, fall back to top-k indices of rfft bins
    mask_spectrum: Optional[torch.Tensor] = None


class FourierFilter(nn.Module):
    def __init__(self, mask_spectrum: torch.Tensor):
        super().__init__()
        self.register_buffer("mask_spectrum", mask_spectrum, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var
        return x_var, x_inv


class MLP(nn.Module):
    def __init__(
        self,
        f_in: int,
        f_out: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.05,
        activation: str = "tanh",
    ):
        super().__init__()
        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise NotImplementedError(f"activation={activation}")

        layers: list[nn.Module] = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(max(0, hidden_layers - 2)):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.K: Optional[torch.Tensor] = None

    def one_step_forward(self, z: torch.Tensor, return_rec: bool = False):
        B, input_len, E = z.shape
        if input_len <= 1:
            raise ValueError("snapshots number should be larger than 1")

        x, y = z[:, :-1], z[:, 1:]
        K = torch.linalg.lstsq(x, y).solution  # [B,E,E]
        if torch.isnan(K).any():
            K = torch.eye(K.shape[1], device=K.device, dtype=K.dtype).unsqueeze(0).repeat(B, 1, 1)
        self.K = K

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred
        return z_pred

    def forward(self, z: torch.Tensor, pred_len: int = 1):
        if pred_len < 1:
            raise ValueError("prediction length should not be less than 1")
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for _ in range(1, int(pred_len)):
            z_pred = torch.bmm(z_pred, self.K)  # type: ignore[arg-type]
            z_preds.append(z_pred)
        return z_rec, torch.cat(z_preds, dim=1)


class KPLayerApprox(nn.Module):
    def __init__(self):
        super().__init__()
        self.K: Optional[torch.Tensor] = None
        self.K_step: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor, pred_len: int = 1):
        B, input_len, E = z.shape
        if input_len <= 1:
            raise ValueError("snapshots number should be larger than 1")

        x, y = z[:, :-1], z[:, 1:]
        K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(K).any():
            K = torch.eye(K.shape[1], device=K.device, dtype=K.dtype).unsqueeze(0).repeat(B, 1, 1)
        self.K = K

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)

        pred_len = int(pred_len)
        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E, device=z.device, dtype=z.dtype).unsqueeze(0).repeat(B, 1, 1)
            temp_z_pred, all_pred = z, []
            for _ in range(int(math.ceil(pred_len / input_len))):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
        return z_rec, z_pred


class TimeVarKP(nn.Module):
    def __init__(
        self,
        *,
        enc_in: int,
        input_len: int,
        pred_len: int,
        seg_len: int,
        dynamic_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        multistep: bool = False,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.seg_len = int(seg_len)
        self.dynamic_dim = int(dynamic_dim)
        self.multistep = bool(multistep)
        self.encoder = encoder
        self.decoder = decoder
        self.freq = int(math.ceil(self.input_len / self.seg_len))
        self.step = int(math.ceil(self.pred_len / self.seg_len))
        self.padding_len = int(self.seg_len * self.freq - self.input_len)
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x: torch.Tensor):
        B, L, C = x.shape
        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)

        res = self.encoder(res)
        x_rec, x_pred = self.dynamics(res, self.step)

        x_rec = self.decoder(x_rec).reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, : self.input_len, :]

        x_pred = self.decoder(x_pred).reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, : self.pred_len, :]
        return x_rec, x_pred


class TimeInvKP(nn.Module):
    def __init__(
        self,
        *,
        input_len: int,
        pred_len: int,
        dynamic_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x.transpose(1, 2)
        res = self.encoder(res)
        res = self.K(res)
        res = self.decoder(res)
        return res.transpose(1, 2)


class Koopa(nn.Module):
    def __init__(
        self,
        *,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        dynamic_dim: int = 128,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        seg_len: int = 48,
        num_blocks: int = 3,
        alpha: float = 0.2,
        multistep: bool = False,
        dropout: float = 0.05,
        mask_spectrum: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.enc_in = int(enc_in)
        self.input_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.seg_len = int(seg_len)
        self.num_blocks = int(num_blocks)
        self.dynamic_dim = int(dynamic_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.multistep = bool(multistep)
        self.alpha = float(alpha)
        self.dropout = float(dropout)

        # If caller doesn't provide mask_spectrum, create a deterministic default:
        # use top-k lowest frequencies (excluding DC) as "invariant" spectrum mask.
        if mask_spectrum is None:
            k = max(1, int((int(self.input_len / 2) + 1) * self.alpha))
            # avoid index 0 (DC)
            idx = torch.arange(1, int(self.input_len / 2) + 1)
            self.register_buffer("mask_spectrum", idx[:k].long(), persistent=False)
        else:
            self.register_buffer("mask_spectrum", mask_spectrum.long(), persistent=False)

        self.disentanglement = FourierFilter(self.mask_spectrum)

        self.time_inv_encoder = MLP(
            f_in=self.input_len,
            f_out=self.dynamic_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.pred_len,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.time_inv_kps = nn.ModuleList(
            [
                TimeInvKP(
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    dynamic_dim=self.dynamic_dim,
                    encoder=self.time_inv_encoder,
                    decoder=self.time_inv_decoder,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.time_var_encoder = MLP(
            f_in=self.seg_len * self.enc_in,
            f_out=self.dynamic_dim,
            activation="tanh",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.time_var_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.seg_len * self.enc_in,
            activation="tanh",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        self.time_var_kps = nn.ModuleList(
            [
                TimeVarKP(
                    enc_in=self.enc_in,
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    seg_len=self.seg_len,
                    dynamic_dim=self.dynamic_dim,
                    encoder=self.time_var_encoder,
                    decoder=self.time_var_decoder,
                    multistep=self.multistep,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](time_var_input)
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = time_inv_output + time_var_output
            else:
                forecast = forecast + (time_inv_output + time_var_output)

        return forecast * std_enc + mean_enc


class Model(Koopa):
    def __init__(self, configs):
        super().__init__(
            enc_in=int(getattr(configs, "enc_in")),
            seq_len=int(getattr(configs, "seq_len")),
            pred_len=int(getattr(configs, "pred_len")),
            dynamic_dim=int(getattr(configs, "dynamic_dim", 128)),
            hidden_dim=int(getattr(configs, "hidden_dim", 64)),
            hidden_layers=int(getattr(configs, "hidden_layers", 2)),
            seg_len=int(getattr(configs, "seg_len", 48)),
            num_blocks=int(getattr(configs, "num_blocks", 3)),
            alpha=float(getattr(configs, "alpha", 0.2)),
            multistep=bool(getattr(configs, "multistep", False)),
            dropout=float(getattr(configs, "dropout", 0.05)),
            mask_spectrum=getattr(configs, "mask_spectrum", None),
        )

