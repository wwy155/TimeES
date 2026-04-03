from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch
import torch.nn as nn
from einops import rearrange


@dataclass
class SKOLRConfig:
    enc_in: int
    seq_len: int
    pred_len: int

    seg_len: int = 48
    num_blocks: int = 3
    dynamic_dim: int = 128
    hidden_dim: int = 64
    hidden_layers: int = 2
    dropout: float = 0.05

    # structured koopman linear RNN options
    CI: bool = False
    shareEncoder: bool = False
    mask_type: str = "global"  # global | channel | FixHalf
    inv_loss: int = 0


class FourierFilter(nn.Module):
    """
    Fourier Filter with Learnable Masks.
    Supports global (shared across channels) or channel-wise masks.
    """

    def __init__(
        self,
        num_frequencies: int,
        num_channels: int,
        num_blocks: int,
        mask_type: str = "global",
    ):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.num_channels = int(num_channels)
        self.num_blocks = int(num_blocks)
        self.mask_type = str(mask_type)

        if self.mask_type == "global":
            self.mask_weights = nn.Parameter(torch.rand(self.num_blocks, self.num_frequencies))
        elif self.mask_type == "channel":
            self.mask_weights = nn.Parameter(
                torch.rand(self.num_blocks, self.num_frequencies, self.num_channels)
            )
        elif self.mask_type == "FixHalf":
            self.register_buffer(
                "mask_weights",
                torch.full((self.num_blocks, self.num_frequencies, self.num_channels), 1 / self.num_blocks),
                persistent=False,
            )
        else:
            raise ValueError("mask_type must be one of: global, channel, FixHalf")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [B, L, C]
        xf = torch.fft.rfft(x, dim=1)  # [B, F, C]

        if self.mask_type == "global":
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1).unsqueeze(-1)  # [K,1,F,1]
        elif self.mask_type == "channel":
            mask = torch.sigmoid(self.mask_weights).unsqueeze(1)  # [K,1,F,C]
        else:  # FixHalf
            mask = self.mask_weights.unsqueeze(1).to(xf.device)  # [K,1,F,C]

        masked_xf = xf.unsqueeze(0) * mask  # [K,B,F,C]
        return [torch.fft.irfft(masked_xf[i], dim=1) for i in range(self.num_blocks)]


class MLP(nn.Module):
    def __init__(
        self,
        f_in: int,
        f_out: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.05,
        activation: str = "relu",
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


class LinearRNN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.Wxh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Whh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor, pred_len: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, H]
        B, L, H = x.shape
        h_t = torch.zeros(B, H, device=x.device, dtype=x.dtype)

        rec = []
        for t in range(L):
            h_t = self.Wxh(x[:, t, :]) + self.Whh(h_t)
            rec.append(h_t.unsqueeze(1))
        rec_t = torch.cat(rec, dim=1)

        outputs = []
        for _ in range(int(pred_len)):
            h_t = self.Whh(h_t)
            outputs.append(h_t.unsqueeze(1))
        out_t = self.layer_norm(torch.cat(outputs, dim=1))
        return rec_t, out_t


class TimeVarKP(nn.Module):
    def __init__(
        self,
        enc_in: int,
        input_len: int,
        pred_len: int,
        seg_len: int,
        dynamic_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        CI: bool = False,
        inv_loss: bool = False,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.seg_len = int(seg_len)
        self.dynamic_dim = int(dynamic_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.freq = int(math.ceil(self.input_len / self.seg_len))
        self.step = int(math.ceil(self.pred_len / self.seg_len))
        self.padding_len = int(self.seg_len * self.freq - self.input_len)
        self.linearRNN = LinearRNN(self.dynamic_dim)
        self.CI = bool(CI)
        self.inv_loss = bool(inv_loss)

    def forward(
        self, x: torch.Tensor
    ) -> Union[Tuple[None, torch.Tensor], Tuple[None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        # x: [B, L, C]
        B, L, C = x.shape
        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)
        res_chunks = res.chunk(self.freq, dim=1)

        if self.CI:
            res_in = rearrange(torch.stack(res_chunks, dim=1), "b f p c -> (b c) f p")
        else:
            res_in = rearrange(torch.stack(res_chunks, dim=1), "b f p c -> b f (p c)")

        inv_in = res_in
        x_enc = self.encoder(res_in)
        _, x_pred = self.linearRNN(x_enc, self.step)

        x_rec = None  # save compute; not used by upstream

        x_pred = self.decoder(x_pred)
        if self.CI:
            x_pred = rearrange(x_pred, "(b c) s p -> b (s p) c", c=C)
        else:
            x_pred = rearrange(x_pred, "b s (p c) -> b (s p) c", c=C)

        x_pred = x_pred[:, : self.pred_len, :]

        if self.inv_loss:
            inv_out = self.decoder(x_enc)
            return x_rec, x_pred, (inv_in, inv_out)

        return x_rec, x_pred


class KoopRNN_Block(nn.Module):
    def __init__(
        self,
        enc_in: int,
        input_len: int,
        pred_len: int,
        seg_len: int,
        dynamic_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
        CI: bool,
        inv_loss: bool,
        share_encoder: bool,
    ):
        super().__init__()
        self.inv_loss = bool(inv_loss)

        if share_encoder:
            if CI:
                encoder = MLP(
                    f_in=seg_len,
                    f_out=dynamic_dim,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
                decoder = MLP(
                    f_in=dynamic_dim,
                    f_out=seg_len,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            else:
                encoder = MLP(
                    f_in=seg_len * enc_in,
                    f_out=dynamic_dim,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
                decoder = MLP(
                    f_in=dynamic_dim,
                    f_out=seg_len * enc_in,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
        else:
            # per-block enc/dec
            if CI:
                encoder = MLP(
                    f_in=seg_len,
                    f_out=dynamic_dim,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
                decoder = MLP(
                    f_in=dynamic_dim,
                    f_out=seg_len,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            else:
                encoder = MLP(
                    f_in=seg_len * enc_in,
                    f_out=dynamic_dim,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
                decoder = MLP(
                    f_in=dynamic_dim,
                    f_out=seg_len * enc_in,
                    activation="relu",
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )

        self.block = TimeVarKP(
            enc_in=enc_in,
            input_len=input_len,
            pred_len=pred_len,
            seg_len=seg_len,
            dynamic_dim=dynamic_dim,
            encoder=encoder,
            decoder=decoder,
            CI=CI,
            inv_loss=inv_loss,
        )

    def forward(self, x: torch.Tensor):
        return self.block(x)


class SKOLR(nn.Module):
    """
    Vendorized from networkslab/SKOLR (ICML'25) implementation.
    """

    def __init__(
        self,
        *,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        seg_len: int = 48,
        num_blocks: int = 3,
        dynamic_dim: int = 128,
        hidden_dim: int = 64,
        hidden_layers: int = 2,
        dropout: float = 0.05,
        CI: bool = False,
        shareEncoder: bool = False,
        mask_type: str = "global",
        inv_loss: int = 0,
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
        self.dropout = float(dropout)
        self.CI = bool(CI)
        self.shareEncoder = bool(shareEncoder)
        self.mask_type = str(mask_type)
        self.inv_loss = int(inv_loss) > 0

        self.disentanglement = FourierFilter(
            num_frequencies=int(self.input_len / 2) + 1,
            num_channels=self.enc_in,
            num_blocks=self.num_blocks,
            mask_type=self.mask_type,
        )

        self.blocks = nn.ModuleList(
            [
                KoopRNN_Block(
                    enc_in=self.enc_in,
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    seg_len=self.seg_len,
                    dynamic_dim=self.dynamic_dim,
                    hidden_dim=self.hidden_dim,
                    hidden_layers=self.hidden_layers,
                    dropout=self.dropout,
                    CI=self.CI,
                    inv_loss=self.inv_loss,
                    share_encoder=self.shareEncoder,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        ifft_results = self.disentanglement(x_enc)
        results = [self.blocks[i](ifft_results[i]) for i in range(self.num_blocks)]
        x_pred_list = [res[1] for res in results]
        combined_x_pred = x_pred_list[0]
        for i in range(1, len(x_pred_list)):
            combined_x_pred = combined_x_pred + x_pred_list[i]

        pred = combined_x_pred * std_enc + mean_enc
        return pred


class Model(SKOLR):
    """
    Backward compatible wrapper: Model(configs).
    """

    def __init__(self, configs):
        cfg = SKOLRConfig(
            enc_in=int(getattr(configs, "enc_in")),
            seq_len=int(getattr(configs, "seq_len")),
            pred_len=int(getattr(configs, "pred_len")),
            seg_len=int(getattr(configs, "seg_len", 48)),
            num_blocks=int(getattr(configs, "num_blocks", 3)),
            dynamic_dim=int(getattr(configs, "dynamic_dim", 128)),
            hidden_dim=int(getattr(configs, "hidden_dim", 64)),
            hidden_layers=int(getattr(configs, "hidden_layers", 2)),
            dropout=float(getattr(configs, "dropout", 0.05)),
            CI=bool(getattr(configs, "CI", False)),
            shareEncoder=bool(getattr(configs, "shareEncoder", False)),
            mask_type=str(getattr(configs, "mask_type", "global")),
            inv_loss=int(getattr(configs, "inv_loss", 0)),
        )
        super().__init__(**cfg.__dict__)

