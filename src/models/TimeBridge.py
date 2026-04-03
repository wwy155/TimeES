import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, d_model, dropout, num_p=1):
        super(PatchEmbed, self).__init__()
        self.num_p = num_p
        self.patch = seq_len // self.num_p
        self.d_model = d_model

        self.proj = nn.Sequential(
            nn.Linear(self.patch, self.d_model, False),
            nn.Dropout(dropout)
        )

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = self.proj(x.reshape(*x.shape[:-1], self.num_p, self.patch))
        return x

class TimeBridge(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        period=24,
        num_p=None,
        d_model=128,
        n_heads=8,
        d_ff=128,
        ia_layers=3,
        pd_layers=1,
        ca_layers=0,
        stable_len=6,
        dropout=0.0,
        attn_dropout=0.15,
        activation="gelu",
        revin=True,
    ):
        super().__init__()

        self.revin = revin  # kept for API parity (not used in current forward)

        self.c_in = enc_in
        self.period = period
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_p = self.seq_len // self.period
        if num_p is None:
            num_p = self.num_p

        self.embedding = PatchEmbed(seq_len=seq_len, d_model=d_model, dropout=dropout, num_p=self.num_p)

        layers = self.layers_init(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            stable_len=stable_len,
            activation=activation,
            ia_layers=ia_layers,
            pd_layers=pd_layers,
            ca_layers=ca_layers,
            num_p=num_p,
        )
        self.encoder = TSEncoder(layers)

        out_p = self.num_p if pd_layers == 0 else num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * d_model, pred_len, bias=False)
        )

    def layers_init(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        attn_dropout,
        stable_len,
        activation,
        ia_layers,
        pd_layers,
        ca_layers,
        num_p,
    ):
        integrated_attention = [IntAttention(
            TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
            d_model, d_ff, dropout=dropout, stable_len=stable_len,
            activation=activation, stable=True, enc_in=self.c_in
        ) for _ in range(ia_layers)]

        patch_sampling = [PatchSampling(
            TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
            d_model, d_ff, stable=False, stable_len=stable_len,
            in_p=self.num_p if i == 0 else num_p, out_p=num_p,
            dropout=dropout, activation=activation
        ) for i in range(pd_layers)]

        cointegrated_attention = [CointAttention(
            TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
            d_model, d_ff, dropout=dropout,
            activation=activation, stable=False, enc_in=self.c_in, stable_len=stable_len,
        ) for _ in range(ca_layers)]

        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros((*x_enc.shape[:-1], 4), device=x_enc.device)

        mean, std = (x_enc.mean(1, keepdim=True).detach(),
                     x_enc.std(1, keepdim=True).detach())
        x_enc = (x_enc - mean) / (std + 1e-5)

        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, :self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)

        return dec_out * std + mean

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# Backward-compatibility wrapper.
class Model(TimeBridge):
    def __init__(self, configs):
        super().__init__(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            period=getattr(configs, "period", 24),
            num_p=getattr(configs, "num_p", None),
            d_model=getattr(configs, "d_model", 128),
            n_heads=getattr(configs, "n_heads", 8),
            d_ff=getattr(configs, "d_ff", 128),
            ia_layers=getattr(configs, "ia_layers", 3),
            pd_layers=getattr(configs, "pd_layers", 1),
            ca_layers=getattr(configs, "ca_layers", 0),
            stable_len=getattr(configs, "stable_len", 6),
            dropout=getattr(configs, "dropout", 0.0),
            attn_dropout=getattr(configs, "attn_dropout", 0.15),
            activation=getattr(configs, "activation", "gelu"),
            revin=getattr(configs, "revin", True),
        )





import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LocalMask():
    def __init__(self, B, L,S,device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask2 = ~torch.triu(torch.ones(mask_shape,dtype=torch.bool),diagonal=-self.len).to(device)
            self._mask = self._mask1+self._mask2
    @property
    def mask(self):
        return self._mask





import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from reformer_pytorch import LSHSelfAttention
from einops import rearrange


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, attn_map=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.attn_map = attn_map
        self.alpha = nn.Parameter(torch.rand(1))
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, long_term=True):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn_map = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(attn_map)
        if self.attn_map is True:
            heat_map = attn_map[:, ...].max(1)[0]
            heat_map = torch.clamp_max(heat_map, 0.15)
            # heat_map = torch.softmax(heat_map, -1)
            for b in range(heat_map.shape[0]):
                # for c in range(heat_map.shape[1]):
                h_map = heat_map[b, ...].detach().cpu().numpy()
                # plt.savefig(heat_map, f'{b} sample {c} channel')
                plt.figure(figsize=(10, 8), dpi=200)
                plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                plt.colorbar()

                # 设置X轴和Y轴的标签为黑体文字
                plt.rcParams['font.family'] = 'serif'
                plt.rcParams['font.serif'] = ['Times New Roman']
                plt.xlabel('Key Channel', fontsize=14)
                plt.ylabel('Query Channel', fontsize=14)

                # 设置标题
                # plt.title('Long-Term Correlations', fontdict={'weight': 'bold'}, fontsize=16, color='green')

                plt.tight_layout()
                plt.savefig(f'./stable map/{b}_sample.png')
                # plt.savefig(f'./non_stable map/{b}_sample.png')
                plt.close()
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.inner_attention is None:
            return self.out_projection(self.value_projection(values)), None
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TSMixer, self).__init__()

        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q_proj(q).reshape(B, L, H, -1)
        k = self.k_proj(k).reshape(B, S, H, -1)
        v = self.v_proj(v).reshape(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn


class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None, attn_map=False, nst=False):
        super(ResAttention, self).__init__()

        self.nst = nst
        self.scale = scale
        self.attn_map = attn_map
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        if self.attn_map is True:
            heat_map = attn_map.reshape(32, -1, H, L, S)
            for b in range(heat_map.shape[0]):
                for c in range(heat_map.shape[1]):
                    h_map = heat_map[b, c, 0, ...].detach().cpu().numpy()
                    # plt.savefig(heat_map, f'{b} sample {c} channel')

                    plt.figure(figsize=(10, 8), dpi=200)
                    plt.imshow(h_map, cmap='Reds', interpolation='nearest')
                    plt.colorbar()

                    # 设置X轴和Y轴的标签为黑体文字
                    plt.rcParams['font.family'] = 'serif'
                    plt.rcParams['font.serif'] = ['Times New Roman']
                    plt.xlabel('Key Time Patch', fontsize=14)
                    plt.ylabel('Query Time Patch', fontsize=14)
                    plt.tight_layout()
                    if self.nst is True:
                        plt.savefig(f'./time map/{b}_sample_{c}_channel.png')
                    else:
                        plt.savefig(f'./stable time map/{b}_sample_{c}_channel.png')
                    # 关闭当前图形窗口
                    plt.close()
        A = self.dropout(attn_map)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A











import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


def PeriodNorm(x, period_len=6):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i:-i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1),
                 mode='replicate', pad=(period_len - 1, 0)).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class IntAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, stable_len=8, attn_map=False,
                 dropout=0.1, activation="relu", stable=True, enc_in=None):
        super(IntAttention, self).__init__()
        self.stable = stable
        self.stable_len = stable_len
        self.attn_map = attn_map
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)

        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = PeriodNorm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        new_x = new_x.reshape(b, c, n, d)
        return new_x


class PatchSampling(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=False, stable_len=8):
        super(PatchSampling, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len

        self.attention = attention
        self.conv1 = nn.Conv1d(
            self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(
            self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat(
            [new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, axial=True, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None, ):
        super(CointAttention, self).__init__()

        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model

        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)

        self.num_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d(
            (0, self.num_rc ** 2 - (enc_in + 4)), 0)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.axial_func is True:
            new_x = self.axial_attn(x)
        else:
            new_x = self.full_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def axial_attn(self, x):
        b, c, n, d = x.shape

        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = (self.pad_ch(new_x.transpose(-1, -2))
                 .transpose(-1, -2).reshape(-1, self.num_rc, d))
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', r=self.num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x

        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b, n=n)
        return new_x[:, :c, :]