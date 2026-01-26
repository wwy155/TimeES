import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband
from torch_timeseries.nn.Transformer_EncDec import Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import FullAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding_inverted
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer
from torch_timeseries.nn.embedding import PatchEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



class PatchTSTEnc(nn.Module):
    def __init__(self, seq_len, pred_len,  factor=1, enc_in=7, patch_len=16, n_heads=8, stride=8, d_ff=2048, activation='gelu', e_layers=2, d_model=512, dropout=0.0):
        super(PatchTSTEnc, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride


        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.head_nf = d_model * \
                       int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(enc_in, self.head_nf, pred_len,
                                    head_dropout=dropout)

    def forward(self, x_enc):
        B, N, L = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        

        # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1) # (B N T)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        return dec_out.reshape(B*N, -1)



