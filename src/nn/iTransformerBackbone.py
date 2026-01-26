import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband
from torch_timeseries.nn.Transformer_EncDec import Encoder, EncoderLayer
from torch_timeseries.nn.SelfAttention_Family import FullAttention, AttentionLayer
from torch_timeseries.nn.embedding import DataEmbedding_inverted
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer



class iTransformerEnc(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=7, factor=1,n_heads=8, d_ff=2048, activation='gelu', e_layers=2, d_model=512, dropout=0.0):
        super(iTransformerEnc, self).__init__()

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, None, None,
                                                    dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)


    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        B, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out)[:, :N, :] # filter the covariates


        return dec_out.reshape(B*N, -1)
