import torch
from torch import nn
from torch_timeseries.nn.encoder  import Encoder, EncoderLayer
from torch_timeseries.nn.decoder import Decoder, DecoderLayer
from torch_timeseries.nn.attention import FullAttention, AttentionLayer
from torch_timeseries.nn.embedding import PatchEmbedding, PositionalEmbedding


class WaveEmbedding(nn.Module):
    def __init__(self, d_model, wave_len, padding, dropout):
        super(WaveEmbedding, self).__init__()
        # Patching
        self.patch_len = wave_len
        # self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(wave_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # x = self.padding_patch_layer(x)
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)




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


class WaveFormer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, seq_len,pred_len,enc_in,n_heads=8, t_feat_size=8, dropout=0.0,e_layers=2, d_layers=1,d_model=512,d_ff=2048, wave_len=16, stride=8, num_class=0):
        """
        wave_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.wave_len = wave_len
        self.t_feat_size = t_feat_size
        padding = stride
        
        input_token_num  =  (seq_len + wave_len - 1) // wave_len
        self.output_token_num = (pred_len + wave_len - 1) // wave_len

        self.time_prior_embed = nn.Sequential(
            nn.Linear(t_feat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, enc_in)
        )
        
        self.proj = nn.Linear(d_model, wave_len)
        
        # patching and embedding
        self.wave_embedding = WaveEmbedding(
            d_model, wave_len, padding, dropout)

        self.mean_predictor = nn.Sequential(
            nn.Linear(input_token_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_token_num)
        )
        self.std_predictor = nn.Sequential(
            nn.Linear(input_token_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_token_num),
            nn.Softplus()
        )
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

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation='gelu',
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # x_enc: B, T, N
        # x_mark_dec: B, O, F
        B, T, N = x_enc.shape
        pad_len = (self.wave_len - T % self.wave_len) % self.wave_len  # 
        if pad_len > 0:
            pad_tensor = torch.zeros(B, pad_len, N, device=x_enc.device, dtype=x_enc.dtype)
            x_padded = torch.cat([x_enc, pad_tensor], dim=1)  # (B, T + pad_len, N)
        else:
            x_padded = x_enc
        T_pad = x_padded.shape[1]
        K = T_pad // self.wave_len  # 波形 token 个数


        x_token = x_padded.view(B, K, self.wave_len, N)

        means = x_token.mean(dim=2) # B, K, N
        stds = x_token.std(dim=2, unbiased=False) # B, K, N
        waves = (x_token - means.unsqueeze(2)) / (stds.unsqueeze(2) + 1e-8) # B, K, wave_len, N
        
        pred_means = self.mean_predictor(means.transpose(1, 2)).transpose(1, 2) # B, oK, N
        pred_stds = self.std_predictor(stds.transpose(1, 2)).transpose(1, 2)   # B, oK, N
        
        
        

        wave_embed = self.wave_embedding(waves.permute(0, 3, 1, 2))  # B*N, iK, D iK:input token num
        
        enc_out, attns = self.encoder(wave_embed)
        
        time_prior = self.time_prior_embed(x_mark_dec) # B, O, N
        tp_token = time_prior.view(B, self.output_token_num, self.wave_len, N) # B, Ko, O, N
        t_waves = (tp_token - pred_means.unsqueeze(2)) / (pred_stds.unsqueeze(2)+ 1e-8) # B, K, wave_len, N
        dec_out = self.wave_embedding(t_waves.permute(0, 3, 1, 2)) # B*N, oK, D
        
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None) # B*N, oK, D
        
        dec_out = self.proj(dec_out).reshape(B, N, self.output_token_num, -1).permute(0, 2, 3, 1) # B, oK, wave_len, N
        dec_out = dec_out*pred_stds.unsqueeze(2) + pred_means.unsqueeze(2)
        dec_out = dec_out.reshape(B, self.output_token_num*self.wave_len, N) # B, O, N
        
        
        
        # dec_out = self.projection(dec_out)
        
        

        # # means = x_enc
        
                
        # # means = x_enc.mean(1, keepdim=True).detach()
        # # x_enc = x_enc - means
        # # stdev = torch.sqrt(
        # #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # # x_enc /= stdev

        # # do patching and embedding
        # x_enc = x_enc.permute(0, 2, 1) # (B N T)
        # # u: [bs * nvars x patch_num x d_model]
        # enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        # z: [bs x nvars x patch_num x d_model]
        # enc_out = torch.reshape(
        #     enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)

        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization 
        
        
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
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
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
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
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
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
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
