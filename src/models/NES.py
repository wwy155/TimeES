import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband



class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class NeuralEvolutionarySpectra(nn.Module):
    def __init__(
        self,
        input_len,
        c_in,
        out_len,
        device,
        omegas,
        M,
        A_init,
        selected_freqs,
        hidden_dim=512,
        task='forecast',
        t_emb=None,
        additive_scale=True,
        use_norm=True,
        pickout_zero_freq=True,
        out_prob=0,
        detrend=True
    ):
    
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.M = M
        self.c_in = c_in
        self.de_trend = detrend

        self.task = task
        all_selected = torch.cat(selected_freqs).unique().sort().values.to(device)
        print("all_selected:",  all_selected)
        self.selected_freqs = selected_freqs

        
        self.use_norm = use_norm
        self.omegas =  omegas.to(device).float() #torch.fft.fftfreq(input_len).to(device)
        self.freq_indices = all_selected
        self.K = len(all_selected)
        
        self.selected_omegas = self.omegas[self.freq_indices]
        self.t_emb = t_emb
        self.A_init = A_init # T, N, M
        if pickout_zero_freq:
            self.A_init[:, 0] = 0
        if t_emb:
            self.time_emb = TemporalEmbedding(512, 'timeF', 'h')
            # self.w_embd = nn.Parameter(torch.randn(self.K, 256))
            self.time_A_predictor = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.K*2)
            )


        self.device = device
        self.pickout_zero_freq = pickout_zero_freq
        self.fft_len = self.M // 2 + 1
        self.A_scale_predictor = nn.Sequential(
            # nn.Linear(input_len , hidden_dim),
            # nn.Linear(2 * input_len * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * input_len * topk + topk*input_len, hidden_dim),
            nn.Linear(input_len , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),

            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (out_len) * self.K)
        )


        self.additive_scale = additive_scale

        if out_prob != 0:
            self.class_projection = nn.Linear(input_len*c_in*self.K*2,  out_prob)

        # # zeor frequencies modeling
        # self.zero_frequencies = nn.Sequential(
        #     nn.Linear(input_len, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, input_len + out_len)
        # )


        # create freq mask
        freq_mask = torch.zeros(self.c_in, self.fft_len, device=device)
        for i, freq_idx in enumerate(self.selected_freqs):
            if len(freq_idx) > 0:
                freq_mask[i, freq_idx] = 1

        self.freq_mask = freq_mask.unsqueeze(0).unsqueeze(-2)  # (c_in, fft_len)
    

    def forecast(self, X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        pass

    def classification(self, X, x_mark=None, y_mark=None, return_A=False, rec=False):
        B = X.shape[0]
        device = X.device
        X = X.permute(0, 2, 1)
        

        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(-1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        X = X.reshape(-1, self.input_len) # Channel independence

        STFT_init_complex = torch.fft.rfft(X, norm="ortho").cfloat() # B*c, M
        STFT_init_complex = STFT_init_complex[:, self.freq_indices] # B*c, K
        STFT_init = torch.view_as_real(STFT_init_complex) # B*c, K, 2




        A_scale = self.A_scale_predictor(torch.concat([X], dim=-1)).view(-1, self.input_len, self.K, 2)
        if self.additive_scale:
            A_half = A_scale + STFT_init.unsqueeze(1) # 
        else:
            A_half = A_scale 

        if self.t_emb:
            # time_e = self.time_emb(torch.concat([x_mark, y_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            time_e = self.time_emb(torch.concat([x_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            # time_w = self.w_embd.unsqueeze(0).unsqueeze(0).expand(B, self.out_len, -1) # B, T, WE
            A_time = self.time_A_predictor(torch.concat([time_e], dim=-1)).view(-1, self.input_len, self.K, 2) # B, T, K
            A_time = A_time.repeat(self.c_in, 1, 1, 1)
            A_half = A_half + A_time
        # A_half used for classification A_half: [B*N, self.K, 2 ]
        output = self.class_projection(A_half.reshape(B, -1))

        if rec:
            # reconstruct
            A_half = torch.view_as_complex(A_half)
            out_real = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
            out_imag = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
            out_real.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.input_len, -1), src=A_half.real)
            out_imag.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.input_len, -1), src=A_half.imag)
            A_half_all = torch.complex(out_real, out_imag)
            A_all = construct_hermitian_spectrum(A_half_all, self.M)
            all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(0, self.input_len).to(device))  # [B, input_len]
            all_rec = all_rec.reshape(B, self.c_in, -1)
            A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
            if self.use_norm:
                # De-Normalization from Non-stationary Transformer
                all_rec = all_rec * stdev
                all_rec = all_rec + means

            if  return_A:
                return output, all_rec.permute(0, 2, 1), A_all
            else:
                return output, all_rec


        return output


    def forward(self, X, t_index_in=None, t_index_out=None, x_mark=None, y_mark=None, rec=False, return_A=False):
        """
        Forward pass.

        Args:
            X: [B, input_len, N] — dummy input (for interface consistency; not used in computation)
            t_index_in: [input_len] or [B, input_len], long
            t_index_out: [out_len] or [B, out_len], long
            x_mark: [B, T, TE], long

        Returns:
            out: [B, input_len + out_len], reconstructed + predicted signal
            A_all_out: [B, input_len + out_len, M], full-spectrum amplitudes (complex)
        """

        if self.task == 'classification':
            return self.classification(X, rec=rec, return_A=return_A)


        B = X.shape[0]
        device = X.device
        X = X.permute(0, 2, 1)
        

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(-1, keepdim=True).detach()
            X = X - means
            # stdev = torch.sqrt(torch.var(X, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            # X /= stdev

        X = X.reshape(-1, self.input_len) # Channel independence


        t_index_stft = t_index_in[:, -1]
        STFT_init_complex = self.A_init[:, t_index_stft.to('cpu'), :][:, :,  self.freq_indices.to('cpu')].permute(1, 0, 2).reshape(B*self.c_in, -1).to(self.device) # B, N, M//2 + 1 clofat
        STFT_init = torch.view_as_real(STFT_init_complex) # B, N, M//2 + 1, 2
        A_scale = self.A_scale_predictor(torch.concat([X], dim=-1)).view(-1, self.out_len, self.K, 2)
        
        if self.additive_scale:
            A_half = A_scale + STFT_init.unsqueeze(1)
        else:
            A_half = A_scale

        if self.t_emb:
            # time_e = self.time_emb(torch.concat([x_mark, y_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            time_e = self.time_emb(torch.concat([y_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            # time_w = self.w_embd.unsqueeze(0).unsqueeze(0).expand(B, self.out_len, -1) # B, T, WE
            A_time = self.time_A_predictor(torch.concat([time_e], dim=-1)).view(-1, self.out_len, self.K, 2) # B, T, K
            A_time = A_time.repeat(self.c_in, 1, 1, 1)
            A_half = A_half + A_time

        A_half = torch.view_as_complex(A_half)

        out_real = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
        out_imag = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
        out_real.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.real)
        out_imag.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.imag)
        A_half_all = torch.complex(out_real, out_imag)

        # # mask out unused freqs
        # A_half_all = A_half_all.reshape(B, self.c_in, A_half_all.shape[-2], A_half_all.shape[-1]) * self.freq_mask 
        # A_half_all = A_half_all.reshape(B*self.c_in, A_half_all.shape[-2], A_half_all.shape[-1])


        A_all = construct_hermitian_spectrum(A_half_all, self.M)
        # all_rec = synthesize_signal_on_subband(A_all, self.selected_omegas, self.M, torch.arange(0, self.out_len).to(device))  # [B, input_len]
        all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(0, self.out_len).to(device))  # [B, input_len]
        
        
        all_rec = all_rec.reshape(B, self.c_in, -1)
        A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            all_rec = all_rec * stdev
            all_rec = all_rec + means

        return all_rec, A_all 
