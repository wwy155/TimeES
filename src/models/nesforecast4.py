import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband, synthesize_per_timestep_from_half
import math


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
        t_emb=False,
        additive_scale=True,
        use_norm=False,
        pickout_zero_freq=False,
        fast_build=False,
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.M = M
        self.c_in = c_in
        # all_selected = torch.cat(selected_freqs).unique().sort().values.to(device)
        # print("all_selected:",  all_selected)
        self.selected_freqs = selected_freqs
        self.fast_build = fast_build

        
        self.use_norm = use_norm
        self.omegas =  omegas.to(device).float() #torch.fft.fftfreq(input_len).to(device)
        # self.freq_indices = all_selected
        # self.K = len(all_selected)
        
        # self.selected_omegas = self.omegas[self.freq_indices]
        self.t_emb = t_emb
        self.A_init = A_init # N, T, M
        if pickout_zero_freq:
            self.A_init[:, 0] = 0
        if t_emb:
            self.time_emb = TemporalEmbedding(512, 'timeF', 'h')
            # self.w_embd = nn.Parameter(torch.randn(self.K, 256))
            self.time_A_predictor = nn.ModuleList()
            for i in range(c_in):
                K_i = len(selected_freqs[i])  # number of selected freqs for channel i
                self.time_A_predictor.append(
                    nn.Sequential(
                        nn.Linear(512, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 2 * K_i)  # real + imag for K_i frequencies
                    )
                )



        self.device = device
        self.pickout_zero_freq = pickout_zero_freq
        self.fft_len = self.M // 2 + 1
        # self.A_scale_predictor = nn.Sequential(
        #     nn.Linear(input_len , hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 2 * (out_len) * self.K)
        # )


        self.input_enc = nn.Linear(input_len , hidden_dim)

        self.A_predictor_per_channel = nn.ModuleList()
        for i in range(c_in):
            self.A_predictor_per_channel.append(nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * (out_len) * len(self.selected_freqs[i]))
        ))

        self.additive_scale = additive_scale

        # create freq mask
        freq_mask = torch.zeros(self.c_in, self.fft_len, device=device)
        for i, freq_idx in enumerate(self.selected_freqs):
            if len(freq_idx) > 0:
                freq_mask[i, freq_idx] = 1

        self.freq_mask = freq_mask.unsqueeze(0).unsqueeze(-2)  # (c_in, fft_len)



    def forward(self, X, t_index_in, t_index_out, x_mark=None, y_mark=None):
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
        B = X.shape[0]
        device = X.device
        X = X.permute(0, 2, 1) # B N T
        
        t_index_stft = t_index_in[:, -1]
        A_slice = self.A_init[:, t_index_stft.to('cpu'), :].to(self.device) # N 1 M


        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(-1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev



        X_enc  = self.input_enc(X) # B N H
        t_last = t_index_in[:, -1]  # [B]

        # Prepare output containers
        out_real_list = []
        out_imag_list = []

        for i in range(self.c_in):
            freq_idx_i = self.selected_freqs[i].to(device)  # [K_i]
            K_i = len(freq_idx_i)

            A_init_i = self.A_init[i, t_last.cpu(), :]  # [B, M]
            A_init_i_selected = A_init_i[:, freq_idx_i.cpu()].to(device)  # [B, K_i]
            A_init_realimag = torch.view_as_real(A_init_i_selected.unsqueeze(1))  # [B, 1, K_i, 2]
            # Predict scale for this channel
            pred_i = self.A_predictor_per_channel[i](X_enc[:, i, :])  # [B, 2 * out_len * K_i]
            A_scale_i = pred_i.view(B, self.out_len, K_i, 2)  # [B, out_len, K_i, 2]

            # Combine with initial (additive)
            if self.additive_scale:
                A_half_realimag = A_scale_i + A_init_realimag  # broadcast over out_len
            else:
                A_half_realimag = A_scale_i

            if self.t_emb and y_mark is not None:
                # y_mark: [B, out_len, TE]
                time_e = self.time_emb(y_mark)  # [B, out_len, 512]
                # Predict time modulation for channel i
                time_mod_i = self.time_A_predictor[i](time_e)  # [B, out_len, 2 * K_i]
                time_mod_i = time_mod_i.view(B, self.out_len, K_i, 2)
                A_half_realimag = A_half_realimag + time_mod_i

            # Convert to complex
            A_half_complex = torch.view_as_complex(A_half_realimag)  # [B, out_len, K_i]


            # Scatter into full spectrum (size: fft_len = M//2 + 1)
            out_real = torch.zeros(B, self.out_len, self.fft_len, device=device)
            out_imag = torch.zeros(B, self.out_len, self.fft_len, device=device)

            # Expand freq_idx_i for scatter
            idx_exp = freq_idx_i.unsqueeze(0).unsqueeze(0).expand(B, self.out_len, -1)  # [B, out_len, K_i]

            out_real.scatter_(2, idx_exp, A_half_complex.real)
            out_imag.scatter_(2, idx_exp, A_half_complex.imag)

            out_real_list.append(out_real)
            out_imag_list.append(out_imag)

        # Stack across channels
        out_real_full = torch.stack(out_real_list, dim=1)  # [B, N, out_len, fft_len]
        out_imag_full = torch.stack(out_imag_list, dim=1)  # [B, N, out_len, fft_len]
        A_half_all = torch.complex(out_real_full, out_imag_full)  # [B, N, out_len, fft_len]

        # # Optional: time embedding modulation (if enabled)
        # if self.t_emb and y_mark is not None:
        #     time_e = self.time_emb(y_mark)  # [B, out_len, 512]
        #     A_time = self.time_A_predictor(time_e)  # [B, out_len, c_in * K_total * 2] ❌ — but K varies per channel!



        if self.fast_build:
            # NOTE: sythesize faster and less memoery but less performance
            all_rec = synthesize_per_timestep_from_half(A_half_all, self.M, self.device)
            all_rec = all_rec.reshape(B, self.c_in, -1)
            A_half_all = A_half_all.reshape(B, self.c_in, A_half_all.shape[-2], A_half_all.shape[-1] )
        else:
            # self build
            # Build full Hermitian spectrum for all channels & time steps
            A_all = construct_hermitian_spectrum(A_half_all, self.M)  # [B, N, out_len, M]
            # Synthesize signal in one batched call
            t_out = torch.arange(self.out_len, device=device).float()  # [out_len]
            all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, t_out)  # [B, N, out_len]
            all_rec = all_rec.reshape(B, self.c_in, -1)
            A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        
        # De-normalize
        if self.use_norm:
            all_rec = all_rec * stdev + means

        return all_rec, A_half_all 
