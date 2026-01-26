import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband, precompute_idft_synthesis_matrix
from src.nn.PatchTSTBackbone import PatchTSTEnc
from src.nn.iTransformerBackbone import iTransformerEnc


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


class TimeES(nn.Module):
    def __init__(
        self,
        input_len,
        c_in,
        out_len,
        device,
        omegas,
        M,
        selected_freqs,
        hidden_dim=512,
        t_emb=None,
        tc_emb=False,
        use_norm=True,
        fast_build=True,
        backbone='linear',
        task='forecast',
        out_prob=0,
        num_samples=100,
        mini_sample_num=1,
        # layer_nums=2,
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.M = M
        self.c_in = c_in
        all_selected = torch.cat(selected_freqs).unique().sort().values.to(device)
        print("all_selected:",  all_selected)
        self.selected_freqs = selected_freqs
        self.fast_build = fast_build

        self.task = task

        
        self.use_norm = use_norm
        self.omegas =  omegas.to(device).float() #torch.fft.fftfreq(input_len).to(device)
        self.freq_indices = all_selected
        self.K = len(all_selected)
        self.F = precompute_idft_synthesis_matrix(self.freq_indices, self.M, device)
        
        self.selected_omegas = self.omegas[self.freq_indices]
        self.t_emb = t_emb
        if t_emb:
            self.time_emb = TemporalEmbedding(512, 'timeF', 'h')
            # self.w_embd = nn.Parameter(torch.randn(self.K, 256))
            self.time_A_predictor = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.K*2*self.c_in)
            )
        self.tc_emb = tc_emb
        if tc_emb:
            self.time_emb = TemporalEmbedding(512, 'timeF', 'h')
            # self.c_emb = nn.Embedding(c_in, 512)
            # self.tc_A = nn.Sequential(
            #     nn.Linear(1024, hidden_dim),
            #     nn.ReLU(),
            #     nn.Linear(hidden_dim, self.K*2)
            # )
            self.c_emb = nn.Parameter(torch.randn(c_in, 512))  

            self.time_A_predictor = nn.Sequential(
                nn.Linear(512 + 512, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.K*2)
            )


        self.device = device
        self.fft_len = self.M // 2 + 1

        if backbone =='PatchTST':
            self.A_scale_predictor = PatchTSTEnc(input_len, 2 * (out_len) * self.K, enc_in=c_in)
        elif backbone=='iTransformer':
            self.A_scale_predictor = iTransformerEnc(input_len, 2 * (out_len) * self.K, enc_in=c_in)
        else:
            self.A_scale_predictor = nn.Sequential(
                nn.Linear(input_len, hidden_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(2 - 1)],
                nn.Linear(hidden_dim, 2 * out_len * self.K)
            )


        if self.task == 'prob_forecast' or  self.task == 'prob_rec':
            self.num_samples = num_samples
            self.mini_sample_num = mini_sample_num
            self.W_mu_real = nn.Parameter(torch.zeros(self.M*self.c_in), requires_grad=True)      # [M]
            self.W_mu_imag = nn.Parameter(torch.zeros(self.M*self.c_in), requires_grad=True)      # [M]
            self.W_logvar_real = nn.Parameter(torch.zeros(self.M*self.c_in), requires_grad=True)  # log(σ²)
            self.W_logvar_imag = nn.Parameter(torch.zeros(self.M*self.c_in), requires_grad=True)

            self.t_all = torch.arange(self.out_len, dtype=torch.float32, device=self.device)  # [T]
            self.phase = torch.exp(1j * torch.outer(self.t_all, self.omegas))  # [T, M]

        elif self.task =='classification':
            self.out_prob = out_prob
            self.out_class_proj = nn.Linear(2 * out_len * self.K*self.c_in, out_prob)

    def sample_W(self, B, device, num_samples):
        """
        Sample W: [num_samples, B, M] (complex)
        Using reparameterization: W = μ + ε * σ, ε ~ N(0,1)
        """
        # Expand to [num_samples, B, M]
        mu_r = self.W_mu_real.unsqueeze(0).unsqueeze(0).expand(num_samples, B, -1)
        mu_i = self.W_mu_imag.unsqueeze(0).unsqueeze(0).expand(num_samples, B, -1)
        logvar_r = self.W_logvar_real.unsqueeze(0).unsqueeze(0).expand(num_samples, B, -1)
        logvar_i = self.W_logvar_imag.unsqueeze(0).unsqueeze(0).expand(num_samples, B, -1)

        std_r = torch.exp(0.5 * logvar_r)
        std_i = torch.exp(0.5 * logvar_i)

        eps_r = torch.randn_like(std_r, device=device)
        eps_i = torch.randn_like(std_i, device=device)

        W_real = mu_r + eps_r * std_r  # [num_samples, B, M*c_in]
        W_imag = mu_i + eps_i * std_i  # [num_samples, B, M*c_in]

        return torch.complex(W_real, W_imag)  # [num_samples, B, M*c_in]
    def build_A_half(self, X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        B = X.shape[0]
        device = X.device
        raw_X = X
        X = X.permute(0, 2, 1)
        
        X = X.reshape(-1, self.input_len) # Channel independence


        return self.A_scale_predictor(torch.concat([X], dim=-1)).view(-1, self.out_len, self.K, 2)

    def fast_build_X(self, X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        B = X.shape[0]
        device = X.device
        X = X.permute(0, 2, 1)
        # X = X.reshape(-1, self.input_len) # Channel independence

        A_half = self.A_scale_predictor(torch.concat([X], dim=-1)).view(-1, self.out_len, self.K, 2)
        if self.t_emb:
            # time_e = self.time_emb(torch.concat([x_mark, y_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            time_e = self.time_emb(torch.concat([y_mark], dim=1)) # B, inp_len+out_len 128 B T TE*c_in
            tB, _, _ = time_e.shape  # T = inp_len + out_len

            # time_w = self.w_embd.unsqueeze(0).unsqueeze(0).expand(B, self.out_len, -1) # B, T, WE
            A_time = self.time_A_predictor(torch.concat([time_e], dim=-1)).view(tB, self.out_len,  self.c_in, self.K, 2) # B, O, C, K, 2
            # A_time = A_time.repeat(self.c_in, 1, 1, 1)
            A_time = A_time.permute(0, 2, 1, 3,4).reshape(-1, self.out_len, self.K, 2)
            A_half = A_half + A_time
        if self.tc_emb:
            time_e = self.time_emb(torch.concat([y_mark], dim=1)) # B, inp_len+out_len 128 B O TE
            tB, tT, _ = time_e.shape  # T = inp_len + out_len
            c_emb_expanded = self.c_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, c, hidden_dim)
            c_emb_expanded = c_emb_expanded.expand(tB, tT, -1, -1)   # (B, T, c, hidden_dim)

            time_e = self.time_emb(y_mark)  # B, T, hidden_dim)
            time_e = time_e.unsqueeze(2).expand(-1, -1, self.c_in, -1)  # (B, T, c, hidden_dim)

            combined = torch.cat([time_e, c_emb_expanded], dim=-1)  # (B, T, c, 2*hidden_dim)

            A_time = self.time_A_predictor(combined)  #  (B, T, c, 2*K)

            A_time = A_time.view(tB, tT, self.c_in, self.K, 2).permute(0, 2, 1, 3, 4)
            A_time = A_time.reshape(-1,tT, self.K, 2)
            A_half = A_half + A_time
        X = torch.einsum('bok,km->bom', torch.view_as_complex(A_half), self.F).real  # [B*c_in, out_len, M]
        return X, A_half




    def build_A(self, X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        B = X.shape[0]
        device = X.device
        raw_X = X
        X = X.permute(0, 2, 1)
        X = X.reshape(-1, self.input_len) # Channel independence


        A_half = self.A_scale_predictor(torch.concat([X], dim=-1)).view(-1, self.out_len, self.K, 2)

        if self.t_emb:
            # time_e = self.time_emb(torch.concat([x_mark, y_mark], dim=1)) # B, inp_len+out_len 128 B T TE
            time_e = self.time_emb(torch.concat([y_mark], dim=1)) # B, inp_len+out_len 128 B T TE*c_in
            tB, _, _ = time_e.shape  # T = inp_len + out_len

            # time_w = self.w_embd.unsqueeze(0).unsqueeze(0).expand(B, self.out_len, -1) # B, T, WE
            A_time = self.time_A_predictor(torch.concat([time_e], dim=-1)).view(tB, self.out_len,  self.c_in, self.K, 2) # B, O, C, K, 2
            # A_time = A_time.repeat(self.c_in, 1, 1, 1)
            A_time = A_time.permute(0, 2, 1, 3,4).reshape(-1, self.out_len, self.K, 2)
            A_half = A_half + A_time
        if self.tc_emb:
            time_e = self.time_emb(torch.concat([y_mark], dim=1)) # B, inp_len+out_len 128 B O TE
            tB, tT, _ = time_e.shape  # T = inp_len + out_len
            c_emb_expanded = self.c_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, c, hidden_dim)
            c_emb_expanded = c_emb_expanded.expand(tB, tT, -1, -1)   # (B, T, c, hidden_dim)

            time_e = self.time_emb(y_mark)  # B, T, hidden_dim)
            time_e = time_e.unsqueeze(2).expand(-1, -1, self.c_in, -1)  # (B, T, c, hidden_dim)

            combined = torch.cat([time_e, c_emb_expanded], dim=-1)  # (B, T, c, 2*hidden_dim)

            A_time = self.time_A_predictor(combined)  #  (B, T, c, 2*K)

            A_time = A_time.view(tB, tT, self.c_in, self.K, 2).permute(0, 2, 1, 3, 4)
            A_time = A_time.reshape(-1,tT, self.K, 2)
            A_half = A_half + A_time


        A_half = torch.view_as_complex(A_half)

        out_real = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
        out_imag = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
        out_real.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.real)
        out_imag.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.imag)
        A_half_all = torch.complex(out_real, out_imag)
        A_all = construct_hermitian_spectrum(A_half_all, self.M)

        return A_all
    
    def forecast(self,X, t_index_in, t_index_out, x_mark=None, y_mark=None):
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

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        B = X.shape[0]
        if self.fast_build:
            all_rec, A_half = self.fast_build_X(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
            A_all = A_half
        else:
            A_all = self.build_A(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
            all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(0, self.out_len).to(X.device))  # [B, input_len]
            A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        
        all_rec = all_rec.reshape(B, self.c_in, -1)
        
        if self.use_norm:
            all_rec = all_rec * stdev.permute(0, 2, 1)
            all_rec = all_rec + means.permute(0, 2, 1)

        return all_rec, A_all 





    def sample_from_A(self, A_all, N, B):
        # Precompute phase: exp(i ω_k t_n) → [T, M]

        W_K = self.sample_W(B, self.device, N) # [num_samples, B, M*c_in]
        W_K = W_K.reshape(N, B*self.c_in, self.M)
        A_expanded = A_all.unsqueeze(0) # [1, B*c_in, T, M]
        phase_expanded = self.phase.unsqueeze(0).unsqueeze(0)  # [1, B*c_in, T, M]
        W_expanded = W_K.unsqueeze(2)       # [num_samples, B*c_in, 1, M]
        integrand = A_expanded * W_expanded  # [num_samples, B*c_in, T, M]
        integrand = integrand  * phase_expanded  # [num_samples, B*c_in, T, M] 
        X_complex = (1.0 / torch.sqrt(torch.tensor(self.M, dtype=torch.float32, device=self.device))) * \
                    torch.sum(integrand, dim=-1)  # [num_samples, B*c_in, T]
        X_real = X_complex.real  # [num_samples, B*c_in, T]
        return X_real        


    def sample_from_A_t(self, A_all, N, B, phase):
        # Precompute phase: exp(i ω_k t_n) → [T, M]

        W_K = self.sample_W(B, self.device, N) # [num_samples, B, M*c_in]
        W_K = W_K.reshape(N, B*self.c_in, self.M)
        A_expanded = A_all.unsqueeze(0) # [1, B*c_in, T, M]
        phase_expanded = phase.unsqueeze(0).repeat(1, self.c_in, 1, 1)  # [1, B*c_in, T, M]
        W_expanded = W_K.unsqueeze(2)       # [num_samples, B*c_in, 1, M]
        integrand = A_expanded * W_expanded  # [num_samples, B*c_in, T, M]
        integrand = integrand  * phase  # [num_samples, B*c_in, T, M] 
        X_complex = (1.0 / torch.sqrt(torch.tensor(self.M, dtype=torch.float32, device=self.device))) * \
                    torch.sum(integrand, dim=-1)  # [num_samples, B*c_in, T]
        X_real = X_complex.real  # [num_samples, B*c_in, T]
        return X_real        



    def prob_forecast(self,X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        # output: 

        # output: 
        # all_rec: B, O, N, S
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        # Full time grid
        t_all = torch.arange(self.out_len, dtype=torch.float32, device=self.device)  # [T]

        B =  X.shape[0]
        # A_all = self.build_A1(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        A_all = self.build_A(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)

        mini_iter = self.num_samples // self.mini_sample_num
        mini_sample_list = []
        for i in range(mini_iter):
            mini_samples = self.sample_from_A(A_all, self.mini_sample_num, B)
            mini_sample_list.append(mini_samples)
        
        samples = torch.concat(mini_sample_list, dim=0)
        all_rec = samples
        A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        all_rec = all_rec.reshape(self.num_samples, B, self.c_in, -1)
        all_rec = all_rec.permute(1, 3,2, 0) # B T N S
        # # Compute mean and variance over samples
        # mean_pred = X_pred_samples.mean(dim=0)      # [B, H]
        # var_pred = X_pred_samples.var(dim=0, unbiased=False)  # [B, H]
        
        if self.use_norm:
            all_rec = all_rec * stdev.unsqueeze(-1)
            all_rec = all_rec + means.unsqueeze(-1)
        return all_rec, A_all



    def prob_rec(self,X, t_index_in, t_index_out, x_mark=None, y_mark=None):
        # output: 

        # output: 
        # all_rec: B, O, N, S
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        # Full time grid
        t_all = torch.arange(self.out_len, dtype=torch.float32, device=self.device)  # [T]

        B =  X.shape[0]
        # A_all = self.build_A1(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        A_all = self.build_A(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        
        omega_expanded = self.omegas.unsqueeze(0).unsqueeze(0)   # [1, 1, 96]
        t_expanded = t_index_in.unsqueeze(-1)//self.M                    # [32, 384, 1]
        phase = omega_expanded * t_expanded   # or torch.mul(omega_expanded, t_expanded)

        mini_iter = self.num_samples // self.mini_sample_num
        mini_sample_list = []
        for i in range(mini_iter):
            mini_samples = self.sample_from_A_t(A_all, self.mini_sample_num, B, phase)
            mini_sample_list.append(mini_samples)
        
        samples = torch.concat(mini_sample_list, dim=0)
        all_rec = samples
        A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        all_rec = all_rec.reshape(self.num_samples, B, self.c_in, -1)
        all_rec = all_rec.permute(1, 3,2, 0) # B T N S
        # # Compute mean and variance over samples
        # mean_pred = X_pred_samples.mean(dim=0)      # [B, H]
        # var_pred = X_pred_samples.var(dim=0, unbiased=False)  # [B, H]
        
        if self.use_norm:
            all_rec = all_rec * stdev.unsqueeze(-1)
            all_rec = all_rec + means.unsqueeze(-1)

        return all_rec, A_all


    def classification(self,X, t_index_in=None, t_index_out=None, x_mark=None, y_mark=None):
        # output: 

        # output: 
        # all_rec: B, O, N, S
        B = X.shape[0]
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev

        # A_all = self.build_A1(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        A_half = self.build_A_half(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        
        A_half_in = A_half.reshape(B, -1)
        ouputs = self.out_class_proj(A_half_in)
        
        A_half = torch.view_as_complex(A_half)
        out_real = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
        out_imag = torch.zeros(B*self.c_in, self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
        out_real.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.real)
        out_imag.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B*self.c_in, self.out_len, -1), src=A_half.imag)
        A_half_all = torch.complex(out_real, out_imag)
        A_all = construct_hermitian_spectrum(A_half_all, self.M)

        A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )

        all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(0, self.out_len).to(X.device))  # [B, input_len]
        all_rec = all_rec.reshape(B, self.c_in, -1)
        A_all = A_all.reshape(B, self.c_in, A_all.shape[-2], A_all.shape[-1] )
        if self.use_norm:
            all_rec = all_rec * stdev.permute(0, 2, 1)
            all_rec = all_rec + means.permute(0, 2, 1)

        return ouputs, all_rec.permute(0, 2, 1), A_all



    def forward(self, X, t_index_in=None, t_index_out=None, x_mark=None, y_mark=None):
        if t_index_out is not None and t_index_in is not None:
            t_index_in = t_index_in.squeeze(-1)
            t_index_out = t_index_out.squeeze(-1)

        if self.task == 'forecast':
            return self.forecast(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        elif self.task == 'prob_forecast':
            return self.prob_forecast(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        elif self.task == 'prob_rec':
            return self.prob_forecast(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
        elif self.task == 'classification':
            return self.classification(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)
   # return self.prob_rec(X, t_index_in, t_index_out, x_mark=x_mark, y_mark=y_mark)

