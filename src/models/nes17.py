import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband

class NeuralEvolutionarySpectra(nn.Module):
    def __init__(
        self,
        input_len,
        out_len,
        device,
        topk=20,
        hidden_dim=512,
        additive_scale=True,
        use_norm=True,
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.topk = topk
        self.M = input_len
        self.use_norm = use_norm
        self.omegas =  torch.fft.fftfreq(input_len).to(device)
        # Register buffers
        # Predictor: history subband → future subband

        self.fft_len = self.input_len // 2 + 1
        self.A_scale_predictor = nn.Sequential(
            # nn.Linear(input_len , hidden_dim),
            # nn.Linear(2 * input_len * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * input_len * topk + topk*input_len, hidden_dim),
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (input_len + out_len) * self.topk)
        )
        self.additive_scale = additive_scale
        # # zeor frequencies modeling
        # self.zero_frequencies = nn.Sequential(
        #     nn.Linear(input_len, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, input_len + out_len)
        # )


        def syn_real_A(self):
            pass



    def forward(self, X, t_index_in, t_index_out):
        """
        Forward pass.

        Args:
            X: [B, input_len] — dummy input (for interface consistency; not used in computation)
            t_index_in: [input_len] or [B, input_len], long
            t_index_out: [out_len] or [B, out_len], long

        Returns:
            out: [B, input_len + out_len], reconstructed + predicted signal
            A_all_out: [B, input_len + out_len, M], full-spectrum amplitudes (complex)
        """
        B = X.shape[0]
        device = X.device

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(-1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev


        STFT_init_complex = torch.fft.rfft(X, norm="ortho").cfloat() # B, M clofat
        STFT_init = torch.view_as_real(STFT_init_complex) # B, M, 2
        # self.omegas = torch.fft.fftfreq(self.input_len).to(device)
        # we don't use zero frequencies
        # X_fft = torch.fft.fft(X - X.mean(dim=1, keepdim=True), self.input_len, dim=-1, norm="ortho") #  B, input_len
        # X_fft[0] = 0 # we don't use zero frequencies
        # _, topk_indices = torch.topk(torch.abs(X_fft), k=self.topk, dim=1, largest=True)  # [B, K]
        # selected_omegas = self.omegas[topk_indices] # [B, K]
        # main_fft_spectra = torch.gather(X_fft, 1, topk_indices)
        # predict_inp = torch.cat([X,  torch.view_as_real(main_fft_spectra).view(B,  -1), selected_omegas], dim=1) # [B, input_len + 2 * input_len * topk + topk]

        # selected_omegas = self.omegas_predictor(selected_omegas).reshape(B, -1, self.topk) # B, O+T, K


        # main_fft_spectra = X_fft[topk_indices]  # [B, K]
        # --- 2. Dynamic top-K frequency selection per batch ---
        # energy_per_frame = torch.abs(A_X_full)  # [B, N, M]
        _, topk_indices = torch.topk( torch.abs(STFT_init_complex) , k=self.topk, dim=1, largest=True)  # [B, K]
        selected_omegas = self.omegas[topk_indices] 
        
        # STFT_topk_unfilled = torch.zeros_like(STFT_init).scatter_(dim=1, index=topk_indices, src=torch.gather(STFT_init, 1, topk_indices))
        STFT_topk = torch.gather(STFT_init_complex, 1, topk_indices)  # [B, input_len, K_eff]
        # _, topk_indices = torch.topk(energy, k=self.topk, dim=1, largest=True)  # [B, K_eff]
        # topk_indices, _ = torch.sort(topk_indices, dim=2)  
        A_scale = self.A_scale_predictor(X).view(B, self.input_len + self.out_len, self.topk, 2)
        if self.additive_scale:
            A_half = torch.view_as_complex(A_scale + torch.view_as_real(STFT_topk.unsqueeze(1)))
        else:
            A_half = torch.view_as_complex(A_scale * torch.view_as_real(STFT_topk.unsqueeze(1)))
        out_real = torch.zeros(B, self.input_len+self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
        out_imag = torch.zeros(B, self.input_len+self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
        out_real.scatter_(dim=2, index=topk_indices.unsqueeze(1).expand(-1, self.input_len+self.out_len, -1), src=A_half.real)
        out_imag.scatter_(dim=2, index=topk_indices.unsqueeze(1).expand(-1, self.input_len+self.out_len, -1), src=A_half.imag)
        A_half_all = torch.complex(out_real, out_imag)
        _, topk_indices1 = torch.topk( torch.abs(A_half_all) , k=self.topk, dim=2, largest=True)  # [B, K]
        import pdb;pdb.set_trace()
        print(topk_indices[0], topk_indices1[0])
        A_all = construct_hermitian_spectrum(A_half_all, self.input_len)
        all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(0, self.input_len + self.out_len).to(device))  # [B, input_len]
        # Gather selected components
        # selected_omegas = self.omegas[topk_indices]  # [B, K_eff]
        # A_X_sub = torch.gather(A_X_full, dim=2, index=topk_indices)  # [B, N, K]
        # A_X_sub = A_X_full[:, :, topk_indices]  # [B, N, K]
        # A_X_sub = torch.gather(A_X_full, 2, topk_indices.unsqueeze(1).expand(-1, self.input_len, -1))  # [B, input_len, K_eff]
        # Reconstruct history
        # X_rec = synthesize_signal_on_subband(A_X_sub, selected_omegas, self.M, t_in)  # [B, input_len]
        # --- Predict future amplitudes on same frequencies ---
        # A_X_flat = torch.view_as_real(A_X_sub).view(B, -1)  # [B, 2 * input_len * K_eff]
        # A_X_flat =torch.view_as_real(A_X_sub[:, -1, :]).view(B, -1)  # [B, 2 * input_len * K_eff]

        # A_X_last_expanded = A_X_sub[:, -1:, :].expand(-1, self.out_len, -1)  # [B, out_len, K_eff]


        # Predict residual (real+imag)
        # residual_complex = torch.view_as_complex(residual_flat.view(B, self.out_len, self.topk, 2).contiguous())

        # Final prediction: base + residual
        # A_Y_sub =  residual_complex
        
        # Synthesize prediction
        # t_out = t_index_out
        # print(selected_omegas)
        # import pdb;pdb.set_trace()
        # Y_pred = synthesize_signal_on_subband(A_Y_sub, selected_omegas, self.M, t_out)  # [B, out_len]


        # # --- Assemble full-spectrum A for output (for analysis/debug) ---
        # A_full_current = self.evolve_a.A0_full + self.evolve_a.A_delta  # [T_total, M]

        # # Historical A (from current global state)
        # A_X_full = torch.stack([A_full_current[t_idx] for t_idx in t_index_in])  # [B, input_len, M]

        # # Future A: only fill selected bins (others remain 0)
        # A_Y_full = torch.zeros(B, self.out_len, self.M, dtype=torch.complex64, device=device)
        # A_Y_full[..., self.freq_indices] = A_Y_sub

        # Concatenate
        # out = torch.cat([X_rec, Y_pred], dim=1)           # [B, input_len + out_len]
        # A_all_out = torch.cat([A_X_sub, A_Y_sub], dim=1)  # [B, input_len+out_len, K]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            all_rec = all_rec * stdev
            all_rec = all_rec + means


        return all_rec, A_all 
