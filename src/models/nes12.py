import torch
import torch.nn as nn


def synthesize_signal_on_subband(A_vals, selected_omegas, M, t_index):
    """
    Synthesize real-valued signal from complex amplitudes on a subset of frequencies.
    
    Args:
        A_vals: [..., T, K], complex64
        selected_omegas: [B, K], angular frequencies (rad/sample)
        t_index: [B, T], time indices (long or float)

    Returns:
        x: [..., T], real
    """
    device = A_vals.device
    phase = torch.exp(1j * torch.einsum('...t,...k->...tk', t_index.float(), selected_omegas.to(device)))
    # phase = torch.exp(1j * t_index.unsqueeze(-1) * selected_omegas)
    integrand = A_vals * phase
    # x_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
    #             torch.sum(integrand, dim=-1)
    x_complex =  torch.sum(integrand, dim=-1)
    return x_complex.real


# class GlobalEvolveAOnSubband(nn.Module):
#     """
#     Amplitude evolution model with full-spectrum storage [T_total, M],
#     but only frequencies in `freq_indices` are used and updated.
#     """
#     def __init__(self, device, init_A):
#         super().__init__()
#         self.A = nn.Parameter(init_A.to(device))
#         self.A.requires_grad  = False

#     def forward(self, t_index, freq_indices=None):
#         """
#         Return amplitudes at selected frequencies for given time indices.
        
#         Args:
#             t_index: LongTensor of any shape [...]
        
#         Returns:
#             A_sub: [..., K_eff], complex64
#         """
#         if freq_indices:
#             return self.A[t_index][..., freq_indices]
#         return self.A[t_index]


class NeuralEvolutionarySpectra(nn.Module):
    def __init__(
        self,
        T_total,
        input_len,
        out_len,
        M,
        device,
        hidden_dim=512
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.M = M
        

        # learn M frequencies
        self.omega_nn =  nn.Sequential(
            nn.Linear(input_len  + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, M)
        )

        # Register buffers
        # self.register_buffer('omegas', torch.tensor(omegas, dtype=torch.float32, device=device))
        # self.register_buffer('freq_indices', freq_indices.to(device).long())
        # self.register_buffer('selected_omegas', omegas[freq_indices].to(device).float())

        # Global amplitude model (stores full [T_total, M])
        # self.A = 

        # Predictor: history subband → future subband
        self.A_predictor = nn.Sequential(
            nn.Linear( input_len  +1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (input_len + out_len) * M)
        )

        # print(f"[NeuralEvolutionarySpectra] Using {self.K_eff} / {M} frequency bins. "
        #       f"A_delta shape: ({T_total}, {M})")

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
        # mean = X.mean(dim=-1, keepdim=True)
        # X = X-mean

        omegas = self.omega_nn(torch.concat([X, t_index_in[:, -1:]], dim=-1)) # [B, M]
        A = self.A_predictor(torch.concat([X, t_index_in[:, -1:]], dim=-1)).reshape(B, (self.input_len+self.out_len), self.M, 2) #
        t_all = torch.concat([t_index_in, t_index_out], dim=-1)
        # t_all = torch.arange(self.input_len + self.out_len).to(device)
        all_rec = synthesize_signal_on_subband(torch.view_as_complex(A), omegas, self.M, t_all)  # [B, input_len]

        # all_rec = all_rec+mean


        return all_rec, A

        # --- Get historical amplitudes on selected frequencies ---
        # A_X_full = self.evolve_a(t_index_in)  # [B, input_len, K_eff]
        # t_in = t_index_in
        # topk_indices = self.freq_indices

        # --- 2. Dynamic top-K frequency selection per batch ---
        # energy = torch.mean(torch.abs(A_X_full) ** 2, dim=1)  # [B, M]
        # energy_per_frame = torch.abs(A_X_full)  # [B, N, M]
        # _, topk_indices = torch.topk(energy_per_frame, k=self.topk, dim=2, largest=True)  # [B, N, K]
        # _, topk_indices = torch.topk(energy, k=self.topk, dim=1, largest=True)  # [B, K_eff]
        # topk_indices, _ = torch.sort(topk_indices, dim=2)  


        # Gather selected components
        # selected_omegas = self.omegas[topk_indices]  # [B, K_eff]
        # A_X_sub = torch.gather(A_X_full, dim=2, index=topk_indices)  # [B, N, K]
        # A_X_sub = A_X_full[:, :, topk_indices]  # [B, N, K]
        # A_X_sub = torch.gather(A_X_full, 2, topk_indices.unsqueeze(1).expand(-1, self.input_len, -1))  # [B, input_len, K_eff]
        # Reconstruct history
        # X_rec = synthesize_signal_on_subband(A, omegas, self.M, t_in)  # [B, input_len]
        # --- Predict future amplitudes on same frequencies ---
        # A_X_flat = torch.view_as_real(A_X_sub).view(B, -1)  # [B, 2 * input_len * K_eff]
        # A_X_flat = torch.view_as_real(A_X_sub[:, -1, :]).view(B, -1)  # [B, 2 * input_len * K_eff]

        # A_X_last_expanded = A_X_sub[:, -1:, :].expand(-1, self.out_len, -1)  # [B, out_len, K_eff]


        # Predict residual (real+imag)
        # predict_inp = torch.cat([X], dim=1) # [B, input_len + 2 * input_len * topk + topk]
        # residual_flat = self.A_predictor(predict_inp)  # [B, 2 * out_len * K_eff]
        # residual_complex = torch.view_as_complex(residual_flat.view(B, self.out_len, self.topk, 2).contiguous())

        # Final prediction: base + residual
        # A_Y_sub =  residual_complex
        
        # Synthesize prediction
        # t_out = t_index_out
        # print(selected_omegas)
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

        # return out, A_all_out
