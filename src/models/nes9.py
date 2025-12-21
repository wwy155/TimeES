# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EvolveA(nn.Module):
#     def __init__(self, T, M, device, init_A):
#         super().__init__()
#         self.T = T
#         self.M = M
#         self.A0 = init_A.to(device)
#         self.A0.requires_grad = False
#         print(f"A0 requires_grad = {self.A0.requires_grad}")
#         self.A_delta =  nn.Parameter(torch.zeros_like(self.A0)).to(device)

#     def forward(self, t_index, w_index=None):
#         """
#         Output: A_full [B, O] with Hermitian symmetry and real DC/Nyquist.
#         """
#         return self.A[t_index]
        
# class NeuralEolutionarySpectra(nn.Module):
#     def __init__(self, T,  M, device, init_A, omegas, K, input_len, out_len):
#         super().__init__()
#         self.M = M # total num of freq
#         self.omegas = torch.tensor(omegas).to(device).float()
#         # else:
#         #     k_vals = torch.arange(M, dtype=torch.float32)
#         #     omega = 2 * torch.pi * k_vals / M  # [M]
#         #     self.omegas = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega).to(device)  # [M]
#         self.evolve_a  = EvolveA(T, M, device, init_A)
#         self.delta_estimator = nn.Sequential(
#             nn.Linear(1 + len(omegas) + input_len, 512),
#             nn.ReLU(),
#             nn.Linear(512, K*input_len)
#         ) 

#     @property
#     def A(self):
#         return topk(self.A0) + (self.A_delta)


#     def forward(self, X, t_index, topk=None, w_index=None):
#         """
#         Deterministic prediction with W_k = 1.

#         Parameters:
#         - x_hist: tensor of shape [B, N] or [B, N, 1], observed history

#         Returns:
#         - x_pred: predicted future signal, shape [B, H]
#         """
#         # Frequency grid ω_k = 2πk / M
#         device = t_index.device
#         t_index = t_index.to(torch.int)
#         A_all = self.evolve_a(t_index) # [T, M]
#         A_to_use = A_all

#         self.delta_estimator()


#         # # Optional: apply top-k masking per time step
#         if topk is not None and topk < A_all.shape[-1]:
#             # Compute magnitude |A|
#             mag = torch.abs(A_all)  # [..., T, M]

#             # Find top-k indices along frequency axis (dim=-1)
#             _, topk_indices = torch.topk(mag, k=topk, dim=-1, largest=True, sorted=False)

#             # Create a mask of same shape as A_all
#             mask = torch.zeros_like(A_all, dtype=torch.bool)
#             mask.scatter_(dim=-1, index=topk_indices, value=True)

#             # Zero out non-top-k components
#             A_sparse = torch.where(mask, A_all, torch.zeros_like(A_all))
#             A_to_use = A_sparse
#         else:
#             A_to_use = A_all
#         # Phase term: exp(i * ω_k * t_n)
#         # Assume self.omegas: [M]
#         phase = torch.exp(1j * torch.einsum('...t,m->...tm', t_index.float(), self.omegas.to(device)))  # [..., T, M]

#         # Synthesize: X = (1/√M) * Σ_k A(t,k) * exp(i ω_k t)
#         integrand = A_to_use * phase  # [..., T, M]
#         M = len(self.omegas)
#         X_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
#                     torch.sum(integrand, dim=-1)  # [..., T]

#         X = X_complex.real  # [..., T]
#         return X, A_to_use  # return sparse A if applied
import torch
import torch.nn as nn


def synthesize_signal_on_subband(A_vals, selected_omegas, M, t_index):
    """
    Synthesize real-valued signal from complex amplitudes on a subset of frequencies.
    
    Args:
        A_vals: [..., T, K], complex64
        selected_omegas: [K], angular frequencies (rad/sample)
        t_index: [..., T], time indices (long or float)

    Returns:
        x: [..., T], real
    """
    device = A_vals.device
    phase = torch.exp(1j * torch.einsum('...t,k->...tk', t_index.float(), selected_omegas.to(device)))
    integrand = A_vals * phase
    x_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
                torch.sum(integrand, dim=-1)
    return x_complex.real


class GlobalEvolveAOnSubband(nn.Module):
    """
    Amplitude evolution model with full-spectrum storage [T_total, M],
    but only frequencies in `freq_indices` are used and updated.
    """
    def __init__(self, T_total, M, freq_indices, device, init_A):
        super().__init__()
        self.T_total = T_total
        self.M = M
        self.freq_indices = freq_indices  # [K_eff]
        self.K_eff = len(freq_indices)
        self.A = nn.Parameter(init_A.to(device))
        # self.A.requires_grad  = False

    def get_A_sub(self, t_index):
        """
        Return amplitudes at selected frequencies for given time indices.
        
        Args:
            t_index: LongTensor of any shape [...]
        
        Returns:
            A_sub: [..., K_eff], complex64
        """
        return self.A[t_index][..., self.freq_indices]

    def forward(self, t_index):
        return self.A[t_index]


class NeuralEvolutionarySpectra(nn.Module):
    def __init__(
        self,
        T_total,
        input_len,
        out_len,
        M,
        device,
        init_A_full,        # [T_total, M], complex64
        omegas,             # [M], angular frequencies (rad/sample)
        freq_indices,       # [K_eff], long tensor of selected bin indices
        hidden_dim=512
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.M = M
        self.K_eff = len(freq_indices)

        # Register buffers
        self.register_buffer('omegas', torch.tensor(omegas, dtype=torch.float32, device=device))
        self.register_buffer('freq_indices', freq_indices.to(device).long())
        self.register_buffer('selected_omegas', omegas[freq_indices].to(device).float())

        # Global amplitude model (stores full [T_total, M])
        self.evolve_a = GlobalEvolveAOnSubband(T_total, M, freq_indices, device, init_A_full)

        # Predictor: history subband → future subband
        self.A_predictor = nn.Sequential(
            # nn.Linear(2 * input_len * self.K_eff, hidden_dim),
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * out_len * self.K_eff)
        )

        print(f"[NeuralEvolutionarySpectra] Using {self.K_eff} / {M} frequency bins. "
              f"A_delta shape: ({T_total}, {M})")

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

        # --- Get historical amplitudes on selected frequencies ---
        A_X_sub = self.evolve_a.get_A_sub(t_index_in)  # [B, input_len, K_eff]
        t_in = t_index_in
        # Reconstruct history
        X_rec = synthesize_signal_on_subband(A_X_sub, self.selected_omegas, self.M, t_in)  # [B, input_len]
        # --- Predict future amplitudes on same frequencies ---
        A_X_flat = torch.view_as_real(A_X_sub).view(B, -1)  # [B, 2 * input_len * K_eff]

        A_X_last_expanded = A_X_sub[:, -1:, :].expand(-1, self.out_len, -1)  # [B, out_len, K_eff]

        # Predict residual (real+imag)
        residual_flat = self.A_predictor(torch.cat([X], dim=1))  # [B, 2 * out_len * K_eff]
        residual_complex = torch.view_as_complex(residual_flat.view(B, self.out_len, self.K_eff, 2).contiguous())

        # Final prediction: base + residual
        A_Y_sub =  residual_complex
        
        # Synthesize prediction
        t_out = t_index_out
        Y_pred = synthesize_signal_on_subband(A_Y_sub, self.selected_omegas, self.M, t_out)  # [B, out_len]

        # # --- Assemble full-spectrum A for output (for analysis/debug) ---
        # A_full_current = self.evolve_a.A0_full + self.evolve_a.A_delta  # [T_total, M]

        # # Historical A (from current global state)
        # A_X_full = torch.stack([A_full_current[t_idx] for t_idx in t_index_in])  # [B, input_len, M]

        # # Future A: only fill selected bins (others remain 0)
        # A_Y_full = torch.zeros(B, self.out_len, self.M, dtype=torch.complex64, device=device)
        # A_Y_full[..., self.freq_indices] = A_Y_sub

        # Concatenate
        out = torch.cat([X_rec, Y_pred], dim=1)           # [B, input_len + out_len]
        A_all_out = torch.cat([A_X_sub, A_Y_sub], dim=1)  # [B, input_len+out_len, M]

        return out, A_all_out
