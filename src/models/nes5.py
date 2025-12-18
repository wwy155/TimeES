import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_hermitian_spectrum(A_half: torch.Tensor, M: int) -> torch.Tensor:
    r"""
    Construct a full Hermitian-symmetric complex spectrum of length M
    from the first half (including DC and Nyquist if M is even).

    Input:
        A_half: Tensor of shape (..., K), where
                K = M // 2 + 1   ← frequency dimension must be the LAST dimension
                - A_half[..., 0]       : DC component
                - A_half[..., 1:-1]    : positive frequencies (1 to M//2 - 1)
                - A_half[..., -1]      : Nyquist frequency (only if M is even)

        M: int, total number of frequency bins (FFT size)

    Output:
        A_full: Tensor of shape (..., M) satisfying Hermitian symmetry,
                so that the inverse DFT yields a real-valued signal.
    """
    *dims, K = A_half.shape
    expected_K = M // 2 + 1
    assert K == expected_K, f"Expected A_half last dim = {expected_K}, got {K}"

    if M % 2 == 0:
        # Even M: [DC, f1, ..., f_{M/2-1}, Nyquist]
        if K > 2:
            # Extract positive frequencies excluding DC and Nyquist: indices 1 to -2
            pos_freqs = A_half[..., 1:-1]               # shape: (..., M//2 - 1)
            neg_freqs = torch.conj(pos_freqs.flip(-1))  # reverse order and conjugate
        else:
            # M=2 → K=2 → no middle frequencies
            neg_freqs = torch.empty(*dims, 0, dtype=A_half.dtype, device=A_half.device)
        A_full = torch.cat([A_half, neg_freqs], dim=-1)
    else:
        # Odd M: [DC, f1, ..., f_{(M-1)/2}]
        if K > 1:
            pos_freqs = A_half[..., 1:]                 # all except DC
            neg_freqs = torch.conj(pos_freqs.flip(-1))
        else:
            # M=1 → K=1 → only DC
            neg_freqs = torch.empty(*dims, 0, dtype=A_half.dtype, device=A_half.device)
        A_full = torch.cat([A_half, neg_freqs], dim=-1)

    # Final sanity check
    assert A_full.shape[-1] == M, f"Output length {A_full.shape[-1]} != M={M}"
    return A_full


class EvolveA(nn.Module):
    def __init__(self, hidden_dim=512, x_dim=1, seq_len=None, out_len=None, M=None):
        super().__init__()
        assert seq_len is not None and M is not None and out_len is not None
        self.T = seq_len
        self.O = out_len
        self.M = M

        self.K = M // 2 + 1  # RFFT length

        # Compute minimal number of real parameters per time step
        if M % 2 == 0:
            # Even: DC (1) + Nyquist (1) + (K-2) complex → total = 2 + 2*(K-2) = 2*K - 2
            self.num_real_params_per_t = 2 * self.K - 2
        else:
            # Odd: DC (1) + (K-1) complex → total = 1 + 2*(K-1) = 2*K - 1
            self.num_real_params_per_t = 2 * self.K - 1

        input_dim = seq_len * x_dim
        total_output_dim = self.num_real_params_per_t * self.O

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, total_output_dim),
        )

    def forward(self, x):
        """
        Output: A_full [B, M, O] with Hermitian symmetry and real DC/Nyquist.
        """
        B = x.shape[0]
        out = self.net(x)  # [B, total_output_dim]

        # Reshape to [B, O, num_real_params_per_t] for easier indexing
        out = out.view(B, self.O, self.num_real_params_per_t)

        # Build A_half [B, O, K] as complex
        A_half = torch.zeros(B, self.K, self.O, dtype=torch.complex64, device=x.device)

        if self.M % 2 == 0:
            # Even M: layout = [DC_real, Nyq_real, Re(f1), Im(f1), ..., Re(f_{K-2}), Im(f_{K-2})]
            dc_real = out[:, :, 0]                     # [B, O]
            nyq_real = out[:, :, 1]                    # [B, O]
            mid = out[:, :, 2:].view(B, self.O, -1, 2) # [B, O, K-2, 2]

            A_half[:, 0, :] = dc_real                  # DC
            A_half[:, -1, :] = nyq_real                # Nyquist
            A_half[:, 1:-1, :] = torch.complex(mid[:, :, :, 0], mid[:, :, :, 1]).permute(0, 2, 1)  # [B, K-2, O]
        else:
            # Odd M: layout = [DC_real, Re(f1), Im(f1), ..., Re(f_{K-1}), Im(f_{K-1})]
            dc_real = out[:, :, 0]                     # [B, O]
            mid = out[:, :, 1:].view(B, self.O, -1, 2) # [B, O, K-1, 2]

            A_half[:, 0, :] = dc_real                  # DC
            A_half[:, 1:, :] = torch.complex(mid[:, :, :, 0], mid[:, :, :, 1]).permute(0, 2, 1)   # [B, K-1, O]

        # Now A_half has real DC (and Nyquist if even), rest complex
        A_full = construct_hermitian_spectrum(A_half.permute(0, 2, 1), self.M)  # [B, O, M]
        return A_full # [B, M, O]



class NeuralEvolutionarySpectra(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim, K=None, x_dim=1):
        super().__init__()
        self.N = input_len
        self.H = pred_len
        self.T = input_len + pred_len

        self.M = input_len + pred_len # total num of freq

        # if K==None:
        #     K = self.M//2+1
        self.spectral_density = EvolveA(hidden_dim=hidden_dim, x_dim=x_dim, seq_len=input_len, out_len=input_len+pred_len, M=input_len+pred_len)

    def forward(self, x_hist, return_A=False):
        """
        Deterministic prediction with W_k = 1.

        Parameters:
        - x_hist: tensor of shape [B, N] or [B, N, 1], observed history

        Returns:
        - x_pred: predicted future signal, shape [B, H]
        """
        device = x_hist.device
        B, N = x_hist.size()[:2]
        H, T, M = self.H, self.T, self.M

        # Ensure x_hist is [B, N, 1]
        if x_hist.ndim == 2:
            x_hist = x_hist.unsqueeze(-1)  # [B, N, 1]

        # Time indices for full sequence
        t_all = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0).expand(B, T)  # [B, T]

        # Frequency grid ω_k = 2πk / M
        k_vals = torch.arange(M, dtype=torch.float32, device=device)
        omega = 2 * torch.pi * k_vals / M  # [M]
        omega = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega)  # Optional shift
        omega = omega.unsqueeze(0).expand(B, M)  # [B, M]


        # Get A(t, ω) for all t in [0, T-1] and all ω
        A_all = self.spectral_density(x_hist)  # [B, T, M]

        # Set W_k = 1 (deterministic)
        W = torch.ones(M, dtype=torch.complex64, device=device).unsqueeze(0).expand(B, M)  # [B, M]

        # Phase matrix: exp(i * ω_k * t_n) for all n,k
        phase = torch.exp(1j * torch.einsum('bt,bm->btm', t_all, omega))  # [B, T, M]

        # Synthesize full signal: X = (1/√M) * sum_k A[n,k] * W_k * exp(i ω_k t_n)
        integrand = A_all * phase  # [B, T, M]
        X_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
                    torch.sum(integrand, dim=-1)  # [B, T]
        # Take real part
        X_real = X_complex.real  # [B, T]

        # Return only future part
        x_pred = X_real[:, -H:]  # [B, H]
        if return_A:
            return x_pred, A_all
        return x_pred







# class EvolveA(nn.Module):
#     def __init__(self, hidden_dim=512, x_dim=1):
#         super().__init__()
#         # Input: [omega, t, x (dim=x_dim), mask]
#         input_dim = 2 + x_dim + 1
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2),  # real, imag
#         )

#     def forward(self, omega, t, x, mask=None):
#         """
#         Parameters:
#         - omega: [M]
#         - t:     [N]
#         - x:     [N, x_dim]
#         - mask:  [N] or None (bool or float); if None, treated as all ones

#         Returns:
#         - A_complex: [N, M]
#         """
#         N = t.shape[0]
#         M = omega.shape[0]
#         x_dim = x.shape[1]

#         # Expand to [N, M]
#         t_grid = t.unsqueeze(1).expand(N, M)          # [N, M]
#         omega_grid = omega.unsqueeze(0).expand(N, M)  # [N, M]
#         x_grid = x.unsqueeze(1).expand(N, M, x_dim)   # [N, M, x_dim]

#         # Handle mask
#         if mask is None:
#             mask_grid = torch.ones(N, M, 1, device=x.device)
#         else:
#             mask_grid = mask.float().unsqueeze(1).unsqueeze(2).expand(N, M, 1)  # [N, M, 1]

#         # Concatenate all inputs: [N, M, 2 + x_dim + 1]
#         input_tensor = torch.cat([
#             omega_grid.unsqueeze(-1),      # [N, M, 1]
#             t_grid.unsqueeze(-1),          # [N, M, 1]
#             x_grid,                        # [N, M, x_dim]
#             mask_grid                      # [N, M, 1]
#         ], dim=-1)

#         # Flatten to [N*M, ...]
#         input_flat = input_tensor.view(-1, 2 + x_dim + 1)  # [N*M, input_dim]

#         # Forward through MLP
#         out = self.net(input_flat)  # [N*M, 2]
#         real = out[:, 0]
#         imag = out[:, 1]

#         # Reshape to [N, M]
#         A_complex = torch.complex(real, imag).view(N, M)

#         return A_complex

# class NeuralEvolutionarySpectra(nn.Module):
#     def __init__(self, input_len, pred_len, hidden_dim, x_dim=1):
#         super().__init__()
#         self.N = input_len      # history length
#         self.H = pred_len       # prediction horizon
#         self.T = input_len + pred_len
#         self.M = self.T         # number of frequency bins

#         # Use your MLP-based EvolveA that takes (omega, t, x, mask)
#         self.spectral_density = EvolveA(hidden_dim=hidden_dim, x_dim=x_dim)

#     def forward(self, x_hist):
#         """
#         Deterministic prediction with W_k = 1.

#         Parameters:
#         - x_hist: tensor of shape [N] or [N, 1], observed history

#         Returns:
#         - x_pred: predicted future signal, shape [H]
#         """
#         device = x_hist.device
#         N, H, T, M = self.N, self.H, self.T, self.M

#         # Ensure x_hist is [N, 1]
#         if x_hist.ndim == 1:
#             x_hist = x_hist.unsqueeze(-1)  # [N, 1]

#         # Time indices for full sequence
#         t_all = torch.arange(T, dtype=torch.float32, device=device)  # [T]

#         # Frequency grid ω_k = 2πk / M
#         k_vals = torch.arange(M, dtype=torch.float32, device=device)
#         omega = 2 * torch.pi * k_vals / M  # [M]
#         # Optional: shift to [-π, π)
#         omega = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega)

#         # Build full x input: [T, 1]
#         x_future_placeholder = torch.zeros(H, 1, device=device)
#         print(x_hist.shape, x_future_placeholder.shape)
#         x_all = torch.cat([x_hist, x_future_placeholder], dim=0)  # [T, 1]

#         # Build mask: 1 for history, 0 for future
#         mask_hist = torch.ones(N, dtype=torch.bool, device=device)
#         mask_fut = torch.zeros(H, dtype=torch.bool, device=device)
#         mask_all = torch.cat([mask_hist, mask_fut], dim=0)  # [T]

#         # Get A(t, ω) for all t in [0, T-1] and all ω
#         A_all = self.spectral_density(omega, t_all, x_all, mask_all)  # [T, M]

#         # Set W_k = 1 (deterministic)
#         W = torch.ones(M, dtype=torch.complex64, device=device)  # [M]

#         # Phase matrix: exp(i * ω_k * t_n) for all n,k
#         phase = torch.exp(1j * torch.outer(t_all, omega))  # [T, M]

#         # Synthesize full signal: X = (1/√M) * sum_k A[n,k] * W_k * exp(i ω_k t_n)
#         integrand = A_all * phase  # [T, M] (W=1)
#         X_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
#                     torch.sum(integrand, dim=1)  # [T]

#         # Take real part
#         X_real = X_complex.real  # [T]

#         # Return only future part
#         x_pred = X_real[-H:]  # [H]

#         return x_pred
# import torch
# import torch.nn as nn

# class EvolveA(nn.Module):
#     def __init__(self, seq_len, hidden_dim=512, x_dim=1):
#         super().__init__()
#         input_dim = 2 + seq_len * x_dim + 1
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2),  # real, imag
#         )

#     def forward(self, omega, t, x, mask=None):
#         """
#         Parameters:
#         - omega: [B, M]
#         - t:     [B, N]
#         - x:     [B, N, x_dim]
#         - mask:  [B, N] or None (bool or float); if None, treated as all ones

#         Returns:
#         - A_complex: [B, N, M]
#         """
#         B, N, x_dim = x.shape
#         M = omega.shape[1]
#         x_flat = x.view(B, N * x_dim)  # [B, N*x_dim]

#         # Expand to [B, N, M]
#         t_grid = t.unsqueeze(-1).expand(B, N, M)          # [B, N, M]
#         omega_grid = omega.unsqueeze(1).expand(B, N, M)   # [B, N, M]
#         x_grid = x_flat.unsqueeze(1).unsqueeze(2).expand(B, N, M, N * x_dim)  # [B, N, M, N*x_dim]

#         if mask is None:
#             mask_grid = torch.ones(B, N, M, 1, device=x.device)
#         else:
#             mask_grid = mask.float().unsqueeze(-1).unsqueeze(-1).expand(B, N, M, 1)  # [B, N, M, 1]

#         # Concatenate: [omega, t, full_x, mask]
#         input_tensor = torch.cat([
#             omega_grid.unsqueeze(-1),      # [B, N, M, 1]
#             t_grid.unsqueeze(-1),          # [B, N, M, 1]
#             x_grid,                        # [B, N, M, N*x_dim]
#             mask_grid                      # [B, N, M, 1]
#         ], dim=-1)  # [B, N, M, 2 + N*x_dim + 1]

#         # Flatten batch dimensions for MLP
#         input_flat = input_tensor.view(-1, 2 + N * x_dim + 1)  # [B*N*M, input_dim]

#         # Forward through MLP
#         out = self.net(input_flat)  # [B*N*M, 2]
#         real = out[:, 0]
#         imag = out[:, 1]

#         # Reshape back
#         A_complex = torch.complex(real, imag).view(B, N, M)

#         return A_complex


# class EvolveA(nn.Module):
#     def __init__(self, hidden_dim=512, x_dim=1, seq_len=None):
#         super().__init__()
#         assert seq_len is not None, "seq_len (T) must be provided"
#         self.seq_len = seq_len
#         self.x_dim = x_dim

#         # Input dimension: [t, omega, full_x (seq_len * x_dim), mask]
#         input_dim = 1 + 1 + seq_len * x_dim + 1  # = 3 + seq_len * x_dim
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2),  # real, imag
#         )

#     def forward(self, omega, t, x, mask=None):
#         """
#         Parameters:
#         - omega: [B, M]
#         - t:     [B, T]
#         - x:     [B, T, x_dim]   ← full sequence (history + future placeholder)
#         - mask:  [B, T] or None

#         Returns:
#         - A_complex: [B, T, M]
#         """
#         B, T, x_dim = x.shape
#         M = omega.shape[1]

#         # Flatten full x for each batch: [B, T*x_dim]
#         x_flat = x.view(B, T * x_dim)  # [B, T*x_dim]

#         # Repeat x_flat for each (t_n, omega_m) → [B, T, M, T*x_dim]
#         # But we avoid expand by using indexing below

#         # Create index grids
#         b_idx = torch.arange(B, device=x.device)
#         t_idx = torch.arange(T, device=x.device)
#         m_idx = torch.arange(M, device=x.device)

#         # Meshgrid to get all combinations (b, n, m)
#         b_grid, t_grid, m_grid = torch.meshgrid(b_idx, t_idx, m_idx, indexing='ij')  # [B, T, M]

#         # Gather values
#         t_vals = t[b_grid, t_grid]          # [B, T, M]
#         omega_vals = omega[b_grid, m_grid]  # [B, T, M]
#         mask_vals = mask[b_grid, t_grid] if mask is not None else torch.ones_like(t_vals)  # [B, T, M]

#         # Expand x_flat to [B, T, M, T*x_dim] without .expand()
#         # Instead: repeat along T and M dimensions
#         x_repeated = x_flat.unsqueeze(1).unsqueeze(2).repeat(1, T, M, 1)  # [B, T, M, T*x_dim]
#         # Note: .repeat() creates new memory, but avoids symbolic expand; if you truly want no repeat,
#         # you'd have to flatten first — but this is the cleanest way.

#         # Alternatively, flatten everything early:
#         # We'll go with explicit flat construction for maximum efficiency:

#         # --- 更高效的做法：直接构造 flat 输入 ---
#         # Repeat x_flat for each (n,m): total B*T*M rows
#         x_input = x_flat.repeat_interleave(T * M, dim=0)  # [B*T*M, T*x_dim]

#         # Time: repeat each t[b, n] for M times → shape [B*T*M]
#         t_input = t.unsqueeze(-1).repeat(1, 1, M).view(-1)  # [B*T*M]

#         # Omega: repeat each omega[b, m] for T times per batch → [B, M, T] then flatten
#         omega_input = omega.unsqueeze(1).repeat(1, T, 1).view(-1)  # [B*T*M]

#         # Mask
#         if mask is not None:
#             mask_input = mask.unsqueeze(-1).repeat(1, 1, M).view(-1, 1)  # [B*T*M, 1]
#         else:
#             mask_input = torch.ones(B * T * M, 1, device=x.device)

#         # Concatenate all: [B*T*M, D]
#         input_flat = torch.cat([
#             t_input.unsqueeze(-1),      # [B*T*M, 1]
#             omega_input.unsqueeze(-1),  # [B*T*M, 1]
#             x_input,                    # [B*T*M, T*x_dim]
#             mask_input                  # [B*T*M, 1]
#         ], dim=-1)  # [B*T*M, 3 + T*x_dim]

#         # Forward through MLP
#         out = self.net(input_flat)  # [B*T*M, 2]

#         # Reshape to [B, T, M]
#         A_real = out[:, 0].view(B, T, M)
#         A_imag = out[:, 1].view(B, T, M)
#         A_complex = torch.complex(A_real, A_imag)

#         return A_complex

