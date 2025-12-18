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
    def __init__(self, T, M, device):
        super().__init__()
        self.T = T
        self.M = M

        self.K = M // 2 + 1  # RFFT length

        # Compute minimal number of real parameters per time step
        if M % 2 == 0:
            # Even: DC (1) + Nyquist (1) + (K-2) complex → total = 2 + 2*(K-2) = 2*K - 2
            self.num_real_params_per_t = 2 * self.K - 2
        else:
            # Odd: DC (1) + (K-1) complex → total = 1 + 2*(K-1) = 2*K - 1
            self.num_real_params_per_t = 2 * self.K - 1

        self.A_params = nn.Parameter(torch.randn((T, self.num_real_params_per_t)).to(device))


    # def _build_A(self):
    #     # self.A_half = torch.zeros(self.T, self.K, dtype=torch.complex64)
    #     # if self.M % 2 == 0:
    #     #     # Even M: layout = [DC_real, Nyq_real, Re(f1), Im(f1), ..., Re(f_{K-2}), Im(f_{K-2})]
    #     #     dc_real = self.A_params[:, 0]                     # [T]
    #     #     nyq_real = self.A_params[:, 1]                    # [T]
    #     #     mid = self.A_params[:, 2:].view(T, self.K - 2, 2) # [T, K, 2]

    #     #     self.A_half[:, 0] = dc_real                  # DC
    #     #     self.A_half[:, -1] = nyq_real                # Nyquist
    #     #     self.A_half[:, 1:-1] = torch.complex(mid[:, :, 0], mid[:, :, 1])  # [T, K-2]
    #     # else:
    #     #     # Odd M: layout = [DC_real, Re(f1), Im(f1), ..., Re(f_{K-1}), Im(f_{K-1})]
    #     #     dc_real = self.A_half[:, 0]                     # [T]
    #     #     mid = self.A_half[:, 1:].view(T, self.K - 1, 2) # [T, K-1, 2]

    #     #     self.A_half[:, 0] = dc_real                  # DC
    #     #     self.A_half[:, 1:] = torch.complex(mid[:, :, 0], mid[:, :, 1])   # [B, K-1, O]


    def _build_A_half(self, params):
        """
        Convert real parameters to Hermitian half-spectrum (complex).
        params: [..., num_real] -> A_half: [..., K]
        """
        M = self.M
        K = self.K
        if M % 2 == 0:
            # Layout: [DC, Nyq, Re(f1), Im(f1), ..., Re(f_{K-2}), Im(f_{K-2})]
            dc = params[..., 0]                          # [...]
            nyq = params[..., 1]                         # [...]
            mid = params[..., 2:].view(*params.shape[:-1], K - 2, 2)  # [..., K-2, 2]
            mid_complex = torch.complex(mid[..., 0], mid[..., 1])     # [..., K-2]
            
            # ✅ 使用 torch.cat 构建，避免 in-place
            A_half = torch.cat([
                dc.unsqueeze(-1),
                mid_complex,
                nyq.unsqueeze(-1)
            ], dim=-1)  # [..., K]
        else:
            # Layout: [DC, Re(f1), Im(f1), ..., Re(f_{K-1}), Im(f_{K-1})]
            dc = params[..., 0]                          # [...]
            mid = params[..., 1:].view(*params.shape[:-1], K - 1, 2)  # [..., K-1, 2]
            mid_complex = torch.complex(mid[..., 0], mid[..., 1])     # [..., K-1]
            
            A_half = torch.cat([
                dc.unsqueeze(-1),
                mid_complex
            ], dim=-1)  # [..., K]
        return A_half


    def forward(self, t_index, w_index=None):
        """
        Output: A_full [B, O] with Hermitian symmetry and real DC/Nyquist.
        """
        selected_params = self.A_params[t_index]  # [B, num_real] or [num_real]
        A_half = self._build_A_half(selected_params)  # [B, K]
        A = construct_hermitian_spectrum(A_half, self.M)  # [B, M]
        return A





class NeuralEvolutionarySpectra(nn.Module):
    def __init__(self, T,  M, device):
        super().__init__()
        self.M = M # total num of freq

        k_vals = torch.arange(M, dtype=torch.float32)
        omega = 2 * torch.pi * k_vals / M  # [M]
        self.omegas = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega).to(device)  # [M]
        self.evolve_a  = EvolveA(T, M, device)

    def forward(self, t_index, w_index=None):
        """
        Deterministic prediction with W_k = 1.

        Parameters:
        - x_hist: tensor of shape [B, N] or [B, N, 1], observed history

        Returns:
        - x_pred: predicted future signal, shape [B, H]
        """
        # Frequency grid ω_k = 2πk / M
        device = t_index.device
        t_index = t_index.to(torch.int)
        A_all = self.evolve_a(t_index) # [T, M]
        phase = torch.exp(1j * torch.einsum('t,m->tm', t_index, self.omegas.to(t_index.device)))  # [T, M]

        # Synthesize full signal: X = (1/√M) * sum_k A[n,k] * W_k * exp(i ω_k t_n)
        integrand = A_all * phase  # [T, M]
        X_complex = (1.0 / torch.sqrt(torch.tensor(self.M, dtype=torch.float32, device=device))) * \
                    torch.sum(integrand, dim=-1)  # [B, T]
        # Take real part
        X = X_complex.real  # [B, T]
        return X, A_all



