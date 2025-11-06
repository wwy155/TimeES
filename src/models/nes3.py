import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolveA(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入 (ω, t)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),   # 输出: log_S0, μ_mod
        )

    def forward(self, omega, t):
        """
        Parameters:
        - omega: tensor of shape [M], frequency grid (e.g., in [-π, π])
        - t:     tensor of shape [N], time indices (e.g., [0, 1, ..., N-1])

        Returns:
        - A: complex tensor of shape [N, M], where A[n, m] = A(t_n, omega_m)
        """
        N = t.shape[0]
        M = omega.shape[0]

        # Create all combinations of (omega, t)
        t_grid = t.unsqueeze(1).expand(N, M)          # [N, M]
        omega_grid = omega.unsqueeze(0).expand(N, M)  # [N, M]

        # Stack into [N*M, 2]
        input_pairs = torch.stack([omega_grid, t_grid], dim=-1)  # [N, M, 2]
        input_pairs = input_pairs.view(-1, 2)                    # [N*M, 2]

        # Forward through network
        out = self.net(input_pairs)      # [N*M, 2]
        real = out[:, 0]                 # [N*M]
        imag = out[:, 1]                 # [N*M]

        # Construct complex number
        A_complex = torch.complex(real, imag)  # [N*M]

        # Reshape to [N, M]
        A_complex = A_complex.view(N, M)

        return A_complex

class NeuralEvolutionrySpectra(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim):
        super().__init__()

        self.M = input_len + pred_len
        self.spectral_density = EvolveA(hidden_dim)



    def forward(self, t):
        """
        Deterministic prediction with W_k = 1.

        Parameters:
        - t: tensor of shape [N], time indices (e.g., [0, 1, ..., N-1])

        Returns:
        - Xt: real-valued predicted signal, shape [N]
        """
        t = t.squeeze()
        N = t.shape[0]
        M = self.M

        # Frequency grid: ω_k = 2πk / M for k=0,...,M-1
        k_vals = torch.arange(M, dtype=t.dtype, device=t.device)
        omega = 2 * torch.pi * k_vals / M  # [M]
        # Optional: map to [-π, π) for interpretability (not strictly necessary)
        omega = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega)

        # Get neural amplitude A(t_n, ω_k): shape [N, M] (complex)
        A_complex = self.spectral_density(omega, t/10000)  # [N, M]

        # Set W_k = 1 (deterministic)
        W = torch.ones(M, dtype=torch.complex64, device=t.device)  # [M]

        # Compute phase matrix: exp(i * ω_k * t_n)
        # Use broadcasting: t[:, None] -> [N, 1], omega[None, :] -> [1, M]
        phase = torch.exp(1j * torch.outer(t, omega))  # [N, M]

        # Weighted sum: A * W * phase = A * phase (since W=1)
        integrand = A_complex * phase  # [N, M]

        # Approximate integral via Riemann sum (scaled by 1/sqrt(M))
        Xt_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=t.device))) * \
                     torch.sum(integrand, dim=1)  # [N]

        # Return real part (imaginary should be ~0 if A satisfies Hermitian symmetry,
        # but we don't enforce it here — just take real part for deterministic output)
        return Xt_complex.real  # [N]
