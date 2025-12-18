import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolveA(nn.Module):
    def __init__(self, T, M, device, init_A):
        super().__init__()
        self.T = T
        self.M = M
        self.A = nn.Parameter(init_A.to(device))

    def forward(self, t_index, w_index=None):
        """
        Output: A_full [B, O] with Hermitian symmetry and real DC/Nyquist.
        """
        return self.A[t_index]


class NeuralEvolutionarySpectra(nn.Module):
    def __init__(self, T,  M, device, init_A, omegas):
        super().__init__()
        self.M = M # total num of freq
        self.omegas = torch.tensor(omegas).to(device).float()
        # else:
        #     k_vals = torch.arange(M, dtype=torch.float32)
        #     omega = 2 * torch.pi * k_vals / M  # [M]
        #     self.omegas = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega).to(device)  # [M]
        self.evolve_a  = EvolveA(T, M, device, init_A)

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
        X_complex = (1.0 / torch.sqrt(torch.tensor(len(self.omegas), dtype=torch.float32, device=device))) * \
                    torch.sum(integrand, dim=-1)  # [B, T]
        # Take real part
        X = X_complex.real  # [B, T]
        return X, A_all



