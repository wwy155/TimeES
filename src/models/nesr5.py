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

        

    def forward(self, t_index, topk=None, w_index=None):
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
        A_to_use = A_all

        # # Optional: apply top-k masking per time step
        if topk is not None and topk < A_all.shape[-1]:
            # Compute magnitude |A|
            mag = torch.abs(A_all)  # [..., T, M]

            # Find top-k indices along frequency axis (dim=-1)
            _, topk_indices = torch.topk(mag, k=topk, dim=-1, largest=True, sorted=False)

            # Create a mask of same shape as A_all
            mask = torch.zeros_like(A_all, dtype=torch.bool)
            mask.scatter_(dim=-1, index=topk_indices, value=True)

            # Zero out non-top-k components
            A_sparse = torch.where(mask, A_all, torch.zeros_like(A_all))
            A_to_use = A_sparse
        else:
            A_to_use = A_all
        # Phase term: exp(i * ω_k * t_n)
        # Assume self.omegas: [M]
        phase = torch.exp(1j * torch.einsum('...t,m->...tm', t_index.float(), self.omegas.to(device)))  # [..., T, M]

        # Synthesize: X = (1/√M) * Σ_k A(t,k) * exp(i ω_k t)
        integrand = A_to_use * phase  # [..., T, M]
        M = len(self.omegas)
        X_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
                    torch.sum(integrand, dim=-1)  # [..., T]

        X = X_complex.real  # [..., T]
        return X, A_to_use  # return sparse A if applied
