import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolveA(nn.Module):
    def __init__(self, hidden_dim=512, x_dim=1, seq_len=None, M=None):
        super().__init__()
        assert seq_len is not None and M is not None, "seq_len (T) and M must be specified"
        self.T = seq_len
        self.M = M
        input_dim = seq_len * x_dim

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * M * seq_len),  # real + imag for A[t,k]
        )

    def forward(self, x):
        B = x.shape[0]
        out = self.net(x)  # [B, 2*M*T]
        out = out.view(B, 2, self.M, self.T)
        real = out[:, 0]   # [B, M, T]
        imag = out[:, 1]   # [B, M, T]
        return torch.complex(real, imag)  # [B, M, T]


class NeuralEvolutionarySpectraProb(nn.Module):
    def __init__(self, input_len, pred_len, hidden_dim, x_dim=1, num_samples=50):
        super().__init__()
        self.N = input_len
        self.H = pred_len
        self.T = input_len + pred_len
        self.M = self.T
        self.num_samples = num_samples

        self.spectral_density = EvolveA(
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            seq_len=self.T,
            M=self.M
        )

        # Learnable parameters for W_k ~ Complex Gaussian
        # We model Re(W_k) and Im(W_k) as independent Gaussians
        self.W_mu_real = nn.Parameter(torch.zeros(self.M))      # [M]
        self.W_mu_imag = nn.Parameter(torch.zeros(self.M))      # [M]
        self.W_logvar_real = nn.Parameter(torch.zeros(self.M))  # log(σ²)
        self.W_logvar_imag = nn.Parameter(torch.zeros(self.M))

    def sample_W(self, B, device):
        """
        Sample W: [num_samples, B, M] (complex)
        Using reparameterization: W = μ + ε * σ, ε ~ N(0,1)
        """
        # Expand to [num_samples, B, M]
        mu_r = self.W_mu_real.unsqueeze(0).unsqueeze(0).expand(self.num_samples, B, -1)
        mu_i = self.W_mu_imag.unsqueeze(0).unsqueeze(0).expand(self.num_samples, B, -1)
        logvar_r = self.W_logvar_real.unsqueeze(0).unsqueeze(0).expand(self.num_samples, B, -1)
        logvar_i = self.W_logvar_imag.unsqueeze(0).unsqueeze(0).expand(self.num_samples, B, -1)

        std_r = torch.exp(0.5 * logvar_r)
        std_i = torch.exp(0.5 * logvar_i)

        eps_r = torch.randn_like(std_r, device=device)
        eps_i = torch.randn_like(std_i, device=device)

        W_real = mu_r + eps_r * std_r  # [num_samples, B, M]
        W_imag = mu_i + eps_i * std_i  # [num_samples, B, M]

        return torch.complex(W_real, W_imag)  # [num_samples, B, M]

    def forward(self, x_hist):
        """
        Probabilistic prediction via sampling W.

        Returns:
          mean_pred: [B, H]
          var_pred:  [B, H]
        """
        device = x_hist.device
        B, N = x_hist.size()[:2]
        H, T, M = self.H, self.T, self.M

        if x_hist.ndim == 2:
            x_hist = x_hist.unsqueeze(-1)  # [B, N, 1]

        # Full time grid
        t_all = torch.arange(T, dtype=torch.float32, device=device)  # [T]

        # Frequency grid
        k_vals = torch.arange(M, dtype=torch.float32, device=device)
        omega = 2 * torch.pi * k_vals / M
        omega = torch.where(omega > torch.pi, omega - 2 * torch.pi, omega)  # [M]

        # Build full x
        x_future_placeholder = torch.zeros(B, H, 1, device=device)
        x_all = torch.cat([x_hist, x_future_placeholder], dim=1)  # [B, T, 1]

        # Get A: [B, M, T] → transpose to [B, T, M] for broadcasting
        A_all = self.spectral_density(x_all)  # [B, M, T]
        A_all = A_all.permute(0, 2, 1)        # [B, T, M]

        # Precompute phase: exp(i ω_k t_n) → [T, M]
        phase = torch.exp(1j * torch.outer(t_all, omega))  # [T, M]

        # Sample W: [num_samples, B, M]
        W_samples = self.sample_W(B, device)  # complex

        # Broadcast A and phase to [num_samples, B, T, M]
        A_expanded = A_all.unsqueeze(0)           # [1, B, T, M]
        phase_expanded = phase.unsqueeze(0).unsqueeze(0)  # [1, 1, T, M]
        W_expanded = W_samples.unsqueeze(2)       # [num_samples, B, 1, M]

        # Synthesize: X = (1/√M) * Σ_k A[n,k] * W[k] * exp(i ω_k t_n)
        integrand = A_expanded * W_expanded * phase_expanded  # [num_samples, B, T, M]
        X_complex = (1.0 / torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))) * \
                    torch.sum(integrand, dim=-1)  # [num_samples, B, T]

        X_real = X_complex.real  # [num_samples, B, T]

        # Extract future part
        X_pred_samples = X_real[:, :, -H:]  # [num_samples, B, H]

        # Compute mean and variance over samples
        mean_pred = X_pred_samples.mean(dim=0)      # [B, H]
        var_pred = X_pred_samples.var(dim=0, unbiased=False)  # [B, H]

        return mean_pred, X_pred_samples