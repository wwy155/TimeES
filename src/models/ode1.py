import torch
import torch.nn as nn
import torch.fft as fft
from torchdiffeq import odeint

class SpectrumTrajectoryEncoder(nn.Module):
    def __init__(self, input_len, freq_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len * freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, A):  # A: [B, input_len, K]
        x = A.view(A.size(0), -1)
        return self.net(x)  # [B, D]

class SpectralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, h):
        t_scalar = t.reshape(-1)[0]
        t_expand = t_scalar.expand(h.size(0), 1)
        input = torch.cat([h, t_expand], dim=-1)
        return self.net(input)

class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim, freq_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, h_t, omega):
        # h_t: [B, H, D], omega: [K]
        B, H, D = h_t.shape
        K = omega.shape[0]
        omega = omega.view(1, 1, K).expand(B, H, K)  # [B, H, K]
        h_t_expand = h_t.unsqueeze(2).expand(B, H, K, D)  # [B, H, K, D]
        input = torch.cat([h_t_expand, omega.unsqueeze(-1)], dim=-1)  # [B, H, K, D+1]
        out = self.decoder(input.view(B * H * K, -1))  # [B*H*K, 2]
        return out.view(B, H, K, 2)  # [B, H, K, 2]

class NeuralSpectralForecaster(nn.Module):
    def __init__(self, input_len, freq_dim, hidden_dim, freqs):
        super().__init__()
        self.encoder = SpectrumTrajectoryEncoder(input_len, freq_dim, hidden_dim)
        self.ode_func = SpectralODEFunc(hidden_dim)
        self.decoder = SpectralDecoder(hidden_dim, freq_dim)
        self.freqs = freqs

    def forward(self, A_obs, ts_future):
        h0 = self.encoder(A_obs)  # [B, D]
        h_future = odeint(self.ode_func, h0, ts_future).permute(1, 0, 2)  # [B, H, D]
        A_pred = self.decoder(h_future, self.freqs)  # [B, H, K, 2]
        return A_pred

def compute_stft_spectrum(x, n_fft):
    # x: [B, T]
    B, T = x.shape
    # window = torch.hann_window(window_size, device=x.device)
    n = n_fft // 2 + 1
    stft_out = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=24,
        # window=window,
        return_complex=True,
        onesided=True,
        center=False
    )  # [B, n_fft // 2 + 1, W]
    # stft_out = stft_out.permute(0, 2, 1)  # [B, W, n_fft // 2 + 1]
    return stft_out

# === Example Run ===
if __name__ == '__main__':
    B, T = 4, 64
    W = 24
    H = 5
    x = torch.rand(B, T)
    
    n_freq_window = 1 + T//W
    stft_spec = compute_stft_spectrum(x, n_fft=W)  # [B, W // 2 + 1, 1 + L // hop_length(W)]
    
    A_obs = stft_spec[:, :T+1].real  # [B, L+1, K//2+1]
    ts_future = torch.linspace(0, T+H-1, H)

    model = NeuralSpectralForecaster(
        input_len=T,
        freq_dim=W // 2 + 1,
        hidden_dim=64,
        freqs=torch.linspace(0, 1, W // 2 + 1)
    )

    A_pred = model(A_obs[:, :-1], ts_future)  # [B, H, K//2+1, 2]
    print("Predicted spectrum shape:", A_pred.shape)
