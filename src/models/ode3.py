import torch
import torch.nn as nn
import torch.fft as fft
from torchdiffeq import odeint

class SpectrumTrajectoryEncoder(nn.Module):
    def __init__(self, freq_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, A):  # A: [B, 1, K]
        x = A.squeeze(1)  # [B, K]
        return self.net(x)  # [B, D]

class SpectralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, h):
        t_scalar = t.reshape(-1)[0]
        t_expand = t_scalar.expand(h.size(0), 1)
        input = torch.cat([h, t_expand], dim=-1)
        return self.net(input)

# class SpectralDecoder(nn.Module):
#     def __init__(self, hidden_dim, freq_dim):
#         super().__init__()
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim + 1, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 2)
#         )

#     def forward(self, h_t, omega):
#         # h_t: [B, H, D], omega: [K]
#         B, H, D = h_t.shape
#         K = omega.shape[0]
#         omega = omega.view(1, 1, K).expand(B, H, K)  # [B, H, K]
#         h_t_expand = h_t.unsqueeze(2).expand(B, H, K, D)  # [B, H, K, D]
#         input = torch.cat([h_t_expand, omega.unsqueeze(-1)], dim=-1)  # [B, H, K, D+1]
#         out = self.decoder(input.view(B * H * K, -1))  # [B*H*K, 2]
#         return out.view(B, H, K, 2)  # [B, H, K, 2]

class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim, freq_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim)
        )

    def forward(self, h_t, omega):
        # h_t: [B, H, D], omega: [K]
        
        B, H, D = h_t.shape
        out = self.decoder(h_t)  # [B*H*K, 2]
        return out # [B, H, freq_dim]


class NeuralSpectralForecaster(nn.Module):
    def __init__(self, input_len, freq_dim, hidden_dim, freqs):
        super().__init__()
        self.encoder = SpectrumTrajectoryEncoder(freq_dim, hidden_dim)
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
    B, L = x.shape
    hop_length = n_fft
    stft_out = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,  # non-overlapping chunks
        return_complex=False,
        onesided=True,
        center=False
    )  # [B, n_fft//2+1, 1 + (L - n_fft) // hop_length, 2]
    
    return stft_out.permute(0, 2, 1, 3).reshape(B, 1 + (L - n_fft) // hop_length, (n_fft//2+1)*2)  # [B, num of freq_spectrum, K//2*2+2]

import matplotlib.pyplot as plt

# === Example Run ===
if __name__ == '__main__':
    B, L = 4, 96
    W = 24
    H = 3

    x = torch.rand(B, L)
    stft_spec = compute_stft_spectrum(x, n_fft=W)  # [B, W // 2 + 1, 1 + L // hop_length]
    A_obs = stft_spec[:, 0:1]  # [B, 1, K//2+1] 只使用第一个频谱帧
    ts_future =( torch.arange(1, H + 1) * W).float()  # 对齐 STFT 时间间隔
    model = NeuralSpectralForecaster(
        input_len=L,
        freq_dim=(W // 2 + 1)*2,
        hidden_dim=64,
        freqs=torch.linspace(0, 1, W // 2 + 1)
    )

    A_pred = model(A_obs, ts_future)  # [B, H, K//2+1, 2]
    print("Predicted spectrum shape:", A_pred.shape)

    # 可视化预测频谱轨道的幅值谱
    A_mag = A_pred.norm(dim=-1)  # [B, H, K] 复数模长
    for b in range(B):
        plt.imshow(A_mag[b].T.cpu().detach(), aspect='auto', origin='lower', cmap='magma')
        plt.title(f"Predicted Magnitude Spectrum for Sample {b}")
        plt.xlabel("Time Step")
        plt.ylabel("Frequency Bin")
        plt.colorbar(label="Magnitude")
        plt.savefig("s.png")
        plt.show()
