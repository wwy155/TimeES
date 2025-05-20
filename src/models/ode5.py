import torch
import torch.nn as nn
import torch.fft as fft
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

class SpectrumTrajectoryEncoder(nn.Module):
    def __init__(self, inp_dim, freq_dim, hidden_dim, t_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(freq_dim+inp_dim+t_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, A, x, xt):  # A: [B, 1, K]
        # xt = xt/2000
        inp = torch.concat([A, x, xt], dim=-1)
        return self.net(inp)  # [B, D]

class SpectralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, h):
        # t: scaler
        # h: B,  D
        # t_scalar = t.reshape(-1)[0]
        # t_expand = t_scalar.expand(h.size(0), 1)
        # input = torch.cat([h, t_expand], dim=-1)
        # return self.net(input)

        # t_scalar = t.reshape(-1)[0]
        # t_expand = t_scalar.expand(h.size(0), 1)
        input = torch.cat([h], dim=-1)
        return self.net(input)


class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim, freq_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim)
        )

    def forward(self, h_t):
        # h_t: [B, H, D]
        
        B, H, D = h_t.shape
        out = self.decoder(h_t)  # [B*H*K, 2]
        return out # [B, H, freq_dim]


class NeuralSpectralForecaster(nn.Module):
    def __init__(self, input_len, freq_dim, fft_length, hidden_dim, time_dim, step_size=0.1, use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        self.time_dim = time_dim
        self.fft_length = fft_length
        self.nf = 1 + (input_len - self.fft_length) // self.fft_length

        self.encoder = SpectrumTrajectoryEncoder(self.nf*freq_dim, freq_dim, hidden_dim, time_dim)
        self.ode_func = SpectralODEFunc(hidden_dim)
        self.step_size = step_size
        self.hidden_dim = hidden_dim
        self.decoder = SpectralDecoder(hidden_dim, freq_dim)

    def forward(self, x, xt, xf, A_obs, ts):
        # x: original values [B, N, T]
        # xf: original values [B, N, F]
        # A_obs: fourier spectrum [B, N, fft_length]
        # ts: [t[0](init time), ...] len = fs_O + 1
        xf = xf.reshape(x.shape[0], x.shape[1], -1)
        xt = xt.unsqueeze(1).expand(xt.shape[0], x.shape[1], xt.shape[-1])
        h0 = self.encoder(A_obs, xf, xt)  # [B, N, D]

        # ts = ts.unsqueeze(1).expand(ts.shape[0], h0.shape[1], ts.shape[-1])
        # ts = ts.reshape(-1, ts.shape[-1]) # [B, N, fs_O+1, ]
        # normalized_ts = (ts[0] - ts[0][0])/ts[0][0]

        h0 = h0.reshape(-1, self.hidden_dim)
        
        h_future = odeint(self.ode_func, h0, ts, method='euler', options=dict(step_size=self.step_size)).permute(1, 0, 2)  # [B, fs_O, D]
        A_pred = self.decoder(h_future)  # [B, H, K, 2]
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

    A_mag = A_pred.norm(dim=-1)  # [B, H, K] 复数模长
    for b in range(B):
        plt.imshow(A_mag[b].T.cpu().detach(), aspect='auto', origin='lower', cmap='magma')
        plt.title(f"Predicted Magnitude Spectrum for Sample {b}")
        plt.xlabel("Time Step")
        plt.ylabel("Frequency Bin")
        plt.colorbar(label="Magnitude")
        plt.savefig("s.png")
        plt.show()
