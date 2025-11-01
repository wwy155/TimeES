import torch
import torch.nn as nn
import torch.fft as fft
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

# class SpectrumTrajectoryEncoder(nn.Module):
#     def __init__(self, inp_dim, freq_dim, hidden_dim, t_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(freq_dim+t_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#     def forward(self, A, x, xt):  # A: [B, 1, K]
#         # xt = xt/2000
#         inp = torch.concat([A, x, xt], dim=-1)
#         return self.net(inp)  # [B, D]

class SpectralODEFunc(nn.Module):
    def __init__(self, c_dim, inp_dim, c_emb_dim, hidden_dim):
        # c_dim: num of channels
        super().__init__()
        self.c_dim = c_dim
        self.inp_dim = inp_dim
        self.c_emb_dim =c_emb_dim
        self.l1 = nn.Linear(inp_dim+1+c_emb_dim, hidden_dim).to(torch.cfloat)
        self.l2 = nn.Linear(hidden_dim, inp_dim).to(torch.cfloat)
        
        # if t_emb_dim:
        #     self.t_embd = nn.Embedding(100, t_emb_dim)
        
        self.channel_embedding = nn.Parameter(torch.randn(c_dim, c_emb_dim))
        self.channel_embedding.requires_grad = True

        # self.t_embedding = nn.Embeddings(c_dim, hidden_dim//2)
        
        # self.net = nn.Sequential(
        #     nn.Linear(inp_dim+1, hidden_dim),
        #     nn.ReLU(),
        #     # nn.Linear(hidden_dim, hidden_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(hidden_dim, hidden_dim),
        #     # nn.ReLU(),
        #     nn.Linear(hidden_dim, inp_dim)
        # ).to(torch.cfloat)

    def forward(self, t, h):
        # t: scaler
        # h: B,  D
        
        h = h.reshape(-1, self.c_dim, h.shape[-1])
        emb = self.channel_embedding.expand(h.shape[0], self.channel_embedding.shape[0], self.channel_embedding.shape[1]) # B, C, c_emb_dim
        emb = emb.to(h.device)
        h = torch.concat([h,emb], dim=-1) # B, C, c_emb_dim+f_dim
        h = h.reshape(-1, self.c_emb_dim+self.inp_dim)
        
        t_scalar = t.reshape(-1)[0]
        t_expand = t_scalar.expand(h.size(0), 1)
        input = torch.cat([h, t_expand], dim=-1)
        
        x1 = self.l1(input)
        x2 = torch.complex(torch.relu(x1.real), torch.relu(x1.imag))
        x3 = self.l2(x2)
        
        return x3
    
        # # t_scalar = t.reshape(-1)[0]
        # # t_expand = t_scalar.expand(h.size(0), 1)
        # input = torch.cat([h], dim=-1)
        # return self.net(input)


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
    def __init__(self, fs, out_fftlen, input_len, c_dim, c_emb_dim, freq_dim, fft_length, hidden_dim, time_dim, step_size=0.1, use_norm=True):
        # fs: numer of  frequency spectrum
        super().__init__()
        self.use_norm = use_norm
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.fft_length = fft_length
        
        self.out_fftlen = out_fftlen
        self.fs = fs
        
        
        # self.nf = 1 + (input_len - self.fft_length) // self.fft_length
        
        

        # self.encoder = SpectrumTrajectoryEncoder(self.nf*freq_dim, freq_dim, hidden_dim, time_dim)
        self.ode_func = SpectralODEFunc(c_dim, freq_dim,c_emb_dim,  hidden_dim)
        self.step_size = step_size
        self.hidden_dim = hidden_dim
        
        
        self.projection = nn.Linear(fs*self.freq_dim, self.out_fftlen).to(torch.cfloat)
        
        
        # self.l2 = nn.Linear(hidden_dim, inp_dim).to(torch.cfloat)

        # self.flatten_head = nn.Sequential(
        #         nn.Linear(ifs*self.freq_dim, hidden_dim),
        #         nn.ReLU(),
        #         # nn.Linear(hidden_dim, hidden_dim),
        #         # nn.ReLU(),
        #         # nn.Linear(hidden_dim, hidden_dim),
        #         # nn.ReLU(),
        #         nn.Linear(ofs*, inp_dim)
        # )
        # self.decoder = SpectralDecoder(hidden_dim, freq_dim)

    def forward(self, h0, ts):
        # x: original values [B, N, T]
        # xf: original values [B, N, F]
        # A_obs: fourier spectrum [B, N, fdim]
        # ts: [t[0](init time), ...] len = fs_O + 1
        
        
        
        B, N, F = h0.shape
        h0 = h0.reshape(-1, self.freq_dim) # B*N, fim
        
        h_future = odeint(self.ode_func, h0, ts, method='euler', options=dict(step_size=self.step_size)).permute(1, 0, 2)  # [B*N, fs_O, f_dim]
        
        h_future = h_future.reshape(B, N, h_future.shape[-2]*h_future.shape[-1])
        
        h_future = self.projection(h_future) # B, N, Of
        
        # output = self.flatten_head(h_future) # B, N, F
        
        
        # xf = xf.reshape(x.shape[0], x.shape[1], -1)
        # xt = xt.unsqueeze(1).expand(xt.shape[0], x.shape[1], xt.shape[-1])
        # h0 = self.encoder(A_obs, xf, xt)  # [B, N, D]

        # ts = ts.unsqueeze(1).expand(ts.shape[0], h0.shape[1], ts.shape[-1])
        # ts = ts.reshape(-1, ts.shape[-1]) # [B, N, fs_O+1, ]
        # normalized_ts = (ts[0] - ts[0][0])/ts[0][0]

        # h0 = h0.reshape(-1, self.hidden_dim)
        
        # A_pred = self.decoder(h_future)  # [B, H, K, 2]
        return h_future

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
