import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

# ODE dynamics for frequency trajectory
class SpectralODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time input
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t, h):
        # h: [B, D], t: scalar
        t_expand = t * torch.ones_like(h[:, :1])
        input = torch.cat([h, t_expand], dim=-1)  # [B, D+1]
        return self.net(input)  # [B, D]

# Decoder: from hidden state + frequency to spectral coefficient
class SpectralDecoder(nn.Module):
    def __init__(self, hidden_dim, freq_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # real and imag parts
        )

    def forward(self, h_t, omega):
        # h_t: [B, D], omega: [K] or [B, K]
        if omega.dim() == 1:
            omega = omega.unsqueeze(0).repeat(h_t.size(0), 1)  # [B, K]
        out = []
        for k in range(omega.size(1)):
            omega_k = omega[:, k].unsqueeze(1)  # [B, 1]
            input = torch.cat([h_t, omega_k], dim=-1)  # [B, D+1]
            out_k = self.decoder(input)  # [B, 2]
            out.append(out_k.unsqueeze(1))
        return torch.cat(out, dim=1)  # [B, K, 2]

# Main model
class NeuralSpectralODE(nn.Module):
    def __init__(self, hidden_dim, freqs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freqs = freqs  # [K]
        self.ode_func = SpectralODEFunc(hidden_dim)
        self.decoder = SpectralDecoder(hidden_dim, len(freqs))

    def forward(self, h0, ts):
        # h0: [B, D], ts: [T]
        
        h_traj = odeint(self.ode_func, h0, ts)  # [T, B, D]
        h_traj = h_traj.permute(1, 0, 2)        # [B, T, D]
        B, T, D = h_traj.shape

        x_recon = []
        for t in range(T):
            h_t = h_traj[:, t, :]              # [B, D]
            spec_coeff = self.decoder(h_t, self.freqs)  # [B, K, 2]
            omega = self.freqs.view(1, -1)     # [1, K]
            time = ts[t].item()
            basis = torch.exp(1j * omega * time)  # [1, K], complex
            spec_complex = spec_coeff[:, :, 0] + 1j * spec_coeff[:, :, 1]
            x_t = (spec_complex * basis).sum(dim=1).real  # [B]
            x_recon.append(x_t.unsqueeze(1))   # [B, 1]
        return torch.cat(x_recon, dim=1)       # [B, T]


# hidden_dim = 32
# K = 64
# freqs = torch.linspace(0.1, 10.0, K)
# ts = torch.linspace(0, 10, 200)

# model = NeuralSpectralODE(hidden_dim, freqs)
# h0 = torch.randn(16, hidden_dim)  # batch size = 16

# x_hat = model(h0, ts)  # output: [16, 200]





import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchdiffeq import odeint

# Define a complex signal with varying frequency and amplitude
def generate_varying_freq_signal(t):
    freq_mod = 2 + 0.5 * np.sin(0.1 * t)
    amp_mod = 1 + 0.3 * np.sin(0.05 * t)
    signal = amp_mod * np.sin(2 * np.pi * freq_mod * t)
    return signal

# Generate the target signal
t_np = np.linspace(0, 10, 200)
target_signal = generate_varying_freq_signal(t_np)

# Plot the target signal
plt.figure(figsize=(10, 4))
plt.plot(t_np, target_signal, label='Varying Frequency Signal')
plt.title('Generated Signal with Time-Varying Frequency and Amplitude')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('data.png')
plt.show()



import torch

ts = torch.linspace(0, 10, 200)
target = torch.tensor(target_signal, dtype=torch.float).unsqueeze(0)  # shape: [1, T]

# 初始化模型
hidden_dim = 32
K = 64
freqs = torch.linspace(0.1, 10.0, K)
model = NeuralSpectralODE(hidden_dim, freqs)

# 随机初始化潜在状态 h0
h0 = torch.randn(1, hidden_dim, requires_grad=True)

# 优化器
optimizer = torch.optim.Adam(list(model.parameters()) + [h0], lr=1e-3)
loss_fn = torch.nn.MSELoss()


for epoch in range(10):
    optimizer.zero_grad()
    x_hat = model(h0, ts)  # output: [1, T]
    loss = loss_fn(x_hat, target)
    loss.backward()
    optimizer.step()
    print(epoch)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

x_hat = model(h0, ts).detach().numpy()[0]

plt.plot(t_np, target_signal, label='True')
plt.plot(t_np, x_hat, label='Predicted', linestyle='--')
plt.title("Signal Reconstruction with NeuralSpectralODE")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig('train.png')
plt.show()


