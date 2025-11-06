import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ================== 稳定版网络 ==================
class StableSlowVaryingESNet(nn.Module):
    def __init__(self, omega_grid, t_grid, sigma_mod=3.0):
        super().__init__()
        self.omega = torch.tensor(omega_grid, dtype=torch.float32)
        self.t = torch.tensor(t_grid, dtype=torch.float32)
        self.sigma_mod = max(sigma_mod, 1.0)  # 防止过窄

        # 输入归一化到 [-1, 1]
        self.omega_norm = (omega_grid - omega_grid.mean()) / (omega_grid.std() + 1e-8)
        self.t_norm = (t_grid - t_grid.mean()) / (t_grid.std() + 1e-8)

        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self):
        device = next(self.net.parameters()).device
        omega = torch.tensor(self.omega_norm, device=device)
        t = torch.tensor(self.t_norm, device=device)

        O, T = torch.meshgrid(omega, t, indexing='ij')
        x = torch.stack([O.flatten(), T.flatten()], dim=1)

        out = self.net(x)  # (Nω*Nt, 2)

        # === 稳定输出处理 ===
        log_S0_raw = out[:, 0].reshape(len(omega), len(t))
        mu_mod_raw = out[:, 1].reshape(len(omega), len(t))

        # 限制 log_S0 范围，防止 exp 爆炸
        log_S0 = 5.0 * torch.tanh(log_S0_raw / 5.0)  # ∈ [-5, 5]
        mu_mod = 2.0 * torch.tanh(mu_mod_raw / 2.0)   # ∈ [-2, 2]（归一化后）

        # 反归一化 mu_mod
        mu_mod = mu_mod * self.t_grid.std() + self.t_grid.mean()
        mu_mod = mu_mod.clamp(self.t_grid[0], self.t_grid[-1])  # 防止越界

        # === 构造 A(ω,t) ===
        S0 = torch.exp(log_S0)  # > 0
        envelope = torch.exp(-0.5 * ((T - mu_mod) / self.sigma_mod)**2)

        A = torch.sqrt(S0) * envelope
        A = A.clamp(min=1e-8)  # 防止为 0

        # === 全局归一化 A(ω,0) ≈ 1 ===
        A0 = A[:, 0:1]  # (Nω, 1)
        A = A / (A0 + 1e-8)  # 防止除零

        S = A ** 2
        return S  # (Nω, Nt)


# ================== 数据准备 ==================
T = 20.0
dt = 0.05
t_grid = np.arange(0, T, dt)
omega_max = 50.0
omega_grid = np.linspace(0, omega_max, 256)

# 构造一个真实演化谱（用于训练）
omega0 = 15.0
bw = 8.0
S0_true = np.exp(-0.5 * ((omega_grid - omega0)/bw)**2)
g_t = np.exp(-0.5 * ((t_grid - T/2)/3.0)**2)
A_true = np.sqrt(S0_true)[:, None] * g_t[None, :]
A_true = A_true / (A_true[:, 0:1] + 1e-12)
S_target = A_true ** 2
S_target = torch.tensor(S_target, dtype=torch.float32)

# ================== 训练 ==================
model = StableSlowVaryingESNet(omega_grid, t_grid, sigma_mod=3.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

losses = []

for epoch in range(10000):
    optimizer.zero_grad()
    S_pred = model()

    loss = F.mse_loss(S_pred, S_target)

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    losses.append(loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.2e}, lr: {optimizer.param_groups[0]['lr']:.2e}")

    # 早停
    if loss.item() < 1e-6:
        print("Converged!")
        break

print("Training finished!")