import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowVaryingESNet(nn.Module):
    def __init__(self, hidden_dim=512, act="relu", sigma_mod=3.0):
        super().__init__()
        # self.omega = torch.tensor(omega_grid, dtype=torch.float32)
        # self.t = torch.tensor(t_grid, dtype=torch.float32)
        self.sigma_mod = sigma_mod  # 控制调制变化快慢

        if act == 'relu':
            self.act_func = nn.ReLU
        elif act == 'tanh':
            self.act_func = nn.Tanh

        # 神经网络预测 "基谱" S0(ω) 和 "调制中心" μ(t)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入 (ω, t)
            self.act_func(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act_func(),
            nn.Linear(hidden_dim, 2),   # 输出: log_S0, μ_mod
            self.act_func(),
        )

    def forward(self, omega, t):
        # 网格化输入
        omega = omega.squeeze().to(self.net[0].weight.device)
        t = t.squeeze().to(self.net[0].weight.device)
        t = t / 1000
        O, T = torch.meshgrid(omega, t, indexing='ij')
        x = torch.stack([O.flatten(), T.flatten()], dim=1)

        # 预测
        out = self.net(x)  # (Nω*Nt, 2)
        log_S0 = torch.sqrt(out[:, 0].reshape(len(omega), len(t)))   # S0(ω)
        mu_mod = out[:, 1].reshape(len(omega), len(t))   # 调制中心

        # 构造 slow-varying 调制函数 A(ω,t)
        # 使用高斯核强制集中在低频 h
        A = torch.exp(log_S0**0.5) * torch.exp(-0.5 * ((T - mu_mod) / self.sigma_mod)**2)
        A = A / (A[:, 0:1] + 1e-8)  # 标准化 A(ω,0) ≈ 1

        S = A ** 2
        return S  # (Nt, Nω)

class NeuralEvolutionrySpectra(nn.Module):
    def __init__(self, hidden_dim, number_of_w=1024, max_w=2*torch.pi/8, act='relu', eps=0):
        super().__init__()

        self.number_of_w = number_of_w
        self.act = act
        # self.Ts = torch.tensor([24, 365])#torch.arange(number_of_w, 0, step=-1) # [number_of_w]

        self.wi = torch.linspace(0, max_w, number_of_w+1)
        self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        self.wi = self.wi[:number_of_w]



        # fixed wi
        # self.Ts = torch.tensor([24, 365])
        # self.wi = torch.concat([torch.zeros(1), (2 * torch.pi/self.Ts)])  # [number_of_w + 1]
        # self.number_of_w = len(self.wi)
        # number_of_w = len(self.wi)
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]

        # uni wi
        # self.wi = torch.linspace(0, 2 * torch.pi, number_of_w+1)
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # self.wi = self.wi[:number_of_w]


        # uni Ts
        # self.Ts = torch.arange(number_of_w, 0, step=-1) # [number_of_w]
        # self.wi = torch.concat([torch.zeros(1), (2 * torch.pi/self.Ts)])  # [number_of_w + 1]
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # self.wi = self.wi[:number_of_w]

        # self.number_of_w = len(self.wi)
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # eps = 0 1e-12
        # self.dw = 1/ (denominator + eps)  # shape: [number_of_w]
        self.spectral_density = SlowVaryingESNet(hidden_dim, act, sigma_mod=3.0)
        self.phase = torch.nn.Parameter(torch.randn(number_of_w))    
        # self.phase = torch.randn(number_of_w)   



    def forward(self, t):
        # w: [number_of_w] N= numberofw - 1
        # t: [B, 1]
        # t = t / 10000
        # w = self.wi[:self.number_of_w].to(t.device)
        w = self.wi.to(t.device)
        B = t.shape[0]

        omegas = w.unsqueeze(0).expand(B, -1).to(t.device) # B, number_of_w
        phase = torch.tanh(self.phase.unsqueeze(0).expand(B, -1).to(t.device))*torch.pi # B, number_of_w
        delta_omega = self.dw.unsqueeze(0).expand(B, -1).to(t.device) # B, number_of_w
        S = self.spectral_density(w, t).transpose(1, 0) # [B, number_of_w]
        allft = torch.sqrt(2*S*delta_omega)*torch.cos(omegas*t + phase) # B, number_of_w
        ft = allft.sum(-1) # B, number_of_w
        return ft

