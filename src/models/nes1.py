import torch
import torch.nn as nn



def ft_simulation(S, w, t):
    # S: [B, nw, t]
    # w: [B, nw]
    # t: [B, 1]

    torch.sqrt(2)

    pass


# ODE dynamics for frequency trajectory
class SpectralDensity(nn.Module):
    def __init__(self, hidden_dim, number_of_w=1024):
        super().__init__()

        self.t_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),  # +1 for time input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.w_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),  # +1 for time input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # +1 for time input
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.abs = torch.nn.ReLU()

    def forward(self, w, t):
        # w: [number_of_w], t: scalar
        # t: [B, 1]

        # B = t.shape[0]

        
        Bw = w.unsqueeze(-1) # [number_of_w, 1]

        w_pj = self.w_proj(Bw)# [number_of_w, dim]
        t_pj = self.t_proj(t) # [B, dim]


        t_pj = t_pj.unsqueeze(1).expand(t_pj.shape[0], w_pj.shape[0], t_pj.shape[-1]) # [Bt, Bw, dim]
        w_pj = w_pj.unsqueeze(0).expand(t_pj.shape[0],  w_pj.shape[-2],  w_pj.shape[-1]) # [Bt, Bw, dim]
        x = torch.concat([w_pj, t_pj], dim=-1)
        out = self.abs(self.out_head(x)) # [Bt, Bw, 1]
        return out.squeeze(-1)


class NeuralEvolutionrySpectra(nn.Module):
    def __init__(self, hidden_dim, number_of_w=1024, eps=0):
        super().__init__()
        self.spectral_density = SpectralDensity(hidden_dim, number_of_w)
        self.phase = torch.nn.Parameter(torch.randn(number_of_w))     

        self.number_of_w = number_of_w
        self.Ts = torch.arange(number_of_w, 0, step=-1) # [number_of_w]

        # self.Ts = torch.tensor([24, 365])#torch.arange(number_of_w, 0, step=-1) # [number_of_w]
        self.wi = torch.concat([torch.zeros(1), (2 * torch.pi/self.Ts)])  # [number_of_w + 1]

        # self.number_of_w = len(self.wi)
        self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # eps = 0 1e-12
        # self.dw = 1/ (denominator + eps)  # shape: [number_of_w]


    def forward(self, t):
        # w: [number_of_w] N= numberofw - 1
        # t: [B, 1]
        # t = t / 10000
        w = self.wi[:self.number_of_w].to(t.device)
        B = t.shape[0]

        omegas = w.unsqueeze(0).expand(B, -1).to(t.device) # B, number_of_w
        phase = self.phase.unsqueeze(0).expand(B, -1).to(t.device) # B, number_of_w
        delta_omega = self.dw.unsqueeze(0).expand(B, -1).to(t.device) # B, number_of_w

        S = self.spectral_density(w, t) # [B, number_of_w]
        # A = S

        # allft = A*torch.cos(omegas*t + phase) # B, number_of_w
        allft = torch.sqrt(2*S*delta_omega)*torch.cos(omegas*t + phase) # B, number_of_w
        ft = torch.sqrt(torch.tensor(2))*allft.sum(-1) # B, number_of_w
        # x = torch.concat([self.w_proj(w), self.t_proj(t)], dim=-1)
        # out = self.out_head(x) # [B, 1]
        return ft


# ODE dynamics for frequency trajectory
class SpectralODEFunc(nn.Module):
    def __init__(self, hidden_dim, number_of_w=1024):
        super().__init__()



        self.t_proj = nn.Sequential(
            nn.Linear(1, 16),  # +1 for time input
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        self.w_proj = nn.Sequential(
            nn.Linear(number_of_w, number_of_w),  # +1 for time input
            nn.ReLU(),
            nn.Linear(number_of_w, number_of_w)
        )








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
