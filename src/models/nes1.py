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
    def __init__(self, hidden_dim, number_of_w=1024, single_head=True):
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


        self.single_out = nn.Sequential(
            nn.Linear(2, hidden_dim),  # +1 for time input
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.single_head = single_head
        self.abs = torch.nn.ReLU()
    def forward(self, w, t):
        # w: [number_of_w], t: scalar
        # t: [B, 1]

        # B = t.shape[0]

        
        Bw = w.unsqueeze(-1) # [number_of_w, 1]

        nt = t.shape[0]
        nw = w.shape[0]

        if self.single_head:
            x = torch.stack([
                t.squeeze(-1).unsqueeze(1).expand(nt, nw), 
                Bw.squeeze(-1).unsqueeze(0).expand(nt, nw) 
            ], dim=-1)
            # x= torch.concat([t, Bw], dim=-1)

            out = self.abs(self.single_out(x)) # [Bt, Bw, 1]
            return out.squeeze(-1)

        else:
            w_pj = self.w_proj(Bw)# [number_of_w, dim]
            t_pj = self.t_proj(t) # [B, dim]

            t_pj = t_pj.unsqueeze(1).expand(t_pj.shape[0], w_pj.shape[0], t_pj.shape[-1]) # [Bt, Bw, dim]
            w_pj = w_pj.unsqueeze(0).expand(t_pj.shape[0],  w_pj.shape[-2],  w_pj.shape[-1]) # [Bt, Bw, dim]
            x = torch.concat([w_pj, t_pj], dim=-1)
            out = self.abs(self.out_head(x)) # [Bt, Bw, 1]
            return out.squeeze(-1)


class NeuralEvolutionrySpectra(nn.Module):
    def __init__(self, hidden_dim, number_of_w=1024, single_head=True, eps=0):
        super().__init__()

        self.number_of_w = number_of_w
        self.single_head = single_head

        # self.Ts = torch.tensor([24, 365])#torch.arange(number_of_w, 0, step=-1) # [number_of_w]



        # fixed wi
        # self.Ts = torch.tensor([24, 365])
        # self.wi = torch.concat([torch.zeros(1), (2 * torch.pi/self.Ts)])  # [number_of_w + 1]
        # self.number_of_w = len(self.wi)
        # number_of_w = len(self.wi)
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]

        # uni wi
        self.wi = torch.linspace(0.001, 2 * torch.pi, number_of_w+1)
        self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        self.wi = self.wi[:number_of_w]


        # uni Ts
        # self.Ts = torch.arange(number_of_w, 0, step=-1) # [number_of_w]
        # self.wi = torch.concat([torch.zeros(1), (2 * torch.pi/self.Ts)])  # [number_of_w + 1]
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # self.wi = self.wi[:number_of_w]

        # self.number_of_w = len(self.wi)
        # self.dw = (self.wi)[1:] - (self.wi)[:-1]  # shape: [number_of_w]
        # eps = 0 1e-12
        # self.dw = 1/ (denominator + eps)  # shape: [number_of_w]
        self.spectral_density = SpectralDensity(hidden_dim, number_of_w, single_head=single_head)
        self.phase = torch.nn.Parameter(torch.zeros(number_of_w))    



        self.amps = torch.nn.Parameter(torch.zeros(number_of_w)) 


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
        S = self.spectral_density(w, t) # [B, number_of_w]
        # A = S

        # allft = S*torch.cos(omegas*t + phase) # B, number_of_w
        allft = torch.sqrt(2*S*delta_omega)*torch.cos(omegas*t + phase) # B, number_of_w

        # allft = torch.cos(omegas*t + phase) # B, number_of_w
        ft = torch.sqrt(torch.tensor(2))*allft.sum(-1) # B, number_of_w
        # x = torch.concat([self.w_proj(w), self.t_proj(t)], dim=-1)
        # out = self.out_head(x) # [B, 1]
        return ft

