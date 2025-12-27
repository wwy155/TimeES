import torch
import torch.nn as nn
from src.utils.evolutionary_spectra import construct_hermitian_spectrum, synthesize_signal_on_subband

class NeuralEvolutionarySpectra(nn.Module):
    def __init__(
        self,
        input_len,
        out_len,
        device,
        omegas,
        M,
        A_init,
        cut_off_K=20,
        hidden_dim=512,
        additive_scale=True,
        use_norm=True,
    ):
        super().__init__()
        self.input_len = input_len
        self.out_len = out_len
        self.cut_off_K = cut_off_K
        self.M = M
        self.use_norm = use_norm
        self.omegas =  omegas.to(device).float() #torch.fft.fftfreq(input_len).to(device)
        # Register buffers
        # Predictor: history subband → future subband
        self.A_init = A_init
        self.device = device
        self.fft_len = self.M // 2 + 1
        self.A_scale_predictor = nn.Sequential(
            # nn.Linear(input_len , hidden_dim),
            # nn.Linear(2 * input_len * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * len(freq_indices), hidden_dim),
            # nn.Linear(input_len + 2 * input_len * topk + topk*input_len, hidden_dim),
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (input_len + out_len) * self.cut_off_K)
        )
        self.additive_scale = additive_scale

        self.freq_indices = torch.arange(0, self.cut_off_K).to(self.device)
        self.selected_omegas = self.omegas[self.freq_indices]
        # # zeor frequencies modeling
        # self.zero_frequencies = nn.Sequential(
        #     nn.Linear(input_len, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, input_len + out_len)
        # )


        def syn_real_A(self):
            pass



    def forward(self, X, t_index_in, t_index_out):
        """
        Forward pass.

        Args:
            X: [B, input_len] — dummy input (for interface consistency; not used in computation)
            t_index_in: [input_len] or [B, input_len], long
            t_index_out: [out_len] or [B, out_len], long

        Returns:
            out: [B, input_len + out_len], reconstructed + predicted signal
            A_all_out: [B, input_len + out_len, M], full-spectrum amplitudes (complex)
        """
        B = X.shape[0]
        device = X.device

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = X.mean(-1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            X /= stdev
        
        t_index_stft = t_index_in[:, -1]
        STFT_init_complex = self.A_init[t_index_stft][:,  self.freq_indices] # B, M clofat
        STFT_init = torch.view_as_real(STFT_init_complex) # B, M, 2

        A_scale = self.A_scale_predictor(X).view(B, self.input_len + self.out_len, self.cut_off_K, 2)
        if self.additive_scale:
            A_half = torch.view_as_complex(A_scale + STFT_init.unsqueeze(1))
        else:
            A_half = torch.view_as_complex(A_scale * STFT_init.unsqueeze(1))
        
        out_real = torch.zeros(B, self.input_len+self.out_len, self.fft_len, device=A_half.device, dtype=A_half.real.dtype)
        out_imag = torch.zeros(B, self.input_len+self.out_len, self.fft_len, device=A_half.device, dtype=A_half.imag.dtype)
        # import pdb;pdb.set_trace()
        out_real.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B, self.input_len+self.out_len, -1), src=A_half.real)
        out_imag.scatter_(dim=2, index=self.freq_indices.unsqueeze(0).unsqueeze(0).expand(B, self.input_len+self.out_len, -1), src=A_half.imag)
        A_half_all = torch.complex(out_real, out_imag)
        A_all = construct_hermitian_spectrum(A_half_all, self.M)
        # all_rec = synthesize_signal_on_subband(A_all, self.selected_omegas, self.M, torch.arange(0, self.input_len + self.out_len).to(device))  # [B, input_len]
        all_rec = synthesize_signal_on_subband(A_all, self.omegas, self.M, torch.arange(self.M - self.input_len + 1, self.M+self.out_len+1).to(device))  # [B, input_len]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            all_rec = all_rec * stdev
            all_rec = all_rec + means


        return all_rec, A_all 
