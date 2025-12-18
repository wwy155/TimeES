import torch
import torch.nn as nn

# -----------------------------
# Helper Module: InputEncoder
# -----------------------------
class InputEncoder(nn.Module):
    def __init__(self, input_len, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        # x: [B, input_len]
        return self.net(x)  # [B, out_dim]


# ----------------------------------------
# Main Model: NeuralEvolutionarySpectra
# ----------------------------------------
class NeuralEvolutionarySpectra(nn.Module):
    def __init__(self, T, M, device, init_A, omegas, input_len, pred_len):
        """
        Args:
            T: total time length (for normalization)
            M: number of frequency bins
            device: torch device
            init_A: initial complex A0, shape [T, M]
            omegas: angular frequencies, shape [M]
            input_len: length of input history (L)
            pred_len: length of prediction horizon (H) — must be fixed
        """
        super().__init__()
        self.T = T
        self.M = M
        self.input_len = input_len
        self.pred_len = pred_len
        self.omegas = torch.tensor(omegas).to(device).float()
        self.A0_mag = torch.abs(init_A).to(device)  # [T, M]

        self.input_encoder = InputEncoder(input_len, 512, 128)

        # --- Heads for reconstructing X (historical A) ---
        self.A_mag_out = nn.Sequential(
            nn.Linear(self.M + 128 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # ensures > 0
        )
        self.A_phase_out = nn.Sequential(
            nn.Linear(self.M + 128 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # --- Global predictor: A_X (full grid) → A_Y (full future grid) ---
        in_features = 2 * self.M * self.input_len  # [mag; phase] flattened
        hidden_dim = 512

        self.Y_mag_predictor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.M * self.pred_len),
            nn.Softplus()
        )
        self.Y_phase_predictor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.M * self.pred_len)
        )

    def forward(self, x, t_index, y_index, w_index=None):
        """
        Args:
            x: [B, input_len] — observed history
            t_index: [B, input_len] — time indices for X (e.g., 0,1,...,L-1)
            y_index: [B, pred_len] — time indices for Y (e.g., L, L+1, ..., L+H-1)
            w_index: unused

        Returns:
            X: [B, input_len] — reconstructed input
            Y: [B, pred_len] — predicted future
            A_Y: [B, pred_len, M] — predicted evolutionary spectrum for Y
        """
        B = x.shape[0]
        L = self.input_len
        H = self.pred_len
        device = x.device

        assert y_index.shape[1] == H, f"y_index length {y_index.shape[1]} != pred_len {H}"

        # ---------------------------
        # Step 1: Predict A_X
        # ---------------------------
        h = self.input_encoder(x)  # [B, 128]
        h_x = h.unsqueeze(1).expand(-1, L, -1)  # [B, L, 128]
        A0_mag_x = self.A0_mag[:L].unsqueeze(0).expand(B, -1, -1)  # [B, L, M]
        t_norm_x = t_index.float().unsqueeze(-1) / self.T  # [B, L, 1]
        feat_x = torch.cat([A0_mag_x, h_x, t_norm_x], dim=-1)  # [B, L, M+128+1]

        mag_x = self.A_mag_out(feat_x).squeeze(-1)      # [B, L, M]
        phase_x = self.A_phase_out(feat_x).squeeze(-1)  # [B, L, M]
        A_X = torch.polar(mag_x, phase_x)               # [B, L, M]

        # Synthesize X
        int_phase_x = torch.exp(1j * torch.einsum('bt,m->btm', t_index, self.omegas))  # [B, L, M]
        scale = 1.0 / torch.sqrt(torch.tensor(self.M, dtype=torch.float32, device=device))
        X_complex = scale * torch.sum(A_X * int_phase_x, dim=-1)  # [B, L]
        X = X_complex.real

        # ---------------------------
        # Step 2: Predict A_Y from full A_X
        # ---------------------------
        # Flatten magnitude and phase of A_X
        A_X_mag_flat = mag_x.reshape(B, L * self.M)          # [B, L*M]
        A_X_phase_flat = phase_x.reshape(B, L * self.M)      # [B, L*M]
        A_X_repr = torch.cat([A_X_mag_flat, A_X_phase_flat], dim=-1)  # [B, 2*L*M]

        # Predict future magnitude and phase (flattened)
        Y_mag_flat = self.Y_mag_predictor(A_X_repr)     # [B, H*M]
        Y_phase_flat = self.Y_phase_predictor(A_X_repr) # [B, H*M]

        # Reshape to [B, H, M]
        Y_mag = Y_mag_flat.reshape(B, H, self.M)
        Y_phase = Y_phase_flat.reshape(B, H, self.M)
        A_Y = torch.polar(Y_mag, Y_phase)  # [B, H, M]

        # ---------------------------
        # Step 3: Synthesize Y
        # ---------------------------
        int_phase_y = torch.exp(1j * torch.einsum('bh,m->bhm', y_index, self.omegas))  # [B, H, M]
        Y_complex = scale * torch.sum(A_Y * int_phase_y, dim=-1)  # [B, H]
        Y = Y_complex.real

        return X, Y, A_Y




# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class EvolveA(nn.Module):
#     def __init__(self, T, M, device, init_A):
#         super().__init__()
#         self.T = T
#         self.M = M
#         self.A = nn.Parameter(init_A.to(device))

#     def forward(self, t_index, w_index=None):
#         """
#         Output: A_full [B, O] with Hermitian symmetry and real DC/Nyquist.
#         """
#         return self.A[t_index]

# class InputEncoder(nn.Module):
#     def __init__(self, input_len, hidden_dim, out_dim):
#         super().__init__()
#         self.input_len = input_len
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim

#         self.encoder = nn.Sequential(
#             nn.Linear(input_len, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(input_len, out_dim),
#         )

#     def forward(self, x):
#         """
#         Output: [..., T] with Hermitian symmetry and real DC/Nyquist.
#         """
#         return self.encoder(x)


# class NeuralEvolutionarySpectra(nn.Module):
#     def __init__(self, T,  M, device, init_A, omegas, input_len):
#         super().__init__()
#         self.M = M # total num of freq
#         self.omegas = torch.tensor(omegas).to(device).float()
#         self.A0_mag = abs(init_A).to(device) # T, M

#         self.input_encoder = InputEncoder(input_len, 512, 128)


#         self.evolve_a  = EvolveA(T, M, device, init_A)

#         self.A_mag_out = nn.Sequential(
#             nn.Linear(self.M + 128 + 1, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.softplus()
#         )
#         self.A_phase_out = nn.Sequential(
#             nn.Linear(self.M + 128 + 1, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )


#     def forward(self, x, t_index, w_index=None):
#         """
#         Deterministic prediction with W_k = 1.

#         input: A_0, X, t, w

#         Parameters:
#         - x_hist: tensor of shape [B, input_len] or [B, N, 1], observed history

#         Returns:
#         - x_pred: predicted future signal, shape [B, H]
#         """
#         B = x.shape[0]
#         device = x.device

#         # 1. Encode input history
#         h = self.input_encoder(x)  # [B, 128]
#         h = h.unsqueeze(1).expand(-1, self.input_len, -1)  # [B, input_len, 128]

#         # 2. Expand A0_mag to batch dimension
#         A0_mag = self.A0_mag.unsqueeze(0).expand(B, -1, -1)  # [B, input_len, M]

#         # 3. Normalize and expand time index
#         t_norm = t_index.float().unsqueeze(-1) / self.input_len  # [B, input_len, 1] (normalized)

#         # 4. Concatenate features: [A0_mag, h, t_norm]
#         feat = torch.cat([A0_mag, h, t_norm], dim=-1)  # [B, input_len, M + 128 + 1]

#         # 5. Predict magnitude and phase
#         mag = self.A_mag_out(feat).squeeze(-1)      # [B, input_len, M]
#         phase = self.A_phase_out(feat).squeeze(-1)        # [B, input_len, M]

#         # Optional: apply activation for positivity
#         # mag = torch.exp(mag_logit)  # ensures > 0; or use F.softplus, or just mag = mag_logit.abs()

#         # 6. Construct complex A
#         A_all = torch.polar(mag, phase)  # [B, input_len, M]

#         int_phase = torch.exp(1j * torch.einsum('t,m->tm', t_index, self.omegas.to(t_index.device)))  # [T, M]

#         # Synthesize full signal: X = (1/√M) * sum_k A[n,k] * W_k * exp(i ω_k t_n)
#         integrand = A_all * int_phase  # [T, M]
#         X_complex = (1.0 / torch.sqrt(torch.tensor(len(self.omegas), dtype=torch.float32, device=device))) * \
#                     torch.sum(integrand, dim=-1)  # [B, T]
#         # Take real part
#         X = X_complex.real  # [B, T]


#         return X, A_all



