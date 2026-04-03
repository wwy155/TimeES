from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

@dataclass
class BOPDMDConfig:
    svd_rank: int = 0
    num_trials: int = 0
    trial_size: float = 0.6
    use_proj: bool = True
    eig_constraints: Optional[set] = None  # e.g. {"stable"}
    tol: float = 1e-6
    maxiter: int = 30
    # sampling for probabilistic output
    num_samples: int = 20
    seed: int = 0
    mean_clip: float = 1e6
    std_clip: float = 1e3
class BOPDMDModel(nn.Module):
    """
    Thin torch wrapper around PyDMD's BOPDMD.
    Forward returns samples shaped [B, pred_len, N, S].
    """
    def __init__(self, seq_len: int, pred_len: int, config: BOPDMDConfig):
        super().__init__()

        self.num_trials = 4
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.cfg = config
        # numpy RNG for sampling
        self._rng = np.random.default_rng(self.cfg.seed)
        # Kept so optimizers / DDP see a parameter; forward does not depend on gradients.
        self._dummy = nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)
    def _fit_and_forecast_one(self, x: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        x: (T, N) float
        returns:
          mean: (pred_len, N)
          var:  (pred_len, N) or None
        """
        # local import so the rest of the repo doesn't require pydmd unless used
        from pydmd.bopdmd import BOPDMD
        # PyDMD expects snapshots shaped (space, time)
        X = x.T  # (N, T)
        
        t = np.arange(self.seq_len, dtype=float)/100
        t_future = np.arange(self.seq_len, self.seq_len + self.pred_len, dtype=float)/100
        # varpro_opts_dict is passed to pydmd's BOPDMDOperator (LevMar-style VarPro),
        # NOT scipy.optimize.least_squares. Valid keys per PyDMD docs:
        # init_lambda, maxlam, lamup, use_levmarq, maxiter, tol, eps_stall,
        # use_fulljac, verbose.
        bop = BOPDMD(
            svd_rank=1,
            num_trials=self.num_trials,
            trial_size=1,
            use_proj=self.cfg.use_proj,
            eig_constraints=self.cfg.eig_constraints,
            varpro_opts_dict={
                "tol": self.cfg.tol,
                "maxiter": self.cfg.maxiter,
                "eps_stall": 1e-8,
                "use_fulljac": True,
                "verbose": False,
            },
        )
        bop.fit(X, t)

        all_x = np.empty(
            (bop._num_trials, bop.proj_basis.shape[0], len(t_future)),
            dtype="complex",
        )

        rng = np.random.default_rng(1)
        for k in range(bop._num_trials):
            # Draw eigenvalues and amplitudes from random distribution.
            eigs_k = bop.eigs + np.multiply(
                rng.standard_normal(*bop.eigs.shape),
                bop.operator.eigenvalues_std,
            )
            b_k = bop.amplitudes + np.multiply(
                rng.standard_normal(*bop.amplitudes.shape),
                bop.operator.amplitudes_std,
            )
            # Compute forecast using average modes and eigs_k, b_k.
            all_x[k] = np.linalg.multi_dot(
                [bop.modes, np.diag(b_k), np.exp(np.outer(eigs_k, t_future))]
            )
        # Return the average forecast and the variance.
        # return np.mean(all_x, axis=0), np.var(all_x, axis=0)
        return all_x     # S N P





        # # out = bop.forecast(t_future)
        # if isinstance(out, tuple):
        #     mean, var = out
        # else:
        #     mean, var = out, None
        # # forecast returns (N, pred_len) possibly complex
        # mean = np.real(mean).T  # (pred_len, N)
        # if var is not None:
        #     var = np.real(var).T
        # return mean, var
    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc: [B, T, N]
        returns: samples [B, pred_len, N, S]
        """

        num_trials = 10


        m = x_enc.mean(dim=1, keepdim=True)
        std = x_enc.std(dim=1, keepdim=True)

        x_enc = (x_enc - m) / std

        if x_enc.ndim != 3:
            raise ValueError(f"Expected x_enc [B,T,N], got {tuple(x_enc.shape)}")
        B, T, N = x_enc.shape
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={T}")
        x_np = x_enc.detach().cpu().numpy().astype(np.float64)
        samples = np.empty((B, self.num_trials, N, self.pred_len), dtype=np.float32)
        for b in range(B):
            all_X = self._fit_and_forecast_one(x_np[b])
            samples[b] = all_X
        out = torch.from_numpy(samples).to(device=x_enc.device, dtype=torch.float32).permute(0, 3, 2, 1) # B P N S
        out = out * std.unsqueeze(-1).detach() + m.unsqueeze(-1).detach()
        return out
