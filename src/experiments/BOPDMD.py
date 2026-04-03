from dataclasses import dataclass
import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type
from src.experiments.prob_forecast import ProbForecastExp
from src.models.BOPDMD import BOPDMDConfig, BOPDMDModel
from typing import Optional

@dataclass
class Parameters:
    # run.py-style aliases
    task_name: str = "long_term_forecast"
    features: str = "M"  # M|S|MS (only affects how you set columns/scripts)    seq_len: Optional[int] = None
    label_len: int = 0
    seq_len: Optional[int] = None
    learning_rate: Optional[float] = None
    train_epochs: Optional[int] = None
    # BOPDMD hyperparams
    svd_rank: int = 3
    num_trials: int = 100
    trial_size: float = 0.6
    use_proj: bool = True
    eig_constraints: str = "stable"  # "", "stable"
    tol: float = 1e-6
    maxiter: int = 30
    seed: int = 0

@dataclass
class BOPDMDProbForecast(ProbForecastExp, Parameters):
    model_type: str = "BOPDMD"
    def __post_init__(self):
        if self.seq_len is not None:
            self.windows = self.seq_len
        if self.learning_rate is not None:
            self.lr = self.learning_rate
        if self.train_epochs is not None:
            self.epochs = self.train_epochs
    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        # Use ProbForecastExp's time_index batches
        self._init_dataset()
        timeenc = 1  # timeF
        self.scaler = parse_type(self.scaler_type, globals=globals())()
        if self.dataset_type[0:3] == "ETT":
            if self.dataset_type[0:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    freq="h",
                    time_index=True,
                    time_enc=timeenc,
                    shuffle_train=shuffle,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_train=True,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
            else:
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    freq="h",
                    time_index=True,
                    time_enc=timeenc,
                    shuffle_train=shuffle,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_train=True,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                freq="h",
                time_index=True,
                shuffle_train=shuffle,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                fast_train=True,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
                time_enc=timeenc,
                fast_test=fast_test,
                fast_val=fast_val,
            )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
    def _init_model(self):
        constraints = None
        if self.eig_constraints:
            if self.eig_constraints == "stable":
                constraints = {"stable"}
            else:
                raise ValueError(f"Unsupported eig_constraints: {self.eig_constraints}")
        cfg = BOPDMDConfig(
            svd_rank=self.svd_rank,
            num_trials=self.num_trials,
            trial_size=self.trial_size,
            use_proj=self.use_proj,
            eig_constraints=constraints,
            tol=self.tol,
            maxiter=self.maxiter,
            seed=self.seed,
        )
        self.model = BOPDMDModel(seq_len=self.windows, pred_len=self.pred_len, config=cfg).to(self.device)
    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        samples = self.model(batch_x)  # [B, O, N, S]
        return samples.mean(dim=-1), batch_y
    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index, return_A=False):
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        samples = self.model(batch_x)  # [B, O, N, S]
        return samples, batch_y
if __name__ == "__main__":
    import fire
    fire.Fire(BOPDMDProbForecast)
