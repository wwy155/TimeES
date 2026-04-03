from dataclasses import dataclass

import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type

from src.experiments.forecast import ForecastExp
from src.models.TimeXer import TimeXer


from typing import Optional, Union

@dataclass
class Parameters:
    # run.py-style aliases / defaults
    task_name: str = "long_term_forecast"
    features: str = "M"  # M|S|MS
    seq_len: Optional[int] = None
    label_len: int = 48  # kept for CLI parity (not used by TimeXer)
    learning_rate: Optional[float] = None
    train_epochs: Optional[int] = None

    # model config (roughly mirroring the "configs.xxx" keys used in the original TimeXer)
    use_norm: bool = False
    patch_len: int = 16
    d_model: int = 128
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    factor: int = 1
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 512
    activation: str = "gelu"


@dataclass
class TimeXerForecast(ForecastExp, Parameters):
    model_type: str = "TimeXer"

    def __post_init__(self):
        # Map run.py-style args onto ForecastExp naming (keep backwards compatibility).
        if self.seq_len is not None:
            self.windows = self.seq_len
        if self.learning_rate is not None:
            self.lr = self.learning_rate
        if self.train_epochs is not None:
            self.epochs = self.train_epochs

    def _init_data_loader(self):
        # Follow TimeES' time-encoding convention, but keep ForecastExp's
        # 6-tensor batch format (no extra time_index fields).
        self._init_dataset()
        timeenc = 0 if self.embed != "timeF" else 1
        if self.dataset_type == "Traffic":
            timeenc = 0
        self.scaler = parse_type(self.scaler_type, globals=globals())()

        if self.dataset_type[0:3] == "ETT":
            if self.dataset_type[0:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    freq=self.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_enc=timeenc,
                )
            else:
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    freq=self.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_enc=timeenc,
                )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                    time_enc=timeenc,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                shuffle_train=True,
                freq=self.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
            )

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

    def _init_model(self):
        if self.windows % self.patch_len != 0:
            raise ValueError(
                f"TimeXer requires windows divisible by patch_len. "
                f"Got windows={self.windows}, patch_len={self.patch_len}."
            )

        enc_in = self.dataset.num_features
        features = self.features
        if features not in {"M", "S", "MS"}:
            raise ValueError(f"features must be one of M/S/MS, got: {features}")

        self.model = TimeXer(
            task_name=self.task_name,
            features=features,
            seq_len=self.windows,
            pred_len=self.pred_len,
            use_norm=self.use_norm,
            patch_len=self.patch_len,
            enc_in=enc_in,
            d_model=self.d_model,
            dropout=self.dropout,
            embed=self.embed,
            freq=self.freq,
            factor=self.factor,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            activation=self.activation,
        ).to(self.device)

    def _process_one_batch(
        self,
        batch_x,
        batch_y,
        batch_origin_x,
        batch_origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ):
        # ForecastExp provides:
        # - batch_x: (B, windows, N)  scaled
        # - batch_y: (B, pred_len, N) scaled
        # - batch_x_date_enc: (B, windows, D)
        # - batch_y_date_enc: (B, pred_len, D)
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device, dtype=torch.float32)
        batch_y_date_enc = batch_y_date_enc.to(self.device, dtype=torch.float32)

        # TimeXer ignores x_dec/x_mark_dec internally, but keep the signature consistent.
        x_dec = torch.zeros_like(batch_y)
        preds = self.model(batch_x, batch_x_date_enc, x_dec, batch_y_date_enc)
        return preds, batch_y


if __name__ == "__main__":
    import fire

    fire.Fire(TimeXerForecast)