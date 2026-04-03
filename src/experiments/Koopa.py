from dataclasses import dataclass
from typing import Optional

import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type

from src.experiments.forecast import ForecastExp
from src.models.Koopa import Koopa


@dataclass
class Parameters:
    # run.py-style aliases
    task_name: str = "long_term_forecast"
    features: str = "M"
    seq_len: Optional[int] = None
    label_len: int = 0
    learning_rate: Optional[float] = None
    train_epochs: Optional[int] = None

    # time features
    embed: str = "timeF"
    freq: str = "h"

    # Koopa hyper-params (from thuml/Koopa/run.py)
    dynamic_dim: int = 128
    hidden_dim: int = 64
    hidden_layers: int = 2
    seg_len: int = 48
    num_blocks: int = 3
    alpha: float = 0.2
    multistep: bool = False
    dropout: float = 0.05


@dataclass
class KoopaForecast(ForecastExp, Parameters):
    model_type: str = "Koopa"

    def __post_init__(self):
        if self.seq_len is not None:
            self.windows = self.seq_len
        if self.learning_rate is not None:
            self.lr = self.learning_rate
        if self.train_epochs is not None:
            self.epochs = self.train_epochs

    def _init_data_loader(self):
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

    def _init_model(self):
        enc_in = self.dataset.num_features
        self.model = Koopa(
            enc_in=enc_in,
            seq_len=self.windows,
            pred_len=self.pred_len,
            dynamic_dim=self.dynamic_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            seg_len=self.seg_len,
            num_blocks=self.num_blocks,
            alpha=self.alpha,
            multistep=self.multistep,
            dropout=self.dropout,
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
        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)

        x_dec = torch.zeros_like(batch_y)
        preds = self.model(batch_x, batch_x_date_enc, x_dec, batch_y_date_enc)
        return preds, batch_y


if __name__ == "__main__":
    import fire

    fire.Fire(KoopaForecast)

