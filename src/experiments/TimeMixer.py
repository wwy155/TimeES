from dataclasses import dataclass

import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type

from src.experiments.forecast import ForecastExp
from src.models.TimeMixer import TimeMixer


from typing import Optional, Union

@dataclass
class Parameters:
    # run.py-style aliases / defaults
    task_name: str = "long_term_forecast"
    features: str = "M"  # M|S|MS
    seq_len: Optional[int] = None
    label_len: int = 0  # default 0 for TimeMixer
    learning_rate: Optional[float] = None
    train_epochs: Optional[int] = None

    # model config (mirrors the old `configs.xxx` fields used in TimeMixer)
    use_norm: int = 1
    channel_independence: int = 1
    down_sampling_window: int = 1
    down_sampling_layers: int = 0
    down_sampling_method: str = "avg"  # max|avg|conv|none

    decomp_method: str = "moving_avg"  # moving_avg|dft_decomp
    moving_avg: int = 25
    top_k: int = 5

    d_model: int = 16
    d_ff: int = 32
    e_layers: int = 2
    dropout: float = 0.1

    embed: str = "timeF"
    freq: str = "h"


@dataclass
class TimeMixerForecast(ForecastExp, Parameters):
    model_type: str = "TimeMixer"

    def __post_init__(self):
        # Map run.py-style args onto ForecastExp naming
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

    def _init_model(self):
        enc_in = self.dataset.num_features
        c_out = enc_in

        self.model = TimeMixer(
            task_name=self.task_name,
            seq_len=self.windows,
            label_len=self.label_len,
            pred_len=self.pred_len,
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers,
            down_sampling_method=self.down_sampling_method,
            channel_independence=self.channel_independence,
            e_layers=self.e_layers,
            moving_avg=self.moving_avg,
            enc_in=enc_in,
            c_out=c_out,
            d_model=self.d_model,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
            use_norm=self.use_norm,
            decomp_method=self.decomp_method,
            top_k=self.top_k,
            d_ff=self.d_ff,
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
        batch_x_date_enc = batch_x_date_enc.to(self.device, dtype=torch.float32)
        batch_y_date_enc = batch_y_date_enc.to(self.device, dtype=torch.float32)

        x_dec = torch.zeros_like(batch_y)
        preds = self.model(batch_x, batch_x_date_enc, x_dec, batch_y_date_enc)
        return preds, batch_y


if __name__ == "__main__":
    import fire

    fire.Fire(TimeMixerForecast)
