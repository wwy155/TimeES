from dataclasses import dataclass

import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type

from src.experiments.forecast import ForecastExp
from src.models.TimeBridge import TimeBridge


from typing import Optional, Union

@dataclass
class Parameters:
    # match provided defaults (only keep what model/experiment needs)
    revin: bool = True
    alpha: float = 0.2  # kept for CLI parity (not used in current ForecastExp loss)
    dropout: float = 0.0
    attn_dropout: float = 0.15

    ia_layers: int = 3
    pd_layers: int = 1
    ca_layers: int = 0

    stable_len: int = 6
    num_p: Optional[int] = None
    period: int = 24

    d_model: int = 128
    d_ff: int = 128
    n_heads: int = 8

    embed: str = "timeF"
    freq: str = "h"
    activation: str = "gelu"


@dataclass
class TimeBridgeForecast(ForecastExp, Parameters):
    model_type: str = "TimeBridge"

    # override key training defaults from the provided command
    batch_size: int = 64
    lr: float = 0.0002
    epochs: int = 100
    patience: int = 10

    # keep label_len for CLI parity (not used by model)
    label_len: int = 48

    def _init_data_loader(self):
        # TimeES-style time encoding, but keep ForecastExp's 6-item batches
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

        self.model = TimeBridge(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=enc_in,
            period=self.period,
            num_p=self.num_p,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            ia_layers=self.ia_layers,
            pd_layers=self.pd_layers,
            ca_layers=self.ca_layers,
            stable_len=self.stable_len,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            activation=self.activation,
            revin=self.revin,
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

    fire.Fire(TimeBridgeForecast)

