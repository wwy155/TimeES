from dataclasses import dataclass
from typing import Optional

import torch
from torch_timeseries.dataloader import ETTHLoader, ETTMLoader, SlidingWindowTS
from torch_timeseries.scaler import *
from torch_timeseries.utils.parse_type import parse_type

from src.experiments.forecast import ForecastExp
from src.models.KNF import KNF, KNFConfig


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

    # KNF config (reasonable defaults from official args.py)
    input_dim: int = 1
    latent_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 5
    add_global_operator: bool = True
    add_control: bool = True
    control_hidden_dim: int = 64
    control_num_layers: int = 3
    use_revin: bool = True
    use_instancenorm: bool = True
    regularize_rank: bool = False
    dropout_rate: float = 0.0

    num_heads: int = 1
    transformer_dim: int = 128
    transformer_num_layers: int = 3

    num_sins: int = -1
    num_poly: int = -1
    num_exp: int = -1


@dataclass
class KNFForecast(ForecastExp, Parameters):
    model_type: str = "KNF"

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

        cfg = KNFConfig(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            encoder_hidden_dim=self.hidden_dim,
            decoder_hidden_dim=self.hidden_dim,
            encoder_num_layers=self.num_layers,
            decoder_num_layers=self.num_layers,
            add_global_operator=self.add_global_operator,
            add_control=self.add_control,
            control_hidden_dim=self.control_hidden_dim,
            control_num_layers=self.control_num_layers,
            use_revin=self.use_revin,
            use_instancenorm=self.use_instancenorm,
            regularize_rank=self.regularize_rank,
            dropout_rate=self.dropout_rate,
            num_heads=self.num_heads,
            transformer_dim=self.transformer_dim,
            transformer_num_layers=self.transformer_num_layers,
            num_sins=self.num_sins,
            num_poly=self.num_poly,
            num_exp=self.num_exp,
        )

        self.model = KNF(seq_len=self.windows, pred_len=self.pred_len, num_feats=enc_in, cfg=cfg).to(self.device)

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

    fire.Fire(KNFForecast)

