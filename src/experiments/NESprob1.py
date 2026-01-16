
from dataclasses import dataclass
import sys
# import codecs
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Type, Union
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *
from src.datasets import *
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from torch_timeseries.dataloader import SlidingWindowTS, SlidingWindowTimeIndex, ETTHLoader, ETTMLoader
from torch_timeseries.utils import asdict_exc

import torch
from src.experiments.forecast import ForecastExp
from src.models.NESprob import NeuralEvolutionarySpectraProb


@dataclass
class NESParameters:
    hidden_dim : int = 512

@dataclass
class NESForecast(ForecastExp, NESParameters):
    model_type: str = "NESProb"

    def _init_model(self):
        self.model = NeuralEvolutionarySpectraProb(
            input_len=self.windows,
            pred_len=self.pred_len,
            hidden_dim=self.hidden_dim,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        # x_index = x_index.to(self.device).float().squeeze(-1) 
        # y_index = y_index.to(self.device).float().squeeze(-1) 
        # inp = torch.concat([x_index, y_index], dim=-1).reshape(-1) # B*[L + P]
        # inp = inp.unsqueeze(-1)
        batch_x = batch_x.squeeze(-1)
        mean, var = self.model(batch_x) # [H]
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        return mean, var, batch_y.squeeze(2)

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
                x_index,
                y_index
            ) in enumerate(self.train_loader):
                start = time.time()
                origin_y = origin_y.to(self.device)
                self.model_optim.zero_grad()
                pred, _, true = self._process_one_batch(
                    batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                )
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y
                loss = self.loss_func(pred, true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()

            return train_loss


    def _init_data_loader(self):
        
        self._init_dataset()
        
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
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_index=True,
                    fast_train=False,
                    fast_test=False,
                    fast_val=False,

                )
            elif  self.dataset_type[0:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_index=True,
                    fast_train=False,
                    fast_test=False,
                    fast_val=False,
            )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                shuffle_train=True,
                freq=self.dataset.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
                time_enc=0,
                time_index=True,
                fast_train=False,
                fast_test=False,
                fast_val=False,
            )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")


    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for (
                    batch_x,
                    batch_y,
                    origin_x,
                    origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc,
                    x_index,
                    y_index
                ) in dataloader:
                    start = time.time()
                    origin_y = origin_y.to(self.device)
                    self.model_optim.zero_grad()
                    preds, _, truths = self._process_one_batch(
                        batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                    )
                    # batch_origin_y = batch_origin_y.to(self.device)
                    # if self.invtrans_loss:
                    #     preds = self.scaler.inverse_transform(preds)
                    #     truths = batch_origin_y
                    # if self.pred_len == 1:
                    #     self.metrics.update(
                    #         preds.contiguous().reshape(batch_size, -1),
                    #         truths.contiguous().reshape(batch_size, -1),
                    #     )
                    # else:
                    self.metrics.update(preds.contiguous(), truths.contiguous())

                    progress_bar.update(batch_x.shape[0])

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result
        
    def plot(self):
        # full_dataset = MultivariateFast(
        #     self.dataset,
        #     scaler=self.scaler,
        #     time_enc=3,
        #     window=self.windows,
        #     horizon=self.horizon,
        #     steps=self.pred_len,
        #     include_raw=True,
        #     freq=self.dataset.freq,
        #     single_variate=False,
        #     scaler_fit=False,
        #     time_index=True,
        # )


        full_dataset = MultiStepTimeFeatureSet(
            self.dataset,
            scaler=self.scaler,
            time_enc=3,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            include_raw=True,
            freq=self.dataset.freq,
            single_variate=False,
            scaler_fit=False,
            time_index=True,
        )



        all_batch_x = []
        all_batch_x_date_enc = []
        all_batch_y_date_enc = []
        all_batch_y = []
        
        all_x_index = []
        all_y_index = []
        

        for i in range(0, len(full_dataset), self.pred_len) :
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index = full_dataset[i]
            all_batch_x.append(torch.tensor(batch_x))
            all_batch_y.append(torch.tensor(batch_y))
            all_batch_x_date_enc.append(torch.tensor(batch_x_date_enc))
            all_batch_y_date_enc.append(torch.tensor(batch_y_date_enc))
            
            all_x_index.append(torch.tensor(x_index))
            all_y_index.append(torch.tensor(y_index))
        all_batch_x = torch.stack(all_batch_x, dim=0).to(self.device).float() 
        all_batch_y = torch.stack(all_batch_y, dim=0).to(self.device).float() 
        all_x_date_enc = torch.stack(all_batch_x_date_enc, dim=0).to(self.device).float() 
        all_y_date_enc = torch.stack(all_batch_y_date_enc, dim=0).to(self.device).float()
        all_x_index = torch.stack(all_x_index, dim=0).to(self.device).float()
        all_y_index = torch.stack(all_y_index, dim=0).to(self.device).float()
        # outs = self._val_batch(all_batch_x, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index) # B, T, N
        
        outs, samples, trues = self._process_one_batch(
            all_batch_x, all_batch_y, None, None, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index
        )

        lower = torch.quantile(samples, 0.04, dim=0)   # 10th percentile → lower bound of 80% PI
        upper = torch.quantile(samples, 0.96, dim=0)   # 90th percentile → upper bound of 80% PI
        # fig, axes = plt.subplots(all_batch_x.shape[2])
        # for i in range(all_batch_x.shape[2]):
        #     out = outs[:, :, i]
        #     pred_all = out.reshape(-1).detach().cpu().numpy()
        #     y_all = all_batch_y[:, :, i].reshape(-1).detach().cpu().numpy()
        #     axes[i].plot(pred_all, label='pred')
        #     axes[i].plot(y_all, label='y')
            
        # plt.legend()
        # plt.savefig(os.path.join(self.run_save_dir, 'full.png'))

        # instance
        if len(self.columns) >1:
            last_n = 9
            n = min(all_batch_x.shape[2], 10)
            fig, axes = plt.subplots(n)
            for i in range(n):
                out = outs[-last_n:, :, i]
                pred_all = out.reshape(-1).detach().cpu().numpy()
                y_all = all_batch_y[-last_n:, :, i].reshape(-1).detach().cpu().numpy()
                axes[i].plot(pred_all, label='pred')
                axes[i].plot(y_all, label='y')
                plt.legend()
                plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

        else:
            fig, axes = plt.subplots(1)
            pred_all = outs.squeeze().reshape(-1).detach().cpu().numpy()
            trues = trues.squeeze().reshape(-1).detach().cpu().numpy()
            lower = lower.squeeze().reshape(-1).detach().cpu().numpy()
            upper = upper.squeeze().reshape(-1).detach().cpu().numpy()
        
            # axes.plot(pred_all, label='pred')
            axes.plot(trues, label='y')
            axes.plot(pred_all, label='Mean')
            axes.fill_between(range(len(pred_all)), lower, upper, color='gray', alpha=0.2, label='Standard Error')

            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

            fig, axes = plt.subplots(1)
            axes.plot(pred_all[-500:], label='Mean')
            axes.fill_between(range(len(pred_all[-500:])),lower[-500:], upper[-500:], color='gray', alpha=0.2, label='Standard Error')
            axes.plot(trues[-500:], label='y')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'instance.png'))





    def _test(self):
        super(NESForecast, self)._test()
        self.plot()




if __name__ == "__main__":
    import fire
    fire.Fire(NESForecast)