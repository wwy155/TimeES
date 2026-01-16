
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
from src.experiments.prob_forecast import ProbForecastExp
import torch
from src.experiments.forecast import ForecastExp
from src.utils.pesudo_spectrum import get_initial_spectrum_benowitz, get_initial_spectrum_benowitz_targetM
from src.utils.pesudo_amplitude import get_initial_amplitude_right_onesided_mv, get_initial_amplitude_stft_torch
from src.utils.evolutionary_spectra import select_frequencies_by_energy_ratio, select_frequencies_by_energy_ratio_batch
from src.models.NES import NeuralEvolutionarySpectra


@dataclass
class NESParameters:
    hidden_dim : int = 512
    additive_scale : bool = False
    layer_nums : int = 2
    use_norm : bool = False
    M : int = 96
    energy_ratio:float = 0.9
    t_emb : bool = False
    tc_emb : bool = False

@dataclass
class NESProbForecast(ProbForecastExp, NESParameters):
    model_type: str = "NESProbForecast"

    def _init_model(self):
        scaled_data = self.scaler.transform(self.dataset.data)
        #  A_init, omegas  = get_initial_amplitude_stft_torch( this will lead to label leak
        #  A_init, omegas  = get_initial_amplitude_right_stft_torch is ok


        A_train_init, _  = get_initial_amplitude_right_onesided_mv(
            torch.tensor(self.dataloader.train_dataset.scaled_data).transpose(0, 1), 
            n_fft=self.M, 
        ) 
        # A_train_init N, T, M

        energy_per_frame = torch.mean(torch.abs(A_train_init)**2, dim=1)  # [T, M] -> M
        if self.energy_ratio == 1:
            selected_freqs = [torch.arange(0, self.M//2 + 1)]
        else:
            selected_freqs = select_frequencies_by_energy_ratio_batch(energy_per_frame, self.energy_ratio)
        print("selected_freqs:", selected_freqs)

        A_init, omegas  = get_initial_amplitude_right_onesided_mv(
            torch.tensor(scaled_data).transpose(0, 1), 
            n_fft=self.M, 
            return_full_omegas=True,
        )

        A0_torch = torch.tensor(A_init).cfloat()
        # energy_per_frame = torch.mean(torch.abs(A0_torch), dim=0)  # [B, N, M]
        # _, topk_indices = torch.topk(energy_per_frame, k=self.topk, dim=0, largest=True)  # [B, N, K]
        

        self.model = NeuralEvolutionarySpectra(
            self.windows,
            self.dataset.num_features,
            self.pred_len,
            self.device,
            omegas=omegas,
            M=self.M,
            A_init=A0_torch,
            selected_freqs=selected_freqs,
            hidden_dim=self.hidden_dim,
            t_emb=self.t_emb,
            additive_scale=self.additive_scale,
            use_norm=self.use_norm,
            layer_nums=self.layer_nums,
            task='prob_forecast'
        )        
        self.model = self.model.to(self.device)


    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
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
        x_index = x_index.to(self.device, dtype=torch.int)
        y_index = y_index.to(self.device, dtype=torch.int)
        # inp = torch.concat([x_index, y_index], dim=-1).reshape(-1) # B*[L + P]
        # inp = inp.unsqueeze(-1)
        batch_x = batch_x
        preds, A = self.model(batch_x, x_index, y_index, batch_x_date_enc, batch_y_date_enc) # [H]
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        return preds.mean(dim=-1), batch_y
    
    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
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
        x_index = x_index.to(self.device, dtype=torch.int)
        y_index = y_index.to(self.device, dtype=torch.int)
        # inp = torch.concat([x_index, y_index], dim=-1).reshape(-1) # B*[L + P]
        # inp = inp.unsqueeze(-1)
        batch_x = batch_x
        preds, A = self.model(batch_x, x_index, y_index, batch_x_date_enc, batch_y_date_enc) # [H]
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        return preds, batch_y

    def _test(self):
        self.plot()
        return super(NESProbForecast, self)._test()


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

        all_batch_x = all_batch_x[ -4:]
        all_batch_y = all_batch_y[-4:]
        all_x_date_enc = all_x_date_enc[-4:]
        all_y_date_enc = all_y_date_enc[-4:]
        
        all_x_index = all_x_index[-4:]
        all_y_index = all_y_index[-4:]
        
        samples, trues = self._process_val_batch(
            all_batch_x, all_batch_y, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index
        )
        # import pdb;pdb.set_trace()
        outs = samples.mean(-1)
        lower = torch.quantile(samples, 0.04, dim=-1)   # 10th percentile → lower bound of 80% PI
        upper = torch.quantile(samples, 0.96, dim=-1)   # 90th percentile → upper bound of 80% PI
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
        if len(self.columns) >1 or not self.columns:
            n = min(all_batch_x.shape[2], 4)
            fig_glob, axes_glob = plt.subplots(n, figsize=(10, n*2)) 
            fig_inst, axes_inst = plt.subplots(n, figsize=(10, n*2)) 

            # outs = outs[:, :, :n]
            for i in range(n):
                pred_all_i = outs[:, :, i].squeeze().reshape(-1).detach().cpu().numpy()
                trues_i = trues[:, :, i].squeeze().reshape(-1).detach().cpu().numpy()
                lower_i = lower[:, :, i].squeeze().reshape(-1).detach().cpu().numpy()
                upper_i = upper[:, :, i].squeeze().reshape(-1).detach().cpu().numpy()
            
                # axes.plot(pred_all, label='pred')
                axes_glob[i].plot(trues_i, label='y')
                axes_glob[i].plot(pred_all_i, label='Mean')
                axes_glob[i].fill_between(range(len(pred_all_i)), lower_i, upper_i, color='gray', alpha=0.2, label='Prediction Interval')
                axes_glob[i].legend()
                # plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

                axes_inst[i].plot(pred_all_i[-500:], label='Mean')
                axes_inst[i].fill_between(range(len(pred_all_i[-500:])),lower_i[-500:], upper_i[-500:], color='gray', alpha=0.2, label='Prediction Interval')
                axes_inst[i].plot(trues_i[-500:], label='y')
                axes_inst[i].legend()
            fig_glob.tight_layout()
            fig_inst.tight_layout()
            fig_glob.savefig(os.path.join(self.run_save_dir, 'global.png'))
            fig_inst.savefig(os.path.join(self.run_save_dir, 'instance.png'))
            plt.close(fig_glob) 
            plt.close(fig_inst) 
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

            plt.close()


if __name__ == "__main__":
    import fire
    fire.Fire(NESProbForecast)