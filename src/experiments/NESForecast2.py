
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
from src.utils.pesudo_spectrum import get_initial_spectrum_benowitz, get_initial_spectrum_benowitz_targetM
from src.utils.pesudo_amplitude import get_initial_amplitude_right_onesided_mv, get_initial_amplitude_stft_torch
from src.utils.evolutionary_spectra import select_frequencies_by_energy_ratio, select_frequencies_by_energy_ratio_batch
from src.models.nesforecast2 import NeuralEvolutionarySpectra

@dataclass
class NESParameters:
    hidden_dim : int = 512
    additive_scale : bool = False
    use_norm : bool = False
    M : int = 96
    energy_ratio:float = 0.9
    pickout_zero_freq : bool = False
    t_emb : bool = False

@dataclass
class NESForecast(ForecastExp, NESParameters):
    model_type: str = "NESForecast2"

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
            pickout_zero_freq=self.pickout_zero_freq,
        )        
        self.model = self.model.to(self.device)


    def _process_train_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)

        x_index = x_index.to(self.device, dtype=torch.int)
        y_index = y_index.to(self.device, dtype=torch.int)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        # x_index = x_index.to(self.device).float().squeeze(-1) 
        # y_index = y_index.to(self.device).float().squeeze(-1) 
        # inp = torch.concat([x_index, y_index], dim=-1).reshape(-1) # B*[L + P]
        # inp = inp.unsqueeze(-1)
        # y = torch.concat([batch_x, batch_y], dim=1)
        results, A = self.model(batch_x, x_index, y_index, batch_x_date_enc, batch_y_date_enc) # [H]
        results = results.permute(0, 2, 1)
        # results: B N T
        # A: B N T M

        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        # return results, y.squeeze(2), A
        return results[:, -self.pred_len:, :], batch_y, A[:, -1, :, :]


    def _process_one_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index, return_A=False):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)
        x_index = x_index.to(self.device, dtype=torch.int)
        y_index = y_index.to(self.device, dtype=torch.int)
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        # x_index = x_index.to(self.device).float().squeeze(-1) 
        # y_index = y_index.to(self.device).float().squeeze(-1) 
        # inp = torch.concat([x_index, y_index], dim=-1).reshape(-1) # B*[L + P]
        # inp = inp.unsqueeze(-1)
        batch_x = batch_x
        results, A = self.model(batch_x, x_index, y_index, batch_x_date_enc, batch_y_date_enc) # [H]
        results = results.permute(0, 2, 1)
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        if return_A:
            return results[:, -self.pred_len:, :], batch_y, A[:, -1, :, :]
        return results[:, -self.pred_len:, :], batch_y



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
                pred, true, A = self._process_train_batch(
                    batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                )
                loss = self.loss_func(pred, true)
                # print(self.loss_func(pred, true), spectral_entropy_loss(A))
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
                    freq='h',
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
                    freq='h',
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
                freq='h',
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
                    preds, truths = self._process_one_batch(
                        batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                    )
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
        
        outs, trues, A = self._process_one_batch(
            all_batch_x, all_batch_y, None, None, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index, return_A=True
        )

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
        if not self.columns or len(self.columns) >1:
            last_n = 9
            n = min(all_batch_x.shape[2], 10)
            fig, axes = plt.subplots(n)
            for i in range(n):
                out = outs[-last_n:, :, i]
                pred_all = out.reshape(-1).detach().cpu().numpy()
                y_all = all_batch_y[-last_n:, :, i].reshape(-1).detach().cpu().numpy()
                axes[i].plot(pred_all, label='pred')
                # axes[i].plot(y_all, label='y')
                plt.legend()
                plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

        else:
            fig, axes = plt.subplots(1)
            pred_all = outs.squeeze().reshape(-1).detach().cpu().numpy()
            trues = trues.squeeze().reshape(-1).detach().cpu().numpy()
        
            axes.plot(pred_all, label='pred')
            axes.plot(trues, label='y')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

            fig, axes = plt.subplots(1)
            axes.plot(pred_all[-2000:], label='pred')
            # axes.plot(trues[-2000:], label='y')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'instance.png'))


            
            A = A.detach().cpu().numpy()  # shape: [T, M]
            A_Y = A[:, -self.pred_len:, :].reshape(-1, A.shape[-1])  # shape: (A*B, C)
            magnitude = np.abs(A_Y)[:, :20]  # [T, M]
            time_steps = np.arange(A_Y.shape[0])      # [T]
            freq_bins = np.arange(A_Y.shape[1])[:20:]       # [M]
            Time, Freq = np.meshgrid(time_steps, freq_bins, indexing='ij')

            fig = plt.figure(figsize=(12, 12))

            ax2 = fig.add_subplot(211, projection='3d')
            surf = ax2.plot_surface(Time, Freq, magnitude, cmap='viridis', linewidth=0, antialiased=False)
            fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency Bin')
            ax2.set_zlabel('Magnitude')
            ax2.set_title('3D Magnitude Spectrogram (Learned A)')

            ax3 = fig.add_subplot(212)
            im = ax3.imshow(
                magnitude.T,             
                aspect='auto',
                origin='lower',            
                cmap='viridis',
                interpolation='nearest'
            )
            ax3.set_xlabel('Time Index')
            ax3.set_ylabel('Frequency Bin')
            ax3.set_title('Learned Magnitude Spectrogram (|A|)')
            fig.colorbar(im, ax=ax3, label='Magnitude')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'spectrum.png'))

            A_X = A[:, :self.windows, :].reshape(-1, A.shape[-1])  # shape: (A*B, C)
            magnitude = np.abs(A_X)  # [T, M]
            time_steps = np.arange(A_X.shape[0])      # [T]
            freq_bins = np.arange(A_X.shape[1])       # [M]
            Time, Freq = np.meshgrid(time_steps, freq_bins, indexing='ij')
            fig = plt.figure(figsize=(12, 12))

            ax2 = fig.add_subplot(211, projection='3d')
            surf = ax2.plot_surface(Time, Freq, magnitude, cmap='viridis', linewidth=0, antialiased=False)
            fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Frequency Bin')
            ax2.set_zlabel('Magnitude')
            ax2.set_title('3D Magnitude Spectrogram (Learned A)')

            ax3 = fig.add_subplot(212)
            im = ax3.imshow(
                magnitude.T,              
                aspect='auto',
                origin='lower',            
                cmap='viridis',
                interpolation='nearest'
            )
            ax3.set_xlabel('Time Index')
            ax3.set_ylabel('Frequency Bin')
            ax3.set_title('Learned Magnitude Spectrogram (|A|)')
            fig.colorbar(im, ax=ax3, label='Magnitude')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'Xspectrum.png'))




    def _test(self):
        self.plot()
        return super(NESForecast, self)._test()
        











if __name__ == "__main__":
    import fire
    fire.Fire(NESForecast)