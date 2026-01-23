
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
from torch_timeseries.experiments import UEAClassificationExp
from src.utils.pesudo_spectrum import get_initial_spectrum_benowitz, get_initial_spectrum_benowitz_targetM
from src.utils.pesudo_amplitude import get_initial_amplitude_right_onesided_mv, get_initial_amplitude_stft_torch
from src.utils.evolutionary_spectra import select_frequencies_by_energy_ratio, select_frequencies_by_energy_ratio_batch
from src.models.NES import NeuralEvolutionarySpectra

@dataclass
class NESParameters:
    hidden_dim : int = 512
    additive_scale : bool = False
    use_norm : bool = False
    energy_ratio:float = 0.9
    pickout_zero_freq : bool = False
    t_emb : bool = False
    rec_X : bool = True

@dataclass
class NESClassification(UEAClassificationExp, NESParameters):
    model_type: str = "NESClassification"
    windows : int = 336
    pred_len : int = 336
    patience : int = 10 
    lr : float = 0.0001


    
    def _init_model(self):

        selected_freqs = self.select_frequency()
        self.model = NeuralEvolutionarySpectra(
            self.windows,
            self.dataset.num_features,
            self.windows,
            self.device,
            omegas=torch.fft.fftfreq(self.windows).to(self.device),
            M=self.windows,
            A_init=None,
            selected_freqs=selected_freqs,
            hidden_dim=self.hidden_dim,
            t_emb=self.t_emb,
            additive_scale=self.additive_scale,
            use_norm=self.use_norm,
            task='classification',
            out_prob=self.dataset.num_classes,
        )        
        self.model = self.model.to(self.device)


    def select_frequency(self):
        all_x = []
        for i, (scaled_x, x, y, padding_masks) in enumerate(self.train_loader):
            all_x.append(x)
        X = torch.concat(all_x, dim=0)
        X = X.permute(2, 0, 1)
        STFT_init_complex = torch.fft.rfft(X, norm="ortho").cfloat() # B, T, M
        energy_per_frame = torch.mean(torch.abs(STFT_init_complex)**2, dim=1)  # [T, M] -> M
        if self.energy_ratio == 1:
            selected_freqs = [torch.arange(0, self.windows//2 + 1)]
        else:
            selected_freqs = select_frequencies_by_energy_ratio_batch(energy_per_frame, self.energy_ratio)
        return selected_freqs


    def _process_train_batch(self, scaled_x, x, y, padding_masks):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        outputs, rec, A = self.model(scaled_x, None, None, None, None) # [H]
        return outputs, rec, A, y.long().squeeze(-1)


    def _process_one_batch(self, batch_x, origin_x, batch_y, padding_masks):

        batch_x = batch_x.to(self.device, dtype=torch.float32)
        batch_y = batch_y.to(self.device, dtype=torch.float32)

        outputs, rec, A = self.model(batch_x, None, None, None, None)  # torch.Size([batch_size, output_length, num_nodes])

        return outputs, batch_y.long().squeeze(-1)


    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (scaled_x, x, y, padding_masks) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                scaled_x = scaled_x.to(self.device, dtype=torch.float32)
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)


                pred, rec, A,  true = self._process_train_batch(
                    scaled_x, x, y, padding_masks
                )
                if self.rec_X:
                    loss = self.loss_func(pred, true) + torch.mean((rec - scaled_x)**2)
                else:
                    loss = self.loss_func(pred, true) 

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(scaled_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.optimizer.step()

            return train_loss



        
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




    # def _test(self):
    #     self.plot()
    #     return super(NESForecast, self)._test()
        











if __name__ == "__main__":
    import fire
    fire.Fire(NESClassification)