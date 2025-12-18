
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
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast, ReconstructSet
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
from scipy.signal import spectrogram

import torch
from src.experiments.reconstruct import ReconstructExp
from src.models.nesr4 import NeuralEvolutionarySpectra
from src.utils.pesudo_spectrum import get_initial_spectrum_benowitz


@dataclass
class NESParameters:
    hidden_dim : int = 128
    M : int = 200

@dataclass
class NESReconstruct(ReconstructExp, NESParameters):
    model_type: str = "NESR4"

    def _init_model(self):
        S_init, omega, R_est = get_initial_spectrum_benowitz(
            self.dataset.data.squeeze(), 
            max_lag=self.M,       # consider lags
            smooth_sigma=3.0  # smooth ACF in time
        )
        mag = torch.tensor(np.sqrt(S_init)).float() 
        phase = torch.rand_like(mag) * (2 * np.pi) - np.pi
        self.model = NeuralEvolutionarySpectra(
            self.dataset.length, 
            self.M,
            self.device,
            torch.polar(mag, phase),
            omega
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, index, batch_x):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)

        batch_x = batch_x.to(self.device, dtype=torch.float32)
        index = index.to(self.device, dtype=torch.float32)
        rec_X, A_all = self.model(index) # [H]
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        return rec_X, A_all

    def _train(self):
        with torch.enable_grad(), tqdm(total=len(self.dataloader.dataset)) as progress_bar:
            self.model.train()
            train_loss = []
            for i, (
                index,
                X,
            ) in enumerate(self.dataloader.dataloader):
                self.model_optim.zero_grad()
                X = X.to(self.device).float()
                index = index.to(self.device).int()
                rec_X, _ =  self._process_one_batch(index, X)
                loss = self.loss_func(rec_X.squeeze(), X.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                progress_bar.update(X.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
        
            return train_loss



    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for (
                    batch_index,
                    batch_x
                ) in dataloader:
                    batch_size = batch_x.size(0)
                    rec_X, X = self._process_one_batch(
                        batch_index, batch_x
                    )
                    self.metrics.update(rec_X.contiguous(), X.contiguous())
                    progress_bar.update(batch_x.shape[0])

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result
    


    def plot(self):
        full_dataset = ReconstructSet(self.dataset, self.scaler)
        all_x = []
        all_index = []
        

        for i in range(0, len(full_dataset)) :
            index, x = full_dataset[i]
            all_x.append(torch.tensor(x))
            all_index.append(torch.tensor(index))
            
        all_x = torch.stack(all_x, dim=0).to(self.device).float() 
        all_index = torch.stack(all_index, dim=0).to(self.device).float()
        rec_X, A = self._process_one_batch(
            all_index, all_x
        )


        # fig, axes = plt.subplots(2)

        X = all_x.squeeze().reshape(-1).detach().cpu().numpy()
        rec_X = rec_X.squeeze().reshape(-1).detach().cpu().numpy()
    
        # axes[0].plot(rec_X, label='rec_X')
        # axes[0].plot(X, label='X')

        # # plot A
        # magnitude = np.abs(A)

        # time = all_index
        # freq = np.arange(0, self.M)
        # time, freq = np.meshgrid(time, freq)
        # surf = axes[1].plot_surface(time, freq, magnitude.T, cmap='viridis')
        
        # fig.colorbar(surf)

        # # 设置坐标轴标签
        # axes[1].set_xlabel('Time')
        # axes[1].set_ylabel('Frequency')
        # axes[1].set_zlabel('Magnitude')

        # plt.title('3D Magnitude Plot over Time and Frequency')
        # plt.legend()
        # plt.savefig(os.path.join(self.run_save_dir, 'global.png'))

        
        A = A.detach().cpu().numpy()  # shape: [T, M]

        magnitude = np.abs(A)  # [T, M]

        # time_steps = np.arange(A.shape[0])      # [T]
        # freq_bins = np.arange(A.shape[1])       # [M]
        # Time, Freq = np.meshgrid(time_steps, freq_bins, indexing='ij')  # 注意 indexing='ij'

        # fig = plt.figure(figsize=(12, 8))
        # ax1 = fig.add_subplot(211)
        # ax1.plot(rec_X, label='rec_X')
        # ax1.plot(X, label='X')
        # ax1.legend()
        # ax1.set_title('Reconstruction vs Ground Truth')

        # # 子图 2: 3D 频谱图
        # ax2 = fig.add_subplot(212, projection='3d')  # ← 关键：projection='3d'
        # surf = ax2.plot_surface(Time, Freq, magnitude, cmap='viridis', linewidth=0, antialiased=False)
        # fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

        # ax2.set_xlabel('Time')
        # ax2.set_ylabel('Frequency')
        # ax2.set_zlabel('Magnitude')
        # ax2.set_title('3D Magnitude Spectrogram')

        # # 保存
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.run_save_dir, 'global.png'), dpi=150)



        time_steps = np.arange(A.shape[0])      # [T]
        freq_bins = np.arange(A.shape[1])       # [M]
        Time, Freq = np.meshgrid(time_steps, freq_bins, indexing='ij')

        # 创建 3 行 1 列的子图
        fig = plt.figure(figsize=(12, 12))

        # 子图 1: 重建 vs 真实信号
        ax1 = fig.add_subplot(311)
        ax1.plot(rec_X, label='rec_X')
        ax1.plot(X, label='X')
        ax1.legend()
        ax1.set_title('Reconstruction vs Ground Truth')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Amplitude')

        # 子图 2: 3D 频谱图（你的 A 的 magnitude）
        ax2 = fig.add_subplot(312, projection='3d')
        surf = ax2.plot_surface(Time, Freq, magnitude, cmap='viridis', linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency Bin')
        ax2.set_zlabel('Magnitude')
        ax2.set_title('3D Magnitude Spectrogram (Learned A)')

        ax3 = fig.add_subplot(313)
        # 注意：magnitude 是 [T, M]，imshow 默认 (行=时间, 列=频率)
        # 但通常 spectrogram 是 频率在 y 轴，时间在 x 轴 → 所以 transpose 或调整 origin
        im = ax3.imshow(
            magnitude.T,               # 转置：shape [M, T] → freq on y, time on x
            aspect='auto',
            origin='lower',            # 低频在底部，高频在顶部
            cmap='viridis',
            interpolation='nearest'
        )
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Frequency Bin')
        ax3.set_title('Learned Magnitude Spectrogram (|A|)')
        fig.colorbar(im, ax=ax3, label='Magnitude')


        plt.tight_layout()
        plt.savefig(os.path.join(self.run_save_dir, 'global_with_spectrogram.png'))


    def run(self, seed=42) -> Dict[str, float]:

        if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        if self._use_wandb():
            wandb.run.summary["parameters"] = model_parameters_num

        # for resumable reproducibility
        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self.plot()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            # val_result = self._val()
            # test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            # self.early_stopper(val_result[self.loss_func_type], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                # wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                # wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            # self.scheduler.step()
        train_losses = self._train()
        best_test_result = np.mean(train_losses) #self._test()
        return best_test_result



if __name__ == "__main__":
    import fire
    fire.Fire(NESReconstruct)