
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
from src.experiments.TimeES import TimeESForecast
from src.models.NES import NeuralEvolutionarySpectra
from src.utils.pesudo_spectrum import get_initial_spectrum_benowitz, get_initial_spectrum_benowitz_targetM
from src.utils.pesudo_amplitude import get_initial_amplitude_right_onesided_mv, get_initial_amplitude_stft_torch
from src.utils.evolutionary_spectra import select_frequencies_by_energy_ratio, select_frequencies_by_energy_ratio_batch
from src.models.TimeES import TimeES


@dataclass
class TimeESReconstruct(TimeESForecast):
    model_type: str = "TimeESReconstruct"
    horizon:int  = 1
    pred_len:int  = 1
    method:str = 'prob_rec' # prob_rec or determin_rec

    def _init_model(self):
        # A_init, omegas  = get_initial_amplitude_right_stft_torch(
        #     torch.tensor(self.dataloader.recon_set.scaled_data.squeeze()), 
        #     n_fft=self.M, 
        #     hop_length=self.hop_length,
        # )
        # self.model = NeuralEvolutionarySpectra(
        #     self.M,
        #     self.device,
        #     torch.tensor(A_init).cfloat(),
        #     omegas
        # )
        # self.model = self.model.to(self.device)

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
        # selected_freqs=[torch.tensor([selected_freqs[0][1]])]
        print("selected_freqs:", selected_freqs)

        A_init, omegas  = get_initial_amplitude_right_onesided_mv(
            torch.tensor(scaled_data).transpose(0, 1), 
            n_fft=self.M, 
            return_full_omegas=True,
        )

        A0_torch = torch.tensor(A_init).cfloat()
        # energy_per_frame = torch.mean(torch.abs(A0_torch), dim=0)  # [B, N, M]
        # _, topk_indices = torch.topk(energy_per_frame, k=self.topk, dim=0, largest=True)  # [B, N, K]
        self.selected_freqs = selected_freqs
        if self.method == 'prob_rec':
            task = 'prob_forecast'
        else:
            task = 'forecsat'
        self.model = TimeES(
            self.windows,
            self.dataset.num_features,
            self.windows,
            self.device,
            omegas=omegas,
            M=self.M,
            selected_freqs=selected_freqs,
            fast_build=self.fast_build,
            t_emb=self.t_emb,
            tc_emb=self.tc_emb,
            use_norm=self.use_norm,
            backbone=self.backbone,
            task=task,
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
        # results = results.permute(0, 2, 1) # 
        # results: B N T
        # A: B N T M

        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        # return results, y.squeeze(2), A
        return results[:, -self.windows:, :], batch_y, A[:, :, :, :]


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
        # results = results.permute(0, 2, 1)
        # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
        if return_A:
            return results[:, -self.windows:, :], batch_y, A[:, :, :, :]
        return results[:, -self.windows:, :], batch_y


    # def _process_one_batch(self, index, batch_x):
    #     # inputs:
    #     # batch_x: (B, T, N)
    #     # batch_y: (B, O, N)
    #     # ouputs:
    #     # - pred: (B, N)/(B, O, N)
    #     # - label: (B, N)/(B, O, N)

    #     batch_x = batch_x.to(self.device, dtype=torch.float32)
    #     index = index.to(self.device, dtype=torch.float32)

    #     rec_X, A_all = self.model(batch_x, index, index) # [H]

        
    #     # out_true = torch.concat([batch_x, batch_y], dim=1).reshape(-1)
    #     return rec_X, A_all

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
                batch_x = batch_x.to(self.device).float()
                self.model_optim.zero_grad()
                rec_X, true, A = self._process_train_batch(
                    batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                )
                if self.method == 'prob_rec':
                    rec_X = rec_X.mean(-1)


                loss = self.loss_func(rec_X, batch_x)
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
                    batch_x = batch_x.to(self.device).float()
                    self.model_optim.zero_grad()
                    rec_X, truths = self._process_one_batch(
                        batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                    )
                    if self.method == 'prob_rec':
                        rec_X = rec_X.mean(-1)

                    self.metrics.update(rec_X.contiguous(), batch_x.contiguous())

                    progress_bar.update(batch_x.shape[0])

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result



    def plot(self):
        if self.method == 'prob_rec':
            self.prob_plot()
        else:
            self.determin_plot()
            
    def prob_plot(self):
        full_dataset = MultiStepTimeFeatureSet(
            self.dataset,
            scaler=self.scaler,
            time_enc=self.timeenc,
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
        

        for i in range(0, len(full_dataset), self.windows) :
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index = full_dataset[i]
            all_batch_x.append(torch.tensor(batch_x))
            all_batch_y.append(torch.tensor(batch_y))
            all_batch_x_date_enc.append(torch.tensor(batch_x_date_enc))
            all_batch_y_date_enc.append(torch.tensor(batch_y_date_enc))
            
            all_x_index.append(torch.tensor(x_index))
            all_y_index.append(torch.tensor(y_index))
        all_batch_x = torch.stack(all_batch_x, dim=0).to(self.device).float()[-32:]
        all_batch_y = torch.stack(all_batch_y, dim=0).to(self.device).float()[-32:]
        all_x_date_enc = torch.stack(all_batch_x_date_enc, dim=0).to(self.device).float()[-32:]
        all_y_date_enc = torch.stack(all_batch_y_date_enc, dim=0).to(self.device).float()[-32:]
        all_x_index = torch.stack(all_x_index, dim=0).to(self.device).float()[-32:]
        all_y_index = torch.stack(all_y_index, dim=0).to(self.device).float()[-32:]
        # outs = self._val_batch(all_batch_x, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index) # B, T, N
        
        samples, trues, A = self._process_one_batch(
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
            


            # plot prob
            outs = samples.mean(-1)
            lower = torch.quantile(samples, 0.04, dim=-1)   # 10th percentile → lower bound of 80% PI
            upper = torch.quantile(samples, 0.96, dim=-1)   # 90th percentile → upper bound of 80% PI

            fig, axes = plt.subplots(1)
            rec_X = outs.squeeze().reshape(-1).detach().cpu().numpy()
            X = all_batch_x.squeeze().reshape(-1).detach().cpu().numpy()
            lower = lower.squeeze().reshape(-1).detach().cpu().numpy()
            upper = upper.squeeze().reshape(-1).detach().cpu().numpy()
        
            # axes.plot(pred_all, label='pred')
            # axes.plot(X, label='X')
            # axes.plot(rec_X, label='rec_X')
            # axes.fill_between(range(len(rec_X)), lower, upper, color='gray', alpha=0.2, label='Standard Error')

            # plt.legend()
            # plt.savefig(os.path.join(self.run_save_dir, 'prob_global.png'))
            print(samples.shape)
            fig, axes = plt.subplots(1)
            axes.plot(rec_X[-2000:], label='Mean')
            axes.fill_between(range(len(rec_X[-2000:])), lower[-2000:], upper[-2000:], color='gray', alpha=0.2, label='Standard Error')
            axes.plot(X[-2000:], label='y')
            plt.legend()
            plt.savefig(os.path.join(self.run_save_dir, 'prob_instance.png'))

            plt.close()



            # plot A_X
            A_X = A[:, :self.windows, :].reshape(-1, A.shape[-1]).detach().cpu()  # shape: (A*B, C)
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







    def determin_plot(self):
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

            # # for resumable reproducibility
            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result[self.loss_func_type], model=self.model)

            self._save_run_check_point(seed)

            if self._use_wandb():
                wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
                wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
                wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

            self.scheduler.step()

        self._load_best_model()
        best_test_result = self._test()
        if self._use_wandb():
            for k, v in best_test_result.items(): wandb.run.summary[f"best_test_{k}"] = v 
        
        if self._use_wandb():  wandb.finish()
        return best_test_result

    # def run(self, seed=42) -> Dict[str, float]:

    #     if self._use_wandb() and not self._init_wandb(self.project, seed): return {}
        
    #     self._setup_run(seed)
    #     if self._check_run_exist(seed):
    #         self._resume_run(seed)

    #     self._run_print(f"run : {self.current_run} in seed: {seed}")

    #     parameter_tables, model_parameters_num = count_parameters(self.model)
    #     self._run_print(f"parameter_tables: {parameter_tables}")
    #     self._run_print(f"model parameters: {model_parameters_num}")

    #     if self._use_wandb():
    #         wandb.run.summary["parameters"] = model_parameters_num

    #     # for resumable reproducibility
    #     while self.current_epoch < self.epochs:
    #         epoch_start_time = time.time()
    #         if self.early_stopper.early_stop is True:
    #             self._run_print(
    #                 f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
    #             )
    #             break

    #         # for resumable reproducibility
    #         reproducible(seed + self.current_epoch)
    #         train_losses = self._train()
    #         self.plot()
    #         self._run_print(
    #             "Epoch: {} cost time: {}s".format(
    #                 self.current_epoch + 1, time.time() - epoch_start_time
    #             )
    #         )
    #         self._run_print(f"Traininng loss : {np.mean(train_losses)}")

    #         # val_result = self._val()
    #         # test_result = self._test()

    #         self.current_epoch = self.current_epoch + 1
    #         # self.early_stopper(val_result[self.loss_func_type], model=self.model)

    #         self._save_run_check_point(seed)

    #         if self._use_wandb():
    #             wandb.log({'training_loss' : np.mean(train_losses)}, step=self.current_epoch)
    #             # wandb.log( {f"val_{k}": v for k, v in val_result.items()}, step=self.current_epoch)
    #             # wandb.log( {f"test_{k}": v for k, v in test_result.items()}, step=self.current_epoch)

    #         # self.scheduler.step()
    #     train_losses = self._train()
    #     best_test_result = np.mean(train_losses) #self._test()
    #     return best_test_result

    def _test(self):
        self.plot()
        return super(TimeESReconstruct, self)._test()


if __name__ == "__main__":
    import fire
    fire.Fire(TimeESReconstruct)