
from dataclasses import dataclass, field
import sys
import os
import time
import torch
from tqdm import tqdm
from src.models.ode5 import NeuralSpectralForecaster
from src.experiments.forecast import ForecastExp
from torch_timeseries.nn.embedding import freq_map
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.dataset import *
from torch_timeseries.scaler import *
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
from src.datasets import *
import matplotlib.pyplot as plt
import math
from torch_timeseries.nn.embedding import freq_map



@dataclass
class WFParameters:
    fft_length: int = 24
    hop_length : int = 24
    hidden_dim: int = 512
    recon_step : float = 1.0 # larger value indicate finer granularity
    step_size : float = 0.1
    use_norm : bool= True

@dataclass
class ODESpecExp(ForecastExp, WFParameters):
    model_type: str = "ODESpecExp5"

    def _init_model(self):
        self.time_feature_size = freq_map[self.dataset.freq]
        # [B, W // 2 + 1, 1 + L // hop_length]
        self.freq_dim = (self.fft_length//2 + 1)*2
        # self.train_spec_num = math.ceil((self.windows+self.pred_len) / self.hop_length) - 1
        self.train_spec_num = math.ceil((self.pred_len) / self.hop_length)
        # self.infer_spec_num = math.ceil(  )
        # self.inp_spec_num = 1 + self.windows // self.hop_length
        # self.total_spec_num = math.ceil((self.windows+self.pred_len) / self.hop_length)
        
        self.freqs = torch.linspace(0, self.freq_dim-1, self.freq_dim).to(self.device) 
        self.model = NeuralSpectralForecaster(
            fft_length=self.fft_length,
            input_len=self.windows,
            freq_dim=self.freq_dim,
            hidden_dim=self.hidden_dim,
            step_size=self.step_size,
            time_dim=self.time_feature_size,
        )
        self.model = self.model.to(self.device)
        
        
        
        # self.ts_future = ( torch.arange(1, self.total_spec_num-1) * self.fft_length).float().to(self.device) 
        # self.ts_future = ( torch.arange(1, self.total_spec_num-1)*self.recon_step ).float().to(self.device) 
        
        # set to number of frequencies spectrum + 1, t[0] is the initial time
        self.ts = ( torch.arange(0, self.train_spec_num+1)*self.recon_step ).float().to(self.device)
        
        # self.pred_future = ( torch.arange(1, self.total_spec_num-1 - self.inp_spec_num) ).float().to(self.device) 
      

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
                    time_enc=3,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_index=True,
                )
            elif  self.dataset_type[0:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=True,
                    time_enc=3,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    time_index=True,
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
                time_enc=3,
                time_index=True,
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

      
        
    def compute_stft_spectrum(self, x):
        # x: [B, N, T]
        # return the real and image values of input [B, num of freq spectrum, freq spectrum]
        B, N, L = x.shape
        x = x.reshape(B*N, L)
        # 2D tensor input
        stft_out = torch.stft(
            x,
            n_fft=self.fft_length,
            hop_length=self.hop_length,  # non-overlapping chunks
            return_complex=False,
            onesided=True,
            center=False
        )  # [B*N, n_fft//2+1, 1 + (L - n_fft) // hop_length, 2]
        
        stft_out = stft_out.permute(0, 2, 1, 3).reshape(B, N, 1 + (L - self.fft_length) // self.hop_length, (self.fft_length//2+1)*2)  # [B, num of freq_spectrum, K//2*2+2]
        return stft_out


    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            with tqdm(total=len(dataloader.dataset)) as progress_bar:
                for (
                    batch_x,
                    batch_y,
                    batch_origin_x,
                    batch_origin_y,
                    batch_x_date_enc,
                    batch_y_date_enc,
                    x_index,
                    y_index
                ) in dataloader:
                    batch_size = batch_x.size(0)
                    truths = batch_y.to(self.device).float()
                    preds = self._val_batch(
                        batch_x, batch_x_date_enc, batch_y_date_enc, x_index, y_index
                    )
                    batch_origin_y = batch_origin_y.to(self.device)
                    if self.invtrans_loss:
                        preds = self.scaler.inverse_transform(preds)
                        truths = batch_origin_y
                    if self.pred_len == 1:
                        self.metrics.update(
                            preds.contiguous().reshape(batch_size, -1),
                            truths.contiguous().reshape(batch_size, -1),
                        )
                    else:
                        self.metrics.update(preds.contiguous(), truths.contiguous())

                    progress_bar.update(batch_x.shape[0])

            result = {
                name: float(metric.compute()) for name, metric in self.metrics.items()
            }
        return result


    def _val_batch(self, batch_x, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        
        # if self.use_norm:
        #     mean = batch_x.mean(1, keepdim=True)
        #     std = batch_x.std(1, keepdim=True)
        #     batch_x = (batch_x - mean)/std

        
        B, T, N = batch_x.shape
        # ts_x = x_index[:, ::self.fft_length][:, -1:]
        # ts_y = y_index[:, ::self.fft_length]
        # ts = torch.concat([ts_x, ts_y], dim=-1).float().to(self.device)
        # input_x = torch.concat([batch_x, batch_y], dim=1).permute(0, 2, 1).reshape(-1, T+O)
        
        # self.model(input_x)
        # xt_inpt = batch_x_date_enc[:, -self.fft_length:, :].reshape(B, -1) # fft_length*time_dim
        
        xt_inpt = batch_x_date_enc[:, -1:, :].reshape(B, -1) # fft_length*time_dim

        batch_x = batch_x.transpose(1, 2) # B, N, T
        stft_spec = self.compute_stft_spectrum(batch_x) #  # [B, N, num of freq_spectrum, self.fft_length//2*2+2]

        A_obs = stft_spec[:, :, -1, :]  # [B, N, K//2+1] use all spectrum to train
        A_pred = self.model(batch_x, xt_inpt, stft_spec, A_obs , self.ts.to(self.device))   # B*N, fs_O, F
        A_pred = A_pred.reshape(A_pred.shape[0], A_pred.shape[1], -1, 2)
        
        complex_tensor = torch.complex(A_pred[..., 0], A_pred[..., 1]) # B*N, 
        pred_values = torch.fft.irfft(complex_tensor, dim=-1, norm="backward")
        pred_values = pred_values.reshape(B, N, -1)[:, :, :self.pred_len].permute(0, 2, 1)
        

        return pred_values




    def _process_one_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        
        
        # if self.use_norm:
        #     mean = batch_x.mean(1, keepdim=True)
        #     std = batch_x.std(1, keepdim=True)
        #     batch_x = (batch_x - mean)/std
        #     # batch_y = (batch_y - mean)/std
        
        
        
        
        B, T, N = batch_x.shape
        B, O, N = batch_y.shape
        xt_inpt = batch_x_date_enc[:, -1:, :].reshape(B, -1) # fft_length*time_dim
        
        ts_x = x_index[:, ::self.fft_length][:, -1:]
        ts_y = y_index[:, ::self.fft_length]
        ts = torch.concat([ts_x, ts_y], dim=-1).float().to(self.device)
        # input_x = torch.concat([batch_x, batch_y], dim=1).permute(0, 2, 1).reshape(-1, T+O)
        # self.model(input_x)
        batch_x = batch_x.transpose(1, 2) # B, N, T
        stft_spec = self.compute_stft_spectrum(batch_x) #  # [B, N, num of freq_spectrum, self.fft_length//2*2+2]
        stft_spec_out = self.compute_stft_spectrum(batch_y.transpose(1, 2)) #  # [B, N, num of freq_spectrum, self.fft_length//2*2+2]

        A_obs = stft_spec[:, :, -1, :]  # [B, N, K//2+1] use all spectrum to train
        A_pred = self.model(batch_x, xt_inpt, stft_spec, A_obs , self.ts.to(self.device))   # B*N, fs_O, F
        
        
        

        # if self.use_norm:
        #     mean = batch_x.mean(1, keep_dim=True)
        #     std = batch_x.std(1, keep_dim=True)
        #     batch_x = (batch_x - mean)/std


        # no decoder input
        # label_len = 1
        # dec_inp_pred = torch.zeros(
        #     [batch_x.size(0), self.pred_len, self.dataset.num_features]
        # ).to(self.device)
        # dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        # )
        return torch.concat([stft_spec[:, :, -1:, :], stft_spec_out], dim=2).reshape(A_pred.shape[0], A_pred.shape[1], -1), A_pred.reshape(A_pred.shape[0], A_pred.shape[1], -1)




    def plot(self):
        
        full_dataset = MultiStepTimeFeatureSet(
            self.dataset,
            scaler=self.scaler,
            time_enc=3,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
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
        outs = self._val_batch(all_batch_x, all_x_date_enc, all_y_date_enc, all_x_index, all_y_index) # B, T, N
        
        fig, axes = plt.subplots(all_batch_x.shape[2])
        for i in range(all_batch_x.shape[2]):
            out = outs[:, :, i]
            pred_all = out.reshape(-1).detach().cpu().numpy()
            y_all = all_batch_y[:, :, i].reshape(-1).detach().cpu().numpy()
            axes[i].plot(pred_all, label='pred')
            axes[i].plot(y_all, label='y')
            
        plt.legend()
        plt.savefig(os.path.join(self.run_save_dir, 'full.png'))

        # instance
        last_n = 9
        fig, axes = plt.subplots(all_batch_x.shape[2])
        for i in range(all_batch_x.shape[2]):
            out = outs[-last_n:, :, i]
            pred_all = out.reshape(-1).detach().cpu().numpy()
            y_all = all_batch_y[-last_n:, :, i].reshape(-1).detach().cpu().numpy()
            axes[i].plot(pred_all, label='pred')
            axes[i].plot(y_all, label='y')
            
        plt.legend()
        plt.savefig(os.path.join(self.run_save_dir, 'instance.png'))



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
                y_index,
            ) in enumerate(self.train_loader):
                start = time.time()
                origin_y = origin_y.to(self.device)
                self.model_optim.zero_grad()
                pred, true = self._process_one_batch(
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




    def _test(self):
        super(ODESpecExp, self)._test()
        self.plot()




if __name__ == "__main__":
    import fire
    fire.Fire(ODESpecExp)