
from dataclasses import dataclass, field
import sys
import os

import torch
from tqdm import tqdm
from src.models.ode3 import NeuralSpectralForecaster
from src.experiments.forecast import ForecastExp
from torch_timeseries.nn.embedding import freq_map
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast
import matplotlib.pyplot as plt

@dataclass
class WFParameters:
    fft_length: int = 24
    hop_length : int = 24
    hidden_dim: int = 512
    recon_step : int = 1

@dataclass
class ODESpecExp(ForecastExp, WFParameters):
    model_type: str = "ODESpecExp"

    def _init_model(self):
        time_feature_size = freq_map[self.dataset.freq]
        # [B, W // 2 + 1, 1 + L // hop_length]
        self.freq_dim = (self.fft_length//2 + 1)*2
        self.inp_spec_num = 1 + self.windows // self.hop_length
        self.total_spec_num = 1 + (self.windows+self.pred_len) // self.hop_length
        
        self.freqs = torch.linspace(0, self.freq_dim-1, self.freq_dim).to(self.device) 
        self.model = NeuralSpectralForecaster(
            input_len=self.windows,
            freq_dim=self.freq_dim,
            hidden_dim=self.hidden_dim,
            freqs=self.freqs
        )
        self.model = self.model.to(self.device)
        
        
        # self.ts_future = ( torch.arange(1, self.total_spec_num-1) * self.fft_length).float().to(self.device) 
        self.ts_future = ( torch.arange(1, self.total_spec_num-1)*self.recon_step ).float().to(self.device) 
        # self.pred_future = ( torch.arange(1, self.total_spec_num-1 - self.inp_spec_num) ).float().to(self.device) 
        

    def compute_stft_spectrum(self, x):
        # x: [B, T]
        B, L = x.shape
        stft_out = torch.stft(
            x,
            n_fft=self.fft_length,
            hop_length=self.hop_length,  # non-overlapping chunks
            return_complex=False,
            onesided=True,
            center=False
        )  # [B, n_fft//2+1, 1 + (L - n_fft) // hop_length, 2]
        
        return stft_out.permute(0, 2, 1, 3).reshape(B, 1 + (L - self.fft_length) // self.hop_length, (self.fft_length//2+1)*2)  # [B, num of freq_spectrum, K//2*2+2]


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
                ) in dataloader:
                    batch_size = batch_x.size(0)
                    truths = batch_y.to(self.device).float()
                    preds = self._val_batch(
                        batch_x, batch_x_date_enc, batch_y_date_enc
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


    def _val_batch(self, batch_x, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x = batch_x.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        
        
        B, T, N = batch_x.shape
        
        input_x = batch_x.permute(0, 2, 1).reshape(-1, T)
        stft_spec = self.compute_stft_spectrum(input_x) #  # [B, num of freq_spectrum, self.fft_length//2*2+2]
        A_obs = stft_spec[:, -1, :]  # [B*N, 1, K//2+1] 
        A_pred = self.model(A_obs, self.ts_future.to(self.device))   # B*N, len(self.ts_future), self.fft_length//2*2+2
        A_pred = A_pred.reshape(A_pred.shape[0], A_pred.shape[1], -1, 2)
        complex_tensor = torch.complex(A_pred[..., 0], A_pred[..., 1]) # B*N, 
        pred_values = torch.fft.irfft(complex_tensor, dim=-1, norm="backward")
        pred_values = pred_values.reshape(B, N, -1)[:, :, :self.pred_len].permute(0, 2, 1)
        
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
        return pred_values




    def _process_one_batch(self, batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc):
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
        
        
        B, T, N = batch_x.shape
        B, O, N = batch_y.shape
        
        input_x = torch.concat([batch_x, batch_y], dim=1).permute(0, 2, 1).reshape(-1, T+O)
        
        # self.model(input_x)
        
        stft_spec = self.compute_stft_spectrum(input_x) #  # [B, num of freq_spectrum, self.fft_length//2*2+2]

        # stft_out = torch.stft(
        #     input_x,
        #     n_fft=self.fft_length,
        #     hop_length=self.hop_length,  # non-overlapping chunks
        #     return_complex=False,
        #     onesided=True,
        #     center=False
        # )  # [B, n_fft//2+1, 1 + (L - n_fft) // hop_length, 2]
        # stft_spec = stft_out.permute(0, 2, 1, 3).reshape(B, 1 + (T - self.fft_length) // self.hop_length, (self.fft_length//2+1)*2)  # [B, num of freq_spectrum, K//2*2+2]
        A_obs = stft_spec[:, 0:1, :]  # [B*N, 1, K//2+1] 
        A_pred = self.model(A_obs, self.ts_future.to(self.device))   # 
        
        
        

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
        return stft_spec[:, 1:, :], A_pred.reshape(A_pred.shape[0], A_pred.shape[1], -1)




    def plot(self):
        
        full_dataset = MultiStepTimeFeatureSet(
            self.dataset,
            scaler=self.scaler,
            time_enc=0,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            freq=self.dataset.freq,
            single_variate=False,
            scaler_fit=False,
        )

        all_batch_x = []
        all_batch_x_date_enc = []
        all_batch_y_date_enc = []
        all_batch_y = []

        for i in range(0, len(full_dataset), self.pred_len) :
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc = full_dataset[i]
            all_batch_x.append(torch.tensor(batch_x))
            all_batch_y.append(torch.tensor(batch_y))
            all_batch_x_date_enc.append(torch.tensor(batch_x_date_enc))
            all_batch_y_date_enc.append(torch.tensor(batch_y_date_enc))
            
        all_batch_x = torch.stack(all_batch_x, dim=0).to(self.device).float() 
        all_batch_y = torch.stack(all_batch_y, dim=0).to(self.device).float() 
        all_x_date_enc = torch.stack(all_batch_x_date_enc, dim=0).to(self.device).float() 
        all_y_date_enc = torch.stack(all_batch_y_date_enc, dim=0).to(self.device).float()
        outs = self._val_batch(all_batch_x, all_x_date_enc, all_y_date_enc) # B, T, N
        
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
        fig, axes = plt.subplots(all_batch_x.shape[2])
        for i in range(all_batch_x.shape[2]):
            out = outs[-3:, :, i]
            pred_all = out.reshape(-1).detach().cpu().numpy()
            y_all = all_batch_y[-3:, :, i].reshape(-1).detach().cpu().numpy()
            axes[i].plot(pred_all, label='pred')
            axes[i].plot(y_all, label='y')
            
        plt.legend()
        plt.savefig(os.path.join(self.run_save_dir, 'instance.png'))


    def _test(self):
        # super(ODESpecExp, self)._test()
        self.plot()




if __name__ == "__main__":
    import fire
    fire.Fire(ODESpecExp)