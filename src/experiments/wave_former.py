
from dataclasses import dataclass, field
import sys
import os

import torch
from src.models.wave_former1 import WaveFormer
from src.experiments.forecast import ForecastExp
from torch_timeseries.nn.embedding import freq_map
from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast
import matplotlib.pyplot as plt

@dataclass
class WFParameters:
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 512  # out of memoery with d_ff = 2048
    dropout: float = 0.0
    n_heads : int = 8
    wave_len : int = 16
    


@dataclass
class WaveFormerExp(ForecastExp, WFParameters):
    model_type: str = "WaveFormer1"

    def _init_model(self):
        time_feature_size = freq_map[self.dataset.freq]
        self.model = WaveFormer(
            seq_len=self.windows,
            pred_len=self.pred_len,
            t_feat_size=time_feature_size,
            enc_in=self.dataset.num_features,
            n_heads=self.n_heads,
            dropout=self.dropout,
            e_layers=self.e_layers,
            d_model=self.d_model,
            d_ff=self.d_ff,
            wave_len=self.wave_len,
            num_class=0
            )
        self.model = self.model.to(self.device)

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

        # no decoder input
        # label_len = 1
        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.dataset.num_features]
        ).to(self.device)
        # dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)

        # dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
        # dec_inp_date_enc = torch.cat(
        #     [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1
        # )
        outputs = self.model(batch_x, batch_x_date_enc,
                             None, batch_y_date_enc)
        return outputs, batch_y




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
        outs, _ = self._process_one_batch(all_batch_x, all_batch_y, None, None, all_x_date_enc, all_y_date_enc) # B, T, N
        
        fig, axes = plt.subplots(all_batch_x.shape[2])
        for i in range(all_batch_x.shape[2]):
            out = outs[:, :, i]
            pred_all = out.reshape(-1).detach().cpu().numpy()
            y_all = all_batch_y[:, :, i].reshape(-1).detach().cpu().numpy()
            axes[i].plot(pred_all, label='pred')
            axes[i].plot(y_all, label='y')
            

        
        # n = 0 
        # fig = plt.figure()
        # outs = outs[:, :, n]
        # pred_all = outs.reshape(-1).detach().cpu().numpy()
        # y_all = all_batch_y[:, :, n].reshape(-1).detach().cpu().numpy()
        # plt.plot(pred_all, label='pred')
        # plt.plot(y_all, label='y')
        plt.legend()
        plt.savefig(os.path.join(self.run_save_dir, 'full.png'))

    def _test(self):
        super(WaveFormerExp, self)._test()
        self.plot()




if __name__ == "__main__":
    import fire
    fire.Fire(WaveFormerExp)