


from dataclasses import dataclass
import sys

import torch
from src.models.FITS import FITS
from src.experiments.forecast import ForecastExp



@dataclass
class FITSExperiment(ForecastExp):
    model_type: str = "FITS"

    individual : bool = False
    cut_freq :int = 25

    def _init_model(self):
        self.model = FITS(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=self.dataset.num_features,
            individual=self.individual,
            cut_freq = self.cut_freq,
        )
        self.model = self.model.to(self.device)

    def _process_one_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
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
        
        pred, low = self.model(batch_x) # (B, O, N)
        return pred[:, -self.pred_len:, :], batch_y # (B, O, N), (B, O, N)

    def _test(self):
        super(FITSExperiment, self)._test()
        self.plot()




if __name__ == "__main__":
    import fire
    fire.Fire(FITSExperiment)



