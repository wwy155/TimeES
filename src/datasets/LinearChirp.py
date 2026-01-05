
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod
from scipy.signal import chirp, spectrogram



from enum import Enum, unique

from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant

class LinearChirp(TimeSeriesDataset):
    name: str = 'LinearChirp'
    num_features:int = 1
    sample_rate:int = 1
    length : int= 1000
    freq: str = 't'
    
    def download(self): 
        pass
        
    def _load(self):
        dt = 0.01 
        N = self.length
        t = np.arange(N)*dt 
        
        f0 = 6  
        f1 = 1   

        x = chirp(t, f0=f0, f1=f1, t1=dt*N, method='hyperbolic')
        x = x.reshape(-1, 1)  # (N, 1)

        dates = pd.date_range(start='2022-01-01', periods=N, freq='t')
        self.df = pd.DataFrame(x, columns=[f"data{i}" for i in range(self.num_features)])
        self.df['date'] =  dates
        self.dates = pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values
        return self.data
