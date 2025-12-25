
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

class LinearChirp1(TimeSeriesDataset):
    name: str = 'LinearChirp1'
    num_features:int = 1
    sample_rate:int = 1
    length : int= 2000
    freq: str = 't'
    
    def download(self): 
        pass
    

    def _load(self):
        # n = 400
        # Generating date series

        fs = 1000          # 采样频率 (Hz)
        T = 2.0            # 信号总时长 (秒)
        t = np.linspace(0, T, int(fs * T), endpoint=False)
        f0 = 10

        f0 = 10            # 起始频率 (Hz)
        f1 = 20           # 结束频率 (Hz)
        method = 'linear'  # 调频方式：'linear', 'quadratic', 'logarithmic', 'hyperbolic'

        x = chirp(t, f0=f0, f1=f1, t1=T, method=method, phi=0)
        dates = pd.date_range(start='2022-01-01', periods= len(t), freq='t')

        # Creating DataFrame with specified column names
        self.df = pd.DataFrame(x, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates =  pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values
        return self.data

