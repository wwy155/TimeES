

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Callable, Generic, NewType, Optional, Sequence, TypeVar, Union
from torch import Tensor
import torch.utils.data
import os
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from abc import ABC, abstractmethod



from enum import Enum, unique

from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant

class SimFreq1F(TimeSeriesDataset):
    name: str = 'SimFreq3F'
    num_features:int = 1
    sample_rate:int = 1
    length : int= 10000
    freq: str = 't'

    def download(self): 
        pass

    def _load(self):
        # Generating date series
        dates = pd.date_range(start='2022-01-01', periods=self.length, freq='t')
        
        # Creating a data matrix
        data = np.zeros((len(dates), self.num_features))
        
        # Define the frequency and magnitude ranges for linear growth
        start_freq, end_freq = 6, 24  # Frequency range
        start_mag, end_mag = 1, 3     # Magnitude range
        
        # Calculate frequencies and magnitudes that linearly increase over time
        freqs_increase = np.linspace(start_freq, end_freq, num=self.length)
        mags_increase = np.linspace(start_mag, end_mag, num=self.length)
        t = np.arange(0, len(dates))
        x = np.sin(2 * np.pi * freqs_increase * t)
            
        data[:, 0] = x
        self.df = pd.DataFrame(data, columns=[ f"data{i}" for i in range(self.num_features)])
        self.df['date'] = dates
        self.dates =  pd.DataFrame({'date': dates})
        self.data = self.df.drop('date', axis=1).values       
        return self.data



