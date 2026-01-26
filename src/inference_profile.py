import os
import sys
import torch
import time
import numpy as np
# sys.path.insert(0,os.path.abspath('/data/yww/notebook/3902_nes'))
# sys.path.insert(0,os.path.abspath('/notebooks/pytorchtimseries'))
from tqdm.notebook import tqdm
from torch_timeseries.dataset import *
from torch_timeseries.experiments import *
from src.datasets import *
from torch_timeseries.utils.model_stats import count_parameters
import pandas as pd

# from src.experiments.iTransformer import iTransformerExp
# exp = DLinearForecast(data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
# exp = DLinearForecast(dataset_type="ExchangeRate", data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/pytorchtimseries/results', device='cuda:0')
# exp = PatchTSTForecast(dataset_type="ExchangeRate", data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/pytorchtimseries/results', device='cuda:0')

# def nsdiff_profile(dataset_type, windows, pred_len ):
#     model_type = "NsDiff"
#     # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/pytorchtimseries/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
#     # exp = PANForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, start_d_model=start_d_model, end_d_model=end_d_model, patch_len=patch_len, stride=stride, data_path='/notebooks/4901_revisit_cdtran/data', save_dir='/notebooks/4901_revisit_cdtran/results', device='cuda:0')
#     exp = NsDiffForecast(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/notebooks/3108Dif/data', save_dir='/notebooks/3108Dif/results', device='cuda:0')
#     exp._setup_run(1000)
#     memories = []
    
#     from torch_timeseries.utils.model_stats import count_parameters
#     _, nparam = count_parameters(exp.model)
#     torch.cuda.empty_cache()
#     steps = 20
#     i = 0
#     self = exp
#     self.model.eval()
#     t = 0
#     for i, (
#         batch_x,
#         batch_y,
#         origin_x,
#         origin_y,
#         batch_x_date_enc,
#         batch_y_date_enc,
#     ) in enumerate(self.train_loader):
#         if i >= steps:
#             break
#         i+= 1
#         # print(i)
#         origin_y = origin_y.to(self.device).float()
#         batch_x = batch_x.to(self.device).float()
#         batch_y = batch_y.to(self.device).float()
#         batch_x_date_enc = batch_x_date_enc.to(self.device).float()
#         batch_y_date_enc = batch_y_date_enc.to(self.device).float()
#         torch.cuda.reset_peak_memory_stats()
#         start_memory = torch.cuda.memory_allocated()
#         pred, true = self._process_val_batch(
#             batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
#         )
#         t += time.time() - start
#         end_memory = torch.cuda.max_memory_allocated()
#         memories.append(end_memory - start_memory)
#         print(end_memory - start_memory)
#         self.model_optim.step()

#     memories = np.array(memories)

#     print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
#     print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
#     return "PAN", dataset_type, max(memories), min(memories), t/i*1000, nparam

def baseline_profile(model_type, dataset_type, windows, pred_len, device='cuda:2'):
    exp = eval(f"{model_type}Forecast", globals())(num_samples=1, batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/data/yww/notebook/pytorchtimseries', device=device)
    exp._setup_run(1000)
    memories = []
    
    from torch_timeseries.utils.model_stats import count_parameters
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.eval()
    torch.cuda.set_device(device)

    t = 0
    ts = []
    for i, (
        batch_x,
        batch_y,
        origin_x,
        origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ) in enumerate(self.train_loader):
        if i >= steps:
            break
        i+= 1
        
        origin_y = origin_y.to(self.device).float()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()
        if model_type in set([ "TimeDiff", "DiffusionTS", "TMDM", "NsDiff"]):
                loss = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
        elif model_type in set(['TimeGrad', 'CSDI']):
            pred, true = self._process_val_batch(
                batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
            )
        ts.append(time.time() - start)
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)

    memories = np.array(memories)
    
    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return max(memories), min(memories), t/i*1000, np.std(ts), nparam


def profile_nes(model_type, dataset_type, windows, pred_len, device='cuda:2'):
    import matplotlib.pyplot as plt
    import numpy as np

    import os
    import sys
    import torch
    import time
    import seaborn as sns
    sys.path.insert(0,os.path.abspath('/data/yww/notebook/pytorchtimseries'))
    sys.path.insert(0,os.path.abspath('/data/yww/notebook/3902_nodespec'))

    from torch_timeseries.dataloader.wrapper import MultiStepTimeFeatureSet, MultivariateFast
    from torch.utils.data.dataloader import DataLoader

    from src.experiments.NESProbForecast import NESProbForecast

    from torch_timeseries.utils.model_stats import count_parameters
    exp = NESProbForecast(batch_size=1, energy_ratio=0.9, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/data/yww/notebook/pytorchtimseries', device=device)
    exp._setup_run(1000)

    memories = []
    _, nparam = count_parameters(exp.model)
    torch.cuda.empty_cache()
    steps = 20
    i = 0
    self = exp
    self.model.eval()
    torch.cuda.set_device(device)

    t = 0
    ts = []
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
        if i >= steps:
            break
        i+= 1
        
        origin_y = origin_y.to(self.device).float()
        batch_x = batch_x.to(self.device).float()
        batch_y = batch_y.to(self.device).float()
        batch_x_date_enc = batch_x_date_enc.to(self.device).float()
        batch_y_date_enc = batch_y_date_enc.to(self.device).float()
        x_index = x_index.to(self.device)
        y_index = y_index.to(self.device)
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start = time.time()

        self._process_val_batch(
            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, x_index, y_index
        )

        ts.append(time.time() - start)
        t += time.time() - start
        end_memory = torch.cuda.max_memory_allocated()
        memories.append(end_memory - start_memory)

    memories = np.array(memories)
    
    print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    return max(memories), min(memories), t/i*1000, np.std(ts), nparam




def all_profile(model_type, dataset_type, windows, pred_len):
    return baseline_profile(dataset_type, windows, pred_len)
    # exp = eval(f"{model_type}Forecast", globals())(batch_size=1, dataset_type=dataset_type, windows=windows, pred_len=pred_len, data_path='/notebooks/4901_revisit_cdtran/data', device='cuda:0')
    # exp._setup_run(1000)
    # memories = []
    
    # from torch_timeseries.utils.model_stats import count_parameters
    # _, nparam = count_parameters(exp.model)
    # torch.cuda.empty_cache()
    # steps = 20
    # i = 0
    # self = exp
    # self.model.train()
    # t = 0
    # for i, (
    #     batch_x,
    #     batch_y,
    #     origin_x,
    #     origin_y,
    #     batch_x_date_enc,
    #     batch_y_date_enc,
    # ) in enumerate(self.train_loader):
    #     if i >= steps:
    #         break
    #     i+= 1
    #     origin_y = origin_y.to(self.device).float()
    #     self.model_optim.zero_grad()
    #     batch_x = batch_x.to(self.device).float()
    #     batch_y = batch_y.to(self.device).float()
    #     batch_x_date_enc = batch_x_date_enc.to(self.device).float()
    #     batch_y_date_enc = batch_y_date_enc.to(self.device).float()
    #     torch.cuda.reset_peak_memory_stats()
    #     start_memory = torch.cuda.memory_allocated()
    #     start = time.time()
        
    #     loss = self._process_one_batch(
    #         batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
    #     )
    #     loss.backward()
    #     t += time.time() - start
    #     end_memory = torch.cuda.max_memory_allocated()
    #     memories.append(end_memory - start_memory)
    #     self.model_optim.step()

    # memories = np.array(memories)

    # print(f"{dataset_type} {model_type} memory1 Max: {(max(memories)) / 1024**2:.2f}MB, Min: {(min(memories)) / 1024**2:.2f}MB, Midian: {(np.median(memories)) / 1024**2:.2f}MB ")
    # print(f"{dataset_type} {model_type} t : {t/i*1000}, nparam: {nparam}")
    # return max(memories), min(memories), t/i*1000, nparam





# def profile_main_all():
#     models = ["PatchTST", "iTransformer", "Crossformer", "CATS", "DLinear", "Informer", "TSMixer", "SCINet"]
#     # models = ["Informer, "Crossformer""]
#     datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","Electricity","ExchangeRate","Weather"]
#     # datasets = ["ETTh1","ETTh2"]
#     # datasets = ["PEMS04","PEMS07","PEMS08","PEMS_BAY"]
#     all_data = []
#     for m in models:
#         for d in datasets:
#             mem, tim, nparam = baseline_profile(m, d, 336, 720)
#             all_data.append((m,d,mem, tim, nparam))
#         df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "traintime", "parameters"])
#         df.to_csv('profile.csv')

#     print(all_data)
#     df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "traintime", "parameters"])
#     df.to_csv('profile.csv')

    
def profile_main_all():
    # models = ["CSDI", "TimeDiff", "DiffusionTS", "NsDiff", "TMDM"]
    models = [ "CSDI", "TimeDiff", "DiffusionTS", "NsDiff", "TMDM"]
    # models = ["TMDM"]
    # models = ["TimeGrad"]
    # datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","ILI","Electricity","ExchangeRate","Weather"]
    # datasets = ["Traffic","ETTh1","ETTh2","ETTm1","ETTm2","Electricity","ExchangeRate","Weather"]
    # datasets = ["ETTh1","ETTh2","ETTm1","ETTm2","ExchangeRate","Weather"]
    datasets = ["ETTh1"]
    all_data = []
    for m in models:
        for d in datasets:
            w = 168
            p = 192
            if d =="ILI":
                w = 168
                p = 36
            # mem, minme, tim, tstd, nparam = baseline_profile(m, d, w, p)
            mem, minme, tim, tstd, nparam = profile_nes(m, d, w, p)
            
            all_data.append(( m, d,mem, minme, tim, tstd, nparam))
        # df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "time", "timestd", "parameters"])
        # df.to_csv('inference_profile.csv')
    print(all_data)
    df = pd.DataFrame(all_data, columns=["model", "dataset", "peak_memory", "min_memory", "time", "timestd", "parameters"])
    df.to_csv('inference_profile.csv')


if __name__ == "__main__":
    from src.experiments import *
    profile_main_all()
# from torch_timeseries.experiments import *
# profile_pan_windows()


# profile_al_lpo()