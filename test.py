import sys

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("2023F/Dataloader")
from data_quality_control import median_filter

def cubization(path_list: list, index_category: str):
    """将一个npy列表转为3维ndarray数组, 并归一化"""
    results = None
    for path in path_list:
        raw_data = median_filter(path, index_category)
        raw_data = raw_data.reshape(1, raw_data.shape[0], raw_data.shape[1])
        if results is not None:
            results = np.concatenate((results, raw_data), axis=0)
        else:
            results = raw_data
    return results

class sequential_dataset(data.Dataset):
    def __init__(self, df, df_radar, transforms=None):
        self.df = df
        self.df_radar = df_radar
        
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df_radar)
    
    def __getitem__(self, index):
        Y_3km_10_path_list = eval(self.df_radar.loc[index]["output"])
        Y_1km_10_path_list = eval(str(Y_3km_10_path_list).replace("3.0km", "1.0km"))
        Y_7km_10_path_list = eval(str(Y_3km_10_path_list).replace("3.0km", "7.0km"))
        
        Zh_3km_10_path_list = eval(self.df_radar.loc[index]["input"])
        Zh_7km_10_path_list = eval(str(Zh_3km_10_path_list).replace("3.0km", "7.0km"))
        Zh_1km_10_path_list = eval(str(Zh_3km_10_path_list).replace("3.0km", "1.0km"))
        
        Zdr_1km_10_path_list = eval(str(Zh_1km_10_path_list).replace("dBZ", "ZDR"))
        Zdr_3km_10_path_list = eval(str(Zh_3km_10_path_list).replace("dBZ", "ZDR"))
        Zdr_7km_10_path_list = eval(str(Zh_7km_10_path_list).replace("dBZ", "ZDR"))
        
        Kdp_1km_10_path_list = eval(str(Zh_1km_10_path_list).replace("dBZ", "KDP"))
        Kdp_3km_10_path_list = eval(str(Zh_3km_10_path_list).replace("dBZ", "KDP"))
        Kdp_7km_10_path_list = eval(str(Zh_7km_10_path_list).replace("dBZ", "KDP"))
        
        Y_1km_10 = cubization(Y_1km_10_path_list, index_category="dBZ")
        Y_3km_10 = cubization(Y_3km_10_path_list, index_category="dBZ")
        Y_7km_10 = cubization(Y_7km_10_path_list, index_category="dBZ")
        
        Zh_1km_10 = cubization(Zh_1km_10_path_list, index_category="dBZ")
        Zh_3km_10 = cubization(Zh_3km_10_path_list, index_category="dBZ")
        Zh_7km_10 = cubization(Zh_7km_10_path_list, index_category="dBZ")

        Zdr_1km_10 = cubization(Zdr_1km_10_path_list, index_category="ZDR")
        Zdr_3km_10 = cubization(Zdr_3km_10_path_list, index_category="ZDR")
        Zdr_7km_10 = cubization(Zdr_7km_10_path_list, index_category="ZDR")
        
        Kdp_1km_10 = cubization(Kdp_1km_10_path_list, index_category="KDP")
        Kdp_3km_10 = cubization(Kdp_3km_10_path_list, index_category="KDP")
        Kdp_7km_10 = cubization(Kdp_7km_10_path_list, index_category="KDP")
        print(Zh_3km_10_path_list)
        # Y_1km_10 = torch.tensor(Y_1km_10)
        # Y_3km_10 = torch.tensor(Y_3km_10)
        # Y_7km_10 = torch.tensor(Y_7km_10)
        # Zh_1km_10 = torch.tensor(Zh_1km_10)
        # Zh_3km_10 = torch.tensor(Zh_3km_10)
        # Zh_7km_10 = torch.tensor(Zh_7km_10)
        # Zdr_1km_10 = torch.tensor(Zdr_1km_10)
        # Zdr_3km_10 = torch.tensor(Zdr_3km_10)
        # Zdr_1km_10 = torch.tensor(Zdr_1km_10)
        # Zdr_3km_10 = torch.tensor(Zdr_3km_10)
        # Zdr_7km_10 = torch.tensor(Zdr_7km_10)
        
        results = {
            # All Shape is 10 * 256 * 256
            "Y_1km_10": Y_1km_10,
            "Y_3km_10": Y_3km_10,
            "Y_7km_10": Y_7km_10,
            
            "Zh_1km_10": Zh_1km_10,
            "Zh_3km_10": Zh_3km_10,
            "Zh_7km_10": Zh_7km_10,
            
            "Zdr_1km_10": Zdr_1km_10,
            "Zdr_3km_10": Zdr_3km_10,
            "Zdr_7km_10": Zdr_7km_10,
            
            "Kdp_1km_10": Kdp_1km_10,
            "Kdp_3km_10": Kdp_3km_10,
            "Kdp_7km_10": Kdp_7km_10,
        }
        return results
df = pd.read_csv("/home/zhouquan/MyDoc/Math_Model/Dataset/q1_radar_raw_dataset.csv")
df_radar = pd.read_csv(f"Dataset/New/radar_3km_062_test.csv")
a = sequential_dataset(df, df_radar)

a.__getitem__(20)


