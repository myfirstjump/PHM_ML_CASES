import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from py_module.config import Configuration

class PHMDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        file_paths: 數據文件的路徑列表，可以包含不同格式的文件
        transform: 用於數據增強的變換（例如標準化、歸一化）
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # 應用變換
        if self.transform:
            sample = self.transform(sample)

        return sample, label

class LoadAndPreprocessingPHMDatasets(object):

    def __init__(self):
        self.config = Configuration()
    
    def read_phm_data(self, data_string, train_flag=True):

        data = pd.DataFrame()
        labels = []
        unique_units = None
        
        ### PHM2008
        if data_string == 'PHM_2008':
            files_range = []
            if train_flag:
                files_range.append('train.txt')
            file_paths = self.config.folder_2008_phm
            for sub_path in os.listdir(file_paths):
                if sub_path in files_range:
                    path = os.path.join(file_paths, sub_path)
                    df = pd.read_csv(path, sep=' ',header=None, names=self.config.features_name_2008_phm)
                    df = df.drop(labels=['sensor_22', 'sensor_23'], axis='columns')
                    unique_units = df['unit'].unique()
                    
                    if data.empty:
                        data = df
                    else:
                        data = pd.concat([data, df], axis=0)

                else:
                    continue

        # print(f'{data_string} data shape: {data.shape}')
        # print(f'head 3 data:\n {data[:3]}')
        # print(f'{data_string} labels shape: {labels.shape}')
        # print(f'head 3 labels:\n {labels[:3]}')

        return data, unique_units

    def phm_2008_labels_define(self, x_data):
        """
        Function:
            定義2008PHM引擎training資料集的supervised learning模式，新增RUL欄位。
            定義方式為cycle的反序列，比如說一個引擎資料有1~200個cycles，那個RUL的序列即為199, 198, 197, ..., 0。
        Input:
            Training Data
        Output:
            RUL label array
        """
        data = x_data.copy()
        units = data['unit'].unique()
        RUL_list = []

        for unit in units:
            unit_data = data.loc[data.unit == unit]
            nrow = len(unit_data.index)
            unit_RUL = [i for i in range(0, nrow)][::-1]
            RUL_list = RUL_list + unit_RUL
      

        return RUL_list