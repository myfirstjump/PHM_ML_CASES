import time
import os
import random
import itertools

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb

from py_module.config import Configuration


class DataTraining(object):

    def __init__(self):
        self.config = Configuration()


    def sys_show_execution_time(method):
        def time_record(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = np.round(end_time - start_time, 3)
            print('Running function:', method.__name__, ' cost time:', execution_time, 'seconds.')
            return result
        return time_record
    
    def phm_2008_data_training(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 提取訓練集特徵和標籤
        X_train, y_train = [], []
        for inputs, labels in train_loader:
            X_train.extend(inputs.numpy())
            y_train.extend(labels.numpy())

        # ------------------------
        # 使用 XGBoost 進行迴歸建模
        # ------------------------
        xgb_model = xgb.XGBRegressor()
        xgb_model.fit(X_train, y_train)

        features = self.config.features_name_2008_phm[:-2]
        features.remove('unit')
        features.remove('cycle')
        importances = xgb_model.feature_importances_
        importances_df = pd.DataFrame({'features':features, 'importance':importances})
        print("Feature importances:\n", importances_df)


        return ('xgb_model', xgb_model)

