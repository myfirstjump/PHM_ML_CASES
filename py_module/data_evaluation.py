import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader


from py_module.config import Configuration
from py_module.plot_module import PlotDesign

class DataEvaluation(object):

    def __init__(self):
        self.config_obj = Configuration()
        self.plotting_obj = PlotDesign()


    def phm_2008_data_evaluation(self, model, test_dataset):

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        X_test, y_test = [], []
        for inputs, labels in test_loader:
            X_test.extend(inputs.numpy())
            y_test.extend(labels.numpy())

        # 預測與評估
        predictions = model.predict(X_test)
        
        # 計算評估指標（均方誤差和 R² 分數）
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"XGBoost Mean Squared Error: {mse:.4f}")
        print(f"XGBoost R² Score: {r2:.4f}")


        # 作圖
        fig_len = 2000
        plt.figure(figsize=(12, 6))

        # 繪製真實值和預測值的折線圖
        plt.plot(range(len(y_test[:fig_len])), y_test[:fig_len], label='True Values', color='b')
        plt.plot(range(len(predictions[:fig_len])), predictions[:fig_len], label='Predictions', color='r', linestyle='dashed')

        # 添加標題和標籤
        plt.title('Engine Remaining Useful Life: True vs Predictions')
        plt.xlabel('Sample Index')
        plt.ylabel('Remaining Cycles')
        plt.legend()

        # 顯示圖表
        plt.show()



