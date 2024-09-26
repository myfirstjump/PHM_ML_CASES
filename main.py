from py_module.config import Configuration
from py_module.data_preprocessing import PHMDataset, LoadAndPreprocessingPHMDatasets

import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EngineCycleTraining(object):

    """
    Main Flow:
    """

    def __init__(self):
        self.preprocess = LoadAndPreprocessingPHMDatasets()

    def phm_2008_data_training(self):
        data, unique_units = self.preprocess.read_phm_data(data_string="PHM_2008")

        train_units, test_units = train_test_split(unique_units, test_size=0.2, random_state=42)
        X_train = data[data['unit'].isin(train_units)]
        X_test = data[data['unit'].isin(test_units)]

        y_train = self.preprocess.phm_2008_labels_define(X_train)
        y_test = self.preprocess.phm_2008_labels_define(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = PHMDataset(X_train_scaled, y_train)
        test_dataset = PHMDataset(X_test_scaled, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        ### 測試DataLoader功能
        # for inputs, labels in test_loader:
        #     print(inputs, labels)
        #     break

        


def main_flow():
    
    main_obj = EngineCycleTraining()
    main_obj.phm_2008_data_training()


if __name__ == "__main__":
    main_flow()

