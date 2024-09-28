from py_module.config import Configuration
from py_module.data_preprocessing import PHMDataset, LoadAndPreprocessingPHMDatasets
from py_module.data_training import DataTraining
from py_module.data_evaluation import DataEvaluation

import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PHMProcedures(object):

    """
    Main Flow:
    """

    def __init__(self):
        self.preprocess = LoadAndPreprocessingPHMDatasets()
        self.training = DataTraining()
        self.evaluation = DataEvaluation()

    def phm_2008_data_case(self):
        data, unique_units = self.preprocess.read_phm_data(data_string="PHM_2008")

        train_units, test_units = train_test_split(unique_units, test_size=0.2, random_state=42)
        X_train = data[data['unit'].isin(train_units)]
        X_test = data[data['unit'].isin(test_units)]

        y_train = self.preprocess.phm_2008_labels_define(X_train)
        y_test = self.preprocess.phm_2008_labels_define(X_test)
        X_train = X_train.drop(labels=['cycle', 'unit'], axis='columns')
        X_test = X_test.drop(labels=['cycle', 'unit'], axis='columns')

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_dataset = PHMDataset(X_train_scaled, y_train)
        test_dataset = PHMDataset(X_test_scaled, y_test)

        model_name, model = self.training.phm_2008_data_training(train_dataset)

        self.evaluation.phm_2008_data_evaluation(test_dataset, (model_name, model))


        


def main_flow():
    
    main_obj = PHMProcedures()
    main_obj.phm_2008_data_case()


if __name__ == "__main__":
    main_flow()

