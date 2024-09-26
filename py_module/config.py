import os
import time

class Configuration(object):
    
    def __init__(self):

        self.working_dir = os.getcwd()
        self.folder_2008_phm = os.path.join(self.working_dir, "datasets", "Data_2008_PHM")
        self.features_name_2008_phm = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',] + ['sensor_' + str(i) for i in range(1, 24)]
