import time
import os
import random
import itertools

import numpy as np

from sklearn import model_selection


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

