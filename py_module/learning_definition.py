from py_module.config import Configuration

import pandas as pd
import numpy as np
import random
from sklearn import model_selection

class LearningDefinition(object):

    def __init__(self):
        self.config_obj = Configuration()

    def phm_2008_evaluation(self, model):

        