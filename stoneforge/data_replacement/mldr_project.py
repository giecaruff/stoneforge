import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os

from . fit import fit
from . predict import predict
from . settings import settings

class project:

    def __init__(self, data, name = "ml", project_path = ".", remove_null = False):

        # ====================================================== #

        _temp_data = {}
        if remove_null:
            if isinstance(remove_null, int) or isinstance(remove_null, float):
                for d in data:
                    _temp_data[d] = data[d][~data[d].isin([remove_null]).any(axis=1)]
            if isinstance(remove_null, bool):
                for d in data:
                    _temp_data[d] = data[d].dropna()
        if not remove_null:
            _temp_data = data
            
        data = _temp_data
        
        self.project_path = os.path.join(project_path, name)

        if os.path.exists(self.project_path):
            shutil.rmtree(self.project_path)
        os.makedirs(self.project_path)
        
        self.data = data
        self.crop_data = data

        self.method = "linear_regression_simple"

        # ====================================================== #

    def settings(self, method = False):
        if not method:
            method = self.method
        settings(method = method, path = self.project_path)

    def fit(self, fit_info):

        X_data = []
        Y_data = []
        for info in fit_info:

            data_x, data_y = self._select_info(info)

            X_data.append(data_x)
            Y_data.append(data_y)

        X = np.concatenate(X_data)
        Y = np.concatenate(Y_data)
        
        fit(X = X, y = Y, method = self.method, path = self.project_path)

    
    def predict(self, predict_info, curve_name = "new_log"):
        
        X_data = []
        for info in predict_info:

            data_x,_ = self._select_info(info)

            data_y = predict(data_x, method = self.method, path = self.project_path)
            
            well = info["well"]
            self.crop_data[well][curve_name] = data_y

    def return_data(self, well = False):
        if well:
            return self.crop_data[well]
        else:
            return self.crop_data
        
    def _select_info(self, info):

        well = info["well"]
        data = self.data[well]
        
        if "depth" in info:
            depth = info["depth"]
        else:
            depth = data.columns[0]
        if "range" in info:
            if "top" in info["range"]:
                top = info["range"]["top"]
            else:
                top = data[depth].min()
            if "bottom" in info["range"]:
                bottom = info["range"]["bottom"]
            else:
                bottom = data[depth].max()
        else:
            top = data[depth].min()
            bottom = data[depth].max()

        self.crop_data[well] = data[data[depth].between(top, bottom)]

        data_x = np.array(self.crop_data[well][list(info["X"])])
        if "Y" in info:
            data_y = np.array(self.crop_data[well][list(info["Y"])])
            return data_x, data_y
        else:
            return data_x, 0