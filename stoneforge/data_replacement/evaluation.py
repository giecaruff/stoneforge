import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
#from sklearn.metrics import mean_absolute_percentage_error
#import pickle mean_absolute_error
import json


def _saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def _mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def evaluation(y_m, y, decimals = 3, path = '.'):

    json_dict = {}

    l = len(y_m)

    #json_dict['mean_absolute_percentage_error'] = np.round(mean_absolute_percentage_error(y_m,y),decimals)
    json_dict['pearson_corrcoef'] = np.round(np.corrcoef(y_m,y)[0, 1],decimals)
    json_dict['mape'] = np.round(_mean_absolute_percentage_error(y_m,y),decimals)
    json_dict['r2'] = np.round(r2_score(y_m,y),decimals)

    #print(list(confusion_matrix(y_m,y)),'\n')
    #print(list(precision_recall_fscore_support(y_m,y)))
    #json_dict['confusion_matrix'] = list(confusion_matrix(y_m,y))
    #json_dict['precision_recall_fscore'] = list(precision_recall_fscore_support(y_m,y))


    if path:
        _saves(json_dict, path+'\\regression_metrics')
    if not path:
        return json_dict