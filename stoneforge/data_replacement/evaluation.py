import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
#import pickle
import json


def _saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)


def evaluation(y_m, y, decimals = 3, path = '.'):

    json_dict = {}

    json_dict['mean_absolute_percentage_error'] = np.round(mean_absolute_percentage_error(y_m,y),decimals)

    #print(list(confusion_matrix(y_m,y)),'\n')
    #print(list(precision_recall_fscore_support(y_m,y)))
    #json_dict['confusion_matrix'] = list(confusion_matrix(y_m,y))
    #json_dict['precision_recall_fscore'] = list(precision_recall_fscore_support(y_m,y))

    lito = list(set(y))
    json_dict['facies'] = lito

    prf = []
    for i in list(mean_squared_error(y_m,y, labels = np.array(lito))):
        values = np.round(i,decimals)
        values = np.array(values,dtype='str')
        prf.append(list(values))
    json_dict['mean_squared_error'] = prf
    
    if path:
        _saves(json_dict, path+'\\evaluation_metrics')
    if not path:
        return json_dict