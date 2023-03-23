import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import pickle

def saves(file, name):
    with open(name + ".pkl", "wb") as write_file:
        pickle.dump(file, write_file)

def evaluation(y_m, y, path):

    json_dict = {}

    json_dict['accuracy_score'] = accuracy_score(y_m,y)
    json_dict['confusion_matrix'] = confusion_matrix(y_m,y)
    json_dict['precision_recall_fscore'] = precision_recall_fscore_support(y_m,y)

    
    saves(json_dict, path+'\\evaluation_metrics')