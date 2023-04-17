import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
#import pickle
import json
from sklearn.preprocessing import LabelEncoder

#def saves(file, name):
#    with open(name + ".pkl", "wb") as write_file:
#        pickle.dump(file, write_file)

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def encoded(y):

    le = LabelEncoder()
    return le.fit_transform(y)

def evaluation(y_m, y, decimals = 3, path = '.'):

    json_dict = {}

    json_dict['accuracy_score'] = np.round(accuracy_score(y_m,y),decimals)

    #print(list(confusion_matrix(y_m,y)),'\n')
    #print(list(precision_recall_fscore_support(y_m,y)))
    #json_dict['confusion_matrix'] = list(confusion_matrix(y_m,y))
    #json_dict['precision_recall_fscore'] = list(precision_recall_fscore_support(y_m,y))

    lito = list(set(y))
    json_dict['facies'] = lito

    prf = []
    for i in list(precision_recall_fscore_support(y_m,y)):
        values = np.round(i,decimals)
        values = np.array(values,dtype='str')
        prf.append(list(values))
    json_dict['precision_recall_fscore'] = prf
    
    cm = []
    for i in list(confusion_matrix(y_m,y)):
        values = np.array(i,dtype='str')
        cm.append(list(values))
    json_dict['confusion_matrix'] = cm

    saves(json_dict, path+'\\evaluation_metrics')