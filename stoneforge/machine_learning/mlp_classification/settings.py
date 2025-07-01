import json

ML_METHODS = [
    "MLPClassifier"]

def saves(file, name):
    with open(name+'.json', 'w') as write_file:
        json.dump(file, write_file)

def settings(method: str = "MLPClassifier", path = ".", **kwargs):

    if method == "MLPClassifier":
        saves(kwargs, path+"\\multi_layer_perceptron_settings")

    saves(ML_METHODS, path+'\\all_methods')