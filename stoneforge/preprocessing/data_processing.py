from sklearn.model_selection import train_test_split
import numpy as np

class predict_processing:

    def __init__(self,data,data_key):
        self.data = data
        self.data_key = data_key
        self.idx = {}
        self.clean_data = {}


    def _nan_idx(self):

        for i in self.data:
            local = []
            for j in range(len(self.data[i][self.data_key].T)):
                if not np.isnan(self.data[i][self.data_key].T[j]).any():
                    local.append(j)
            self.idx[i] = local


    def matrix_values(self):
        self._nan_idx()

        clean_data = {}
        for i in self.data:
           
           clean_data[i] = self._remove_dummies(self.data[i][self.data_key])

        return clean_data


    def return_curve(self,y):

        curves = {}
        for i in self.data:
            curve = np.empty((np.shape(self.data[i][self.data_key])[1]))
            curve[:] = np.nan
            #print('h',np.shape(y[i]),np.shape(self.idx[i]),np.shape(curve))
            for idx in range(len(y[i])):
                #print(self.idx[i][idx])
                curve[self.idx[i][idx]] = y[i][idx]
            curves[i] = curve

        return curves
    

    def train_test_split(self, X, y, test_size = 0.30, random_state = 99):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        curves = {
            "X_train":X_train,
            "X_test":X_test,
            "y_train":y_train,
            "y_test":y_test
        }
        return curves

    
    def _remove_dummies(self,data):
        data_1 = np.array(data).T
        data_2 = data_1[~np.isnan(data_1).any(axis=1)]

        return data_2

    def train_test_split(self, X, y, test_size = 0.30, random_state = 99):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        curves = {
            "X_train":X_train,
            "X_test":X_test,
            "y_train":y_train,
            "y_test":y_test
        }
        return curves

    
    def _remove_dummies(self,data):
        data_1 = np.array(data).T
        data_2 = data_1[~np.isnan(data_1).any(axis=1)]

        return data_2

def well_train_test_split(well_names,well_database):

    all_wells = set(well_database.keys())
    v_wells = set(well_names)
    t_wells = all_wells - v_wells

    t_database = {}
    for w in list(t_wells):
        t_database[w] = well_database[w]

    v_database = {}
    for w in list(v_wells):
        v_database[w] = well_database[w]

    return (t_database,v_database)

# ===================================================== #

def data_assemble(main_data, data_key):
    """Transform a dictionary of dict[wells]['data_key'][data]
    into a dictionary of compact data like
    dict[data], mostly used for machine learning purpose
    """
    wells = list(main_data.keys())
    I = np.shape(main_data[wells[0]][data_key])[0]

    mega_data = []
    for j in range(I):
        local = []
        for i in main_data:
            local = local+list(main_data[i][data_key][j])
        mega_data.append(local)

    mega_data = np.array(mega_data)

    return mega_data

def _remove_dummies(data):
    data_1 = np.array(data).T
    data_2 = data_1[~np.isnan(data_1).any(axis=1)]

    return data_2



