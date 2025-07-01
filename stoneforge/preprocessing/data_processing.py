from sklearn.model_selection import train_test_split
import numpy as np
from typing import Annotated

class predict_processing:
    """Processes the data for machine learning predictions.
    This class handles the preparation of data for machine learning predictions, including handling NaN values and splitting data into training and testing sets.
    
    Example
    -------
    >>> pp = predict_processing(data, data_key='data')
    >>> clean_data = pp.matrix_values() # Returns a dictionary of cleaned data without NaN values.
    >>> curves = pp.return_curve(y) # Returns a dictionary of curves with values filled in
    >>> train_test_data = pp.train_test_split(X, y) # Splits the data into training and testing sets.
    >>> train_data, valid_data = well_train_test_split(well_names, well_database) 
    >>> mega_data = data_assemble(main_data, data_key='data') # Assembles data from multiple wells into a single matrix.
    """

    def __init__(
        self,
        data : Annotated[dict, "Dictionary of well data with well names as keys and data as values"],
        data_key : Annotated[str, "Key for the data in the well data dictionary"]):
        """Initializes the predict_processing class with well data and a data key."""
        
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
        """Returns a dictionary of cleaned data without NaN values.
        
        Example
        -------
        >>> clean_data = pp.matrix_values() # Returns a dictionary of cleaned data without NaN"""
        self._nan_idx()

        clean_data = {}
        for i in self.data:
           
           clean_data[i] = self._remove_dummies(self.data[i][self.data_key])

        return clean_data

    def return_curve(
        self,
        y : Annotated[dict, "Dictionary of values to be filled in the curves"]):
        """Returns a dictionary of curves with values filled in.
        """

        curves = {}
        for i in self.data:
            curve = np.empty((np.shape(self.data[i][self.data_key])[1]))
            curve[:] = np.nan
            for idx in range(len(y[i])):
                curve[self.idx[i][idx]] = y[i][idx]
            curves[i] = curve

        return curves
    
    def _remove_dummies(self,data):
        data_1 = np.array(data).T
        data_2 = data_1[~np.isnan(data_1).any(axis=1)]
        return data_2

    def train_test_split(
        self,
        X : Annotated[np.array, "Feature data for training and testing"],
        y : Annotated[np.array, "Target data for training and testing"],
        test_size : Annotated[float, "Split rule"] = 0.30,
        random_state : Annotated[float, "Seed"] = 99):
        """Splits the data into training and testing sets."""

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

def well_train_test_split(
    well_names : Annotated[list, "List of well names for validation"],
    well_database : Annotated[dict, "Dictionary of well data with well names as keys and data as values"]):
    """Splits the well database into training and testing sets based on well names.
    
    Parameters
    ----------
    well_names : list
        A list of well names to be used for validation.
    well_database : dict
        A dictionary containing well data, where keys are well names and values are the corresponding data.
        
    Returns
    -------
    tuple
        A tuple containing two dictionaries: the first for training wells and the second for validation wells.
        
    Example
    -------
    >>> well_names = ['Well1', 'Well2']
    >>> well_database = {'Well1': data1, 'Well2': data2, 'Well3': data3}
    >>> train_data, valid_data = well_train_test_split(well_names, well_database)
    """

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
    """Transform a dictionary of dict[wells]['data_key'][data] into a dictionary of compact data like dict[data], mostly used for machine learning purpose.
    
    Parameters
    ----------
    main_data : dict
        A dictionary containing well data, where keys are well names and values are dictionaries with data.
    data_key : str
        The key for the data in the well data dictionary.
        
    Returns
    -------
    np.array
        A numpy array containing the assembled data from all wells, with each row corresponding to a data point from each well.
        
    Example
    -------
    >>> main_data = {
    ...     'Well1': {'data_key': [[1, 2], [3, 4]]},
    ...     'Well2': {'data_key': [[5, 6], [7, 8]]}
    ... }
    >>> data_key = 'data_key'
    >>> mega_data = data_assemble(main_data, data_key)
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



