import numpy as np
import os
from typing import Annotated
import pandas as pd

from . import las2

class project():
    """Creates a project object to manage well log data.
    
    Example
    -------
    >>> proj = project(data_path='path/to/well/logs')
    >>> proj.import_folder(ext='.las')  # Import all LAS files in the folder
    >>> print(proj.well_names_paths)  # Check imported well names and paths
    
    >>> proj.import_several_wells()  # Import all wells data in a given folder into the project
    
    """
    
    def __init__(
        self,
        data_path : Annotated [str, "Directory / Folder with well log data"] = '.'):
        """Initializes the project with a specified data path.
        
        Parameters
        ----------
        data_path : str, optional
            The directory where well log data files are located. Default is the current directory ('.').
        """
        
        self.project = {}
        self.data_path = data_path
        self.outpath = '.'
        self.well_names_paths = {}
        self.well_data = {}
        self.well_names_las = []
        
    # ============================================ #

    def import_folder(
        self,
        ext : Annotated [str, "Extension of the imported data"] = '.las') -> None:
        """Imports all file paths with a given extension from a folder into the project.
        
        Parameters
        ----------
        ext : str, optional
            The file extension to look for in the folder. Default is '.las'.
            
        Example
        -------
        >>> proj.import_folder(ext='.las')  # Import all LAS files in the folder
        >>> print(proj.well_names_paths)  # Check imported well names and paths
        """

        # ------------------------------------ #
        # all paths 
        files = []
        # r=root, d=directories, f = files
        for r, _, f in os.walk(self.data_path):
            for file in f:
                if ext in file:
                    files.append(os.path.join(r, file))

        c_resumo = self.data_path+'\\'


        for i in files:
            n1 = i.replace(c_resumo, '')
            self.well_names_paths[n1.replace(ext,'')] = i

    # ============================================ #

    def import_well(
        self,
        name : Annotated [str, "path to the well log data (for .las files)"]) -> None:
        """Imports a single well log data from a specified file path into the project.
        
        Parameters
        ----------
        name : str
            The name of the well log data file (without extension) to be imported.
            
        Example
        -------
        >>> proj.import_well(name='well1')  # Import well log data from 'well1.las'
        """
        
        # ------------------------------------ #
        
        path = self.well_names_paths[name]
        self.well_names_las.append(name)

        read_data = las2.read(path)

        mnemonic = [a['mnemonic'] for a in read_data['curve']]
        unit = [a['unit'] for a in read_data['curve']]
        self.well_data[name] = {}
 
        for i in range(len(mnemonic)):
            self.well_data[name][mnemonic[i]] = {}
            self.well_data[name][mnemonic[i]]['data'] = read_data['data'][i]
            self.well_data[name][mnemonic[i]]['unit'] = unit[i]

    # ============================================ #

    def import_several_wells(self):
        """Imports all well log data from the specified folder into the project.
        
        Example
        -------
        >>> proj.import_several_wells()
        """

        for name in self.well_names_paths:
            self.import_well(name)

    # ============================================ #

    def data_replacement(
        self,
        ref : Annotated [dict, "dictionary with new mnemonics as keys and lists of old mnemonics as values"]) -> None:
        """Replaces mnemonics in the well data with those from a reference dictionary.
        
        Parameters
        ----------
        ref : dict
            A dictionary where keys are new mnemonics and values are lists of old mnemonics to be replaced.
            
        Example
        -------
        >>> ref = {
        ...     'RHOB': ['RHO', 'RHOZ'],   # New mnemonic 'RHOB' replaces 'RHO' and 'RHOZ'
        ...     'NPHI': ['PHI', 'PHIN']    # New mnemonic 'NPHI' replaces 'PHI' and 'PHIN'
        ... }
        >>> proj.data_replacement(ref)
        """

        mnemonics_list = list(ref.keys())

        new_well_data = {}
        for i in self.well_data:
            new_well_data[i] = {}
            local = {}

            for j in self.well_data[i]:
                new_mnemonic = self._find_mnemonic(j,ref)
                if new_mnemonic:
                    local[new_mnemonic[0]] = self.well_data[i][j]
                else:
                    pass
            new_well_data[i] = local

        self.well_data = new_well_data

    def _find_mnemonic(self,value,ref):

        for i in ref:
            for j in ref[i]:
                if value == j:
                    return i,value

    # ============================================ #

    def convert_into_matrix(
        self,
        reference_mnemonics : Annotated [list," A list of mnemonics to be used as a reference for the well data"]=False):
        """Converts an manly dictionary database into an matrix database with tree values: mnemonics, units and data.
        
        Parameters
        ----------
        reference_mnemonics : list, optional
            A list of mnemonics to be used as a reference for the well data. (If not provided, all mnemonics in the well data will be used).
            
        Example
        -------
        >>> proj.convert_into_matrix(reference_mnemonics=['RHOB', 'NPHI', 'GR'])
        """
        
        wells = {}
        for i in self.well_data:
            data = []
            units = []
            mnemonics = []
            well = {}
            if reference_mnemonics:
                well_data = reference_mnemonics
            else:
                well_data = self.well_data[i]

            for j in well_data:
                # print(i,j) - search for wells mnemonics
                data.append(self.well_data[i][j]['data'])
                units.append(self.well_data[i][j]['unit'])
                mnemonics.append(j)

            well['mnemonics'] = mnemonics
            well['units'] = units
            well['data'] = np.array(data)
            wells[i] = well

        self.well_data = wells

    # ============================================ #

    def class_counts(
        self,
        class_value : Annotated [list, "List of class values to count"],
        class_dict : Annotated [dict, "Dictionary "] = False,
        seed : Annotated [dict, "Dictionary "] = 99):
        """Counts the occurrences of each class value in a list and returns a dictionary with class names,  random colors, and counts. (for fast plot)
        
        Parameters
        ----------
        class_value : list
            A list of class values to count.
        class_dict : dict, optional
            A dictionary containing class codes, names, and colors for substitution. If not provided, random colors will be generated.
        seed : int, optional
            A seed for random number generation to ensure reproducibility. Default is 99.
            
        Example
        -------
        >>> class_values = [57, 54, 25, 49]
        >>> class_dict = [
        ...     {"code": 57, "name": "Sand", "patch_property": {"color": "#FF0000"}},
        ...     {"code": 54, "name": "Shale", "patch_property   : {"color": "#00FF00"}},
        ...     {"code": 25, "name": "Coal", "patch_property": {"color": "#0000FF"}},
        ...     {"code": 49, "name": "Limestone", "patch_property": {"color": "#FFFF00"}}
        """

        np.random.seed(seed)

        n_class = list(set(class_value))
        class_count = []
        for c in n_class:
            name = c
            r = lambda: np.random.randint(0,255)
            color = '#%02X%02X%02X' % (r(),r(),r())
            values_dictionary = {}
            values_dictionary['value'] = str(c)
            if class_dict:
                substitution_dict = 0
                for i in class_dict:
                    if i["code"] == c:
                        substitution_dict = i
                        name = substitution_dict['name']
                        color = substitution_dict['patch_property']['color']
                values_dictionary['name'] = name
                values_dictionary['color'] = color
            else:
                values_dictionary['name'] = name
                values_dictionary['color'] = color
            counts = 0
            for i in class_value:
                if i == c:
                    counts += 1
            values_dictionary['count'] = str(counts)
            class_count.append(values_dictionary)

        return class_count

    def shape_check(
        self,
        ref : Annotated [dict, "dictionary with new mnemonics as keys and lists of old mnemonics as values"]) -> None:
        """If an well has less mnemonics than the others, than this function removes this well.
        
        Parameters
        ----------
        ref : dict
            A dictionary where keys are new mnemonics and values are lists of old mnemonics to be replaced.
            
        Example
        -------
        >>> ref = {
        ...     'RHOB': ['RHO', 'RHOZ'],
        ...     'NPHI': ['PHI', 'PHIN']
        ... }
        
        >>> proj.shape_check(ref) # Removes wells with less mnemonics than the reference dictionary.
        """
        value = len(ref.keys())

        well_data = {}

        for i in self.well_data:
            if np.shape(self.well_data[i]['data'])[0] == value:
                well_data[i] = self.well_data[i]
            else:
                print("well: '{}'".format(i),"because it has less logs")

        self.well_data = well_data

# ============================================ #
    
def depth_zones(
    df : Annotated [pd.DataFrame, "Pandas DataFrame with depth data"],
    dept : Annotated [str, "Depth column name"],
    ranges : Annotated [tuple, "Depth column name"]):
    """Given a DataFrame and a depth column, this function creates zones based on the specified depth ranges.
    
    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the depth data.
    dept : str
        The name of the depth column in the DataFrame.
    ranges : tuple
        A tuple containing the depth ranges to create zones. The first element is the top depth, the last element is the bottom depth, and the middle elements are the range boundaries.
        
    Returns
    -------
    dict
        A dictionary where keys are zone indices and values are DataFrames containing the data for each zone.
        
    Example
    -------
    >>> df = pd.DataFrame({'Depth': [100, 200, 300, 400, 500], 'Value': [1, 2, 3, 4, 5]})
    >>> dept = 'Depth'
    >>> ranges = (150, 250, 350)
    >>> zones = depth_zones(df, dept, ranges)
    >>> for zone, data in zones.items():
    ...     print(f"Zone {zone}:")
    ...     print(data)
    >>> # Output:
    >>> # Zone 0:
    >>> #    Depth  Value
    >>> # 0    100      1
    >>> # Zone 1:
    >>> #    Depth  Value
    >>> # 1    200      2

    """

    DEPT = np.array(df[dept])
    ranges = list(ranges)
    top = sorted(DEPT)[0]
    bot = sorted(DEPT)[-1]
    ranges = [top] + ranges + [bot]

    _zones = {}
    for i in range(len(ranges)-1):
        top = ranges[i]
        bot = ranges[i+1]
        _zones[i] = df[df[dept].between(top, bot)]
    
    return _zones