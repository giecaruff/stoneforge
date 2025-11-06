
import os
import pandas as pd
import warnings

if __package__:
    from ..io.dlisio_r import DLISAccess
    from ..io.las2 import LAS2Parser
    from ..io.las3 import LAS3Parser
    from ..io.tabr import TABParser
else:
    from stoneforge.io.dlisio_r import DLISAccess
    from stoneforge.io.las2 import LAS2Parser
    from stoneforge.io.las3 import LAS3Parser
    from stoneforge.io.tabr import TABParser



class DataLoader:
    
    def __init__(self, filepath, filetype=None, gui=False, sep="\t", std="US"):
        """
        Import a file into the project.
        
        Parameters
        ----------
        filepath : str
            Path to the file to be imported.
        filetype : str, optional
            Type of the file. If None, it will be inferred from the file extension.
        

        Returns
        -------
        - Depending on the file type, it will return:
        """
        self.data_obj = None

        if filetype == 'las2':
            self.data_obj = LAS2Parser(filepath)

        if filetype == 'las3':
            self.data_obj = LAS3Parser(filepath)

        if filetype == 'dlis':
            self.data_obj = DLISAccess(filepath, gui=gui)

        if filetype == 'tabr':
            try:
                self.data_obj = TABParser(filepath, sep=sep, std=std)
            except:
                print("Failed to parse tabular data file.")

        if filetype is None:
            filext = self._get_file_extension(filepath)
            if filext == '.las':
                try:
                    print("filetype '.las' assumed to be LAS2, trying to parse as LAS2...")
                    self.data_obj = LAS2Parser(filepath)
                    print("LAS2 parsing successful.")
                except:
                    try:
                        print("Failed to parse as LAS2, trying LAS3")
                        self.data_obj = LAS3Parser(filepath)
                        print("LAS3 parsing successful.")
                    except:
                        raise ValueError("Failed to parse .las file as either LAS2 or LAS3.")
            elif filext == '.dlis':
                print("Trying to parse as DLIS data file due to '.dlis' extention ...")
                self.data_obj = DLISAccess(filepath, gui=gui)
                print("DLIS parsing successful.")
            elif filext in ['.csv', '.txt', '.dat', '.tsv']:
                try:
                    self.data_obj = TABParser(filepath, sep=sep, std=std)
                except:
                    print("Failed to parse tabular data file.")
            else:
                raise ValueError(f"Unsupported file extension: {filext}")
    
    def dataframe(self, data):
        """
        Convert data to a pandas DataFrame.
        
        Parameters
        ----------
        - data: dict or array-like, input data
        
        Returns
        -------
        - df: pandas DataFrame
        """
        dataframe = {}
        units = {}
        for d in data:
            dataframe[d] = data[d]['values']
            units[d] = data[d]['unit']

        df = pd.DataFrame.from_dict(dataframe)
        return df, units
                
    def _get_file_extension(self, file_path):
        """
        Returns the extension of a file from its path
        """
        return os.path.splitext(file_path)[1]