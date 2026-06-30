
import os
import pandas as pd
import warnings
from urllib.parse import urlparse
import requests
import tempfile
from pathlib import Path
import numpy as np
from typing import Annotated

if __package__:
    #from ..io.dlisio_r import DLISAccess
    from ..io.dlist import DLISAccess
    from ..io.las2 import LAS2Parser
    from ..io.las3 import LAS3Parser
    from ..io.tabr import TABParser
else:
    #from stoneforge.io.dlisio_r import DLISAccess
    from stoneforge.io.dlist import DLISAccess
    from stoneforge.io.las2 import LAS2Parser
    from stoneforge.io.las3 import LAS3Parser
    from stoneforge.io.tabr import TABParser

def _download_to_tempfile(url):
    response = requests.get(url)
    response.raise_for_status()

    suffix = Path(url).suffix  # preserves .las, .dlis, etc.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    tmp.write(response.content)
    tmp.close()

    return tmp.name

class DataLoader:
    
    def __init__(self, filepath, filetype=None, sep="\t", std="US"):
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
        self._tmpfile = None  # track temp file for cleanup
        
        # --- URL handling ---
        if self._is_url(filepath):
            self._tmpfile = _download_to_tempfile(filepath)
            filepath = self._tmpfile

        if filetype == 'las2':
            self.data_obj = LAS2Parser(filepath)

        if filetype == 'las3':
            self.data_obj = LAS3Parser(filepath)

        if filetype == 'dlis':
            self.data_obj = DLISAccess(filepath)

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
                self.data_obj = DLISAccess(filepath)
                print("DLIS parsing successful.")
            elif filext in ['.csv', '.txt', '.dat', '.tsv']:
                try:
                    self.data_obj = TABParser(filepath, sep=sep, std=std)
                except:
                    print("Failed to parse tabular data file.")
            else:
                raise ValueError(f"Unsupported file extension: {filext}")
            
    def _is_url(self, path):
        try:
            result = urlparse(path)
            return result.scheme in ("http", "https")
        except Exception:
            return False
        
    #def _download_to_tempfile(self, url):
    #    response = requests.get(url)
    #    response.raise_for_status()

    #    suffix = Path(url).suffix  # preserves .las, .dlis, etc.
    #    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    #    tmp.write(response.content)
    #    tmp.close()

    #    return tmp.name
            
    def __del__(self):
        if self._tmpfile and os.path.exists(self._tmpfile):
            try:
                os.remove(self._tmpfile)
            except Exception:
                pass
    
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


class DataManager(DataLoader):
    def __init__(self, data_source, depth, filetype=None, sep="\t", std="US"):
        if isinstance(data_source, DataLoader):
            # Copy attributes from the existing DataLoader instance
            self.__dict__.update(data_source.__dict__)
        else:
            super().__init__(data_source, filetype=filetype, sep=sep, std=std)
            
        self.depth_col = depth
        
        if hasattr(self, "data_obj") and hasattr(self.data_obj, "data"):
            self.df, self.units = self.dataframe(self.data_obj.data)
        else:
            raise ValueError("Parsed object has no 'data' attribute.")
            
        min_depth = self.df[self.depth_col].min()
        max_depth = self.df[self.depth_col].max()
        
        self._facies_dict = {
            'unidentified': (min_depth, max_depth),
            'ALL': (min_depth, max_depth)
        }

    def facies(self):
        """
        Returns a list of all added facies names.
        """
        return list(self._facies_dict.keys())

    def add_facies(self, facies_dict):
        """
        Adds multiple facies intervals.
        
        Parameters
        ----------
        facies_dict : dict
            Dictionary mapping facies name to a tuple of (top, bottom) depths.
        """
        for name, (top, bottom) in facies_dict.items():
            self._facies_dict[name] = (top, bottom)
            
    def add_facie(self, name, top, bottom):
        """
        Adds a single facies interval.
        """
        self._facies_dict[name] = (top, bottom)
        
    def __getattr__(self, name):
        if name in self._facies_dict:
            top, bottom = self._facies_dict[name]
            mask = (self.df[self.depth_col] >= top) & (self.df[self.depth_col] <= bottom)
            df_slice = self.df.loc[mask].copy()
            
            def add_log(log_name, unit, values):
                # Update the main DataFrame
                if log_name not in self.df.columns:
                    self.df[log_name] = np.nan
                self.df.loc[mask, log_name] = values
                
                # Update units dictionary
                self.units[log_name] = unit
                
                # Update the original data dictionary (from DataLoader)
                if hasattr(self, "data_obj") and hasattr(self.data_obj, "data"):
                    if log_name not in self.data_obj.data:
                        self.data_obj.data[log_name] = {
                            "values": np.full(len(self.df), np.nan),
                            "unit": unit,
                            "description": "Calculated by DataManager"
                        }
                    
                    import pandas as pd
                    if isinstance(values, (pd.Series, pd.DataFrame)):
                        np_values = values.to_numpy().squeeze()
                    else:
                        np_values = values
                        
                    self.data_obj.data[log_name]["values"][mask.values] = np_values
                    self.data_obj.data[log_name]["unit"] = unit
                    
                # Update the current slice so it's immediately available
                df_slice[log_name] = values
                
            # Bind the add_log method to this specific DataFrame instance
            df_slice.add_log = add_log
            return df_slice
            
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __dir__(self):
        return sorted(set(super().__dir__() + list(self._facies_dict.keys())))
