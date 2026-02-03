
import os
import pandas as pd
import warnings
from urllib.parse import urlparse
import requests
import tempfile
from pathlib import Path

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
        self._tmpfile = None  # track temp file for cleanup
        
        # --- URL handling ---
        if self._is_url(filepath):
            self._tmpfile = self._download_to_tempfile(filepath)
            filepath = self._tmpfile

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
            
    def _is_url(self, path):
        try:
            result = urlparse(path)
            return result.scheme in ("http", "https")
        except Exception:
            return False
        
    def _download_to_tempfile(self, url):
        response = requests.get(url)
        response.raise_for_status()

        suffix = Path(url).suffix  # preserves .las, .dlis, etc.
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

        tmp.write(response.content)
        tmp.close()

        return tmp.name
            
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