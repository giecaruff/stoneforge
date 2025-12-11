import pandas as pd
import numpy as np
from collections import Counter
import re

class LAS3Parser:

    def __init__(self, las3_file_path):
        """
        Initialize the LAS3Parser with the path to a LAS3 file.

        Parameters
        ----------
        las3_file_path : str
            The file path to the LAS3 file to be parsed.
        
        Example
        -------
        >>> from stoneforge.io.las3 import LAS3Parser
        >>> parser = LAS3Parser("path/to/file.las")
        """
        self.filepath = las3_file_path
        self.data = self._parse_las3()
        self.tables = list(self.data.keys())

    def force_association(self, to_dict = True, forced=True, sep = "|"):
        """
        Given the parsed data dictionary and a description string, return the associated DataFrame if it exists. This method forces the association even if the description is not found.

        Parameters
        ----------
        to_dict : bool, optional
            If True, convert the final DataFrame to a dictionary of numpy arrays. Default is True.
        forced : bool, optional
            If True, force the association even if the description is not found. Default is True.
        sep : str, optional
            The separator used in the table names to identify associations. Default is "|".
        """
        _data = []
        _desc = []
        _clen = []

        for i, k1 in enumerate(self.tables):
            for j, k2 in enumerate(self.tables):
                if i != j:
                    # clean k2 if it has "|"
                    k2_clean = k2.split(sep)[0].strip()
                    if k1 in k2:
                        _data.append(k2)
                        _desc.append(k1)
                        _clen.append(k2_clean)

        # --- remove all duplicates in _desc completely ---
        counts = Counter(_desc)

        _data_unique = []
        _desc_unique = []
        _clen_unique = []

        for d, desc, c in zip(_data, _desc, _clen):
            if counts[desc] == 1:  # keep only if desc is unique
                _data_unique.append(d)
                _desc_unique.append(desc)
                _clen_unique.append(c)

        for i in range(len(_desc_unique)):
            _name = _clen_unique[i]
            _key = _data_unique[i]
            _val = _desc_unique[i]

            self.data[_name] = self.table_association(_key, _val, forced=forced, to_dict = to_dict)
            self.data.pop(_key)
            self.data.pop(_val)

        self.tables = list(self.data.keys())

    def table_association(self, data, description, forced=True, to_dict = True):
        """
        Given the parsed data dictionary and a description string, return the associated DataFrame if it exists.

        Parameters
        ----------
        data : str
            The key in the data dictionary corresponding to the main data table.
        description : str
            The key in the data dictionary corresponding to the description table.
        forced : bool, optional
            If True, force the association even if the description is not found. Default is True.
        to_dict : bool, optional
            If True, convert the final DataFrame to a dictionary of numpy arrays. Default is True.

        Returns
        -------
        dict
            A dictionary where each key is a mnemonic from the description, and each value is another dictionary with 'values' (numpy array) and 'unit' (string).
        """
        
        if forced:
            try:
                _desc = list(self.data[description][0])
                _mnem = []
                _unit = []

                for item in _desc:
                    # split only on first "."
                    parts = item.split('.', 1)
                    left = parts[0].strip().replace(" ", "")  # remove blanks and dot
                    right = '.' + parts[1].strip() if len(parts) > 1 else ''  # keep dot, strip spaces
                    
                    _mnem.append(left)
                    _unit.append(right)
            except KeyError:
                Warning (f"Description '{description}' not found in data. Using raw mnemonics and units.")
                _mnem = list(_desc = list(self.data[description][0]))
                _unit = list(_desc = list(self.data[description][1]))
        else:
            _mnem = list(_desc = list(self.data[description][0]))
            _unit = list(_desc = list(self.data[description][1]))

        self.data[data].columns = _mnem
        main_data = self.data[data]
        if to_dict:
            main_data = main_data.to_dict(orient="list")

            converted_data = {}
            for key, values in main_data.items():
                try:
                    # Try converting to float
                    converted_data[key] = np.array(values, dtype=float)
                except ValueError:
                    Warning(f"Could not convert column '{key}' to float. Keeping as string array.")
                    # If fails, keep as string array
                    converted_data[key] = np.array(values, dtype=str)

            main_data = converted_data

        overall_data = {}
        for i in range(len(_mnem)):
            _m = _mnem[i]
            _u = _unit[i]
            overall_data[_m] = {'values': main_data[_m], 'unit': _u}

        return overall_data

    def _parse_las3(self):
        "Take the path to a LAS3 file and return a dictionary of DataFrames, one for each section. utf-8 based"
        data_sections = {}
        current_key = None
        current_data_lines = []

        with open(self.filepath, 'r', encoding='utf-8') as file:
            for line in file:
                stripped = line.strip()

                # Skip blank lines and comments
                if not stripped or stripped.startswith('#'):
                    continue

                if stripped.startswith('~'):
                    # Save previous section
                    if current_key and current_data_lines:
                        df = self._parse_data_block(current_data_lines)
                        data_sections[current_key] = df
                        current_data_lines = []

                    current_key = stripped.lstrip('~').strip()
                else:
                    current_data_lines.append(stripped)

            # Add the final section
            if current_key and current_data_lines:
                df = self._parse_data_block(current_data_lines)
                data_sections[current_key] = df

        return data_sections

    def _parse_data_block(self, lines):
        "Try to read the elements of a data block, first with default tab/space/colon logic, then with comma."
        structured_data = []

        # First parse with default (tab/space/colon) logic
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue

            parts = re.split(r'(?:\t| {4,})', line, maxsplit=1)

            if len(parts) == 2:
                key = parts[0].strip()
                rest = parts[1].strip()
                rest_parts = re.split(r'\s*:\s*', rest, maxsplit=1)

                if len(rest_parts) == 2:
                    structured_data.append([key, rest_parts[0].strip(), rest_parts[1].strip()])
                else:
                    structured_data.append([key, rest.strip(), ""])
            else:
                structured_data.append([parts[0].strip(), "", ""])

        # Create initial DataFrame and drop empty columns
        df_original = pd.DataFrame(structured_data)
        df_original = df_original.loc[:, ~df_original.apply(lambda col: col.astype(str).str.strip().eq('').all(), axis=0)]
        n_original_cols = df_original.shape[1]

        # Try parsing with comma separator
        comma_data = [line.split(",") for line in lines if line.strip()]
        df_comma = pd.DataFrame(comma_data)
        df_comma = df_comma.loc[:, ~df_comma.apply(lambda col: col.astype(str).str.strip().eq('').all(), axis=0)]

        # Use the comma-parsed version only if it has more columns
        if df_comma.shape[1] > n_original_cols:
            return df_comma
        else:
            return df_original