import os
import re
import warnings

import numpy as np
import pandas as pd

# Custom formatting for warnings
def clean_formatwarning(message, category, filename, lineno, line=None):
    # Get only the filename (no user path)
    short_filename = os.path.basename(filename)
    return f"{category.__name__}: {message} (in {short_filename}:{lineno})\n"

warnings.formatwarning = clean_formatwarning

class TABParser:
    def __init__(self, file_path, sep=",", std="US"):
        """
        Initializes the TabularDataLoader with the given file path, separator, and numeric format.
        
        Args:
            file_path (str): Path to the CSV/TSV file.
            sep (str): Field separator (default is ",").
            std (str): Numeric formatting standard, either "US" [standard one] (1,234.56) or "BR" (1.234,56).
        
        Returns:
            None
        """
        self.file_path = file_path
        self.data = self._load_csv_as_dict(file_path = file_path, sep=sep, std=std)

    def _load_csv_as_dict(self, file_path, sep=",", std="US"):
        """
        Reads a CSV/TSV file line by line, robust against malformed rows.

        Args:
            file_path (str): path to file
            sep (str): field separator (default ",")
            std (str): "US" (1,234.56) or "BR" (1.234,56) numeric formatting

        Returns:
            dict: { column_name: {"unit": str, "values": np.ndarray or list[str]} }
        """
        data_dict = {}

        df = pd.read_csv(file_path, sep=sep, encoding="utf-8", dtype=str, skip_blank_lines=True)
        df = df.fillna("")

        if df.shape[0] < 1:
            raise ValueError(f"CSV file '{file_path}' does not contain enough rows")

        headers = [str(col).strip() for col in df.columns]
        for col in headers:
            name = col.split(" ", 1)[0].strip()
            data_dict[name] = {"unit": "", "values": []}

        if df.shape[0] >= 2:
            units_row = df.iloc[0]
            for col_name, value in zip(headers, units_row):
                col_name = col_name.split(" ", 1)[0].strip()
                data_dict[col_name]["unit"] = str(value).strip().strip('"')

        for i in range(1, df.shape[0]):
            row = df.iloc[i]
            if len(row) != len(headers):
                warnings.warn(
                    f"Skipping line {i + 1}: expected {len(headers)} fields, got {len(row)}"
                )
                continue

            for col_name, value in zip(headers, row):
                col_name = col_name.split(" ", 1)[0].strip()
                data_dict[col_name]["values"].append(str(value).strip().strip('"'))

        # --- Step 4: convert lists to numpy arrays
        for col, content in data_dict.items():
            vals = content["values"]
            try:
                arr = np.array(vals, dtype=int)
                data_dict[col]["values"] = arr
            except ValueError:
                try:
                    arr = np.array(vals, dtype=float)
                    data_dict[col]["values"] = arr
                except ValueError:
                    data_dict[col]["values"] = np.array(vals, dtype=str)

        ordered_data_dict = {}
        for k in data_dict.keys():
            ordered_data_dict[k] = {'values' : data_dict[k]['values'], 'unit' : data_dict[k]['unit'], 'description' : ''}

        return ordered_data_dict