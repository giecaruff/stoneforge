import warnings
import re
import numpy as np

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

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # --- Step 1: extract header
        header_line = lines[0].strip()
        headers = header_line.split(sep)

        for col in headers:
            col = col.strip()
            name = col.split(" ", 1)[0].strip()
            data_dict[name] = {"unit": "", "values": []}

        # --- Step 2: normalization helper
        def normalize_value(val):
            val = val.strip().strip('"')
            if re.match(r"^-?\d+[.,]?\d*$", val):  # numeric-like
                if std.upper() == "US":
                    val = val.replace(",", "")
                elif std.upper() == "BR":
                    val = val.replace(".", "").replace(",", ".")
            return val

        # --- Step 3: process rows
        for i, line in enumerate(lines[1:], start=2):
            row = line.strip().split(sep)
            if len(row) != len(headers):
                warnings.warn(
                    f"Skipping line {i}: expected {len(headers)} fields, got {len(row)}"
                )
                continue

            # units row
            if i == 2:
                for col_name, value in zip(headers, row):
                    col_name = col_name.split(" ", 1)[0].strip()
                    data_dict[col_name]["unit"] = value.strip().strip('"')
                continue

            # values rows
            for col_name, value in zip(headers, row):
                col_name = col_name.split(" ", 1)[0].strip()
                data_dict[col_name]["values"].append(normalize_value(value))

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