import re

import numpy as np
import pandas as pd


class TABParser:
    """Read tabular files with a header row, units row, and data rows."""

    def __init__(self, file_path, sep=",", std="US"):
        self.file_path = file_path
        self.data = self._load(file_path, sep=sep, std=std)

    def _load(self, file_path, sep=",", std="US"):
        if std not in {"US", "BR"}:
            raise ValueError("std must be either 'US' or 'BR'")

        df = pd.read_csv(
            file_path,
            sep=sep,
            header=0,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )

        if len(df) < 1:
            raise ValueError(
                f"File '{file_path}' must contain a units row"
            )

        headers = [str(column).strip() for column in df.columns]
        units = [str(value).strip() for value in df.iloc[0]]

        result = {
            header: {
                "unit": unit,
                "values": [],
                "description": "",
            }
            for header, unit in zip(headers, units)
        }

        data = df.iloc[1:]

        for index, header in enumerate(headers):
            values = data.iloc[:, index].map(str.strip).tolist()
            result[header]["values"] = self._convert_values(values, std)

        return result

    @staticmethod
    def _convert_values(values, std):
        if not values:
            return np.array([], dtype=str)

        normalized = []

        for value in values:
            value = value.strip()

            if std == "BR":
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")

            normalized.append(value)

        numeric = pd.Series(
            pd.to_numeric(normalized, errors="coerce")
        )

        if numeric.notna().all():
            if (numeric % 1 == 0).all():
                return numeric.astype(int).to_numpy()

            return numeric.to_numpy()

        return np.array(values, dtype=str)