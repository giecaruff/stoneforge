from __future__ import print_function

import numpy as np
import pandas as pd

if __package__:
    from ..data_management.preprocessing import DataLoader
else:
    from stoneforge.data_management.preprocessing import DataLoader

class TOCProcessor(object):
    """
    Simple Passey TOC processor for one well.

    Inputs:
    - csv_path: path to the laboratory/measurement table
    - las_path: path to the LAS file
    - mnemonics: mapping of logical names to LAS mnemonics
      (depth, dt, logrt, gr, cali, toc)
    - cot_position: column position/index or name for measured COT in the CSV/text file
    - lom: LOM value (default 10.0)
    - dt_baseline: DT baseline (default 100.0)
    - logrt_baseline: log(RT90) baseline (default 0.0)
    """

    def __init__(
        self,
        csv_path,
        las_path,
        mnemonics,
        cot_position=None,
        lom=10.0,
        dt_baseline=100.0,
        logrt_baseline=0.0,
    ):
        self.csv_path = csv_path
        self.las_path = las_path
        self.mnemonics = mnemonics
        self.cot_position = cot_position
        self.lom = lom
        self.dt_baseline = dt_baseline
        self.logrt_baseline = logrt_baseline

        self.lab_df = None
        self.measured_cot = None
        self.log_data = None
        self.result = {}

    def load_lab_data(self):
        """
        Load the laboratory/measurement data from CSV or TXT.
        """
        self.lab_df = pd.read_csv(self.csv_path, sep=None, engine="python")

        if self.cot_position is None:
            if "toc" in self.mnemonics and self.mnemonics["toc"] in self.lab_df.columns:
                column_name = self.mnemonics["toc"]
            else:
                column_name = self.lab_df.columns[0]
        elif isinstance(self.cot_position, int):
            column_name = self.lab_df.columns[self.cot_position]
        elif isinstance(self.cot_position, str):
            column_name = self.cot_position
        else:
            raise TypeError("cot_position must be None, an integer index, or a column name.")

        self.measured_cot = self.lab_df[column_name].astype(float).to_numpy()

        return self.lab_df

    def load_las_data(self):
        """
        Load the LAS curves using lasio.
        """

        data = {}
        for key in ["depth", "dt", "logrt", "gr", "cali"]:
            mnemonic = self.mnemonics.get(key)
            if not mnemonic:
                continue

            las2 = DataLoader(self.las_path, filetype='las2')
            data[key] = las2.data_obj.data[mnemonic]['values']

        self.log_data = data
        return data

    def calculate(self):
        """
        Compute the three requested outputs:
        1) log(RT90) - baseline
        2) -0.02 x (DT - baseline)
        3) calculated COT in %
        """
        self.load_lab_data()
        self.load_las_data()

        depth = np.asarray(self.log_data["depth"], dtype=float)
        dt = np.asarray(self.log_data["dt"], dtype=float)
        logrt = np.asarray(self.log_data["logrt"], dtype=float)

        # Avoid log(0) issues
        logrt = np.where(logrt > 0, np.log10(logrt), np.nan)

        # Keep only finite values
        mask = np.isfinite(depth) & np.isfinite(dt) & np.isfinite(logrt)
        depth = depth[mask]
        dt = dt[mask]
        logrt = logrt[mask]

        logrt_minus_baseline = logrt - self.logrt_baseline
        dt_minus_baseline_scaled = -0.02 * (dt - self.dt_baseline)

        dlogrt = logrt_minus_baseline + dt_minus_baseline_scaled
        calculated_cot = np.clip(
            dlogrt * 10 ** (2.297 - 0.1688 * self.lom),
            0.0,
            100.0,
        )

        self.result = {
            "depth": depth,
            "logrt_minus_baseline": logrt_minus_baseline,
            "dt_minus_baseline_scaled": dt_minus_baseline_scaled,
            "calculated_cot_pct": calculated_cot,
            "measured_cot_pct": self.measured_cot[: len(calculated_cot)] if self.measured_cot is not None else None,
        }

        return self.result

    def run(self):
        return self.calculate()


if __name__ == "__main__":
    mnemonics = {
        "depth": "DEPTH",
        "dt": "DT",
        "logrt": "RT90",
        "gr": "GR",
        "cali": "CAL",
        "toc": "TOC",
    }

    processor = TOCProcessor(
        csv_path=r"C:\Users\mario\Documents\GitHub\stoneforge\tests\tests_total_organic_carbon\toc.csv",
        las_path=r"C:\Users\mario\Documents\massape\7-MP-56D-BA.las",
        mnemonics=mnemonics,
        cot_position=1,  # column index or column name
        lom=10.0,
        dt_baseline=100.0,
        logrt_baseline=0.0,
    )

    result = processor.run()

    print("log(RT90) - baseline:")
    print(result["logrt_minus_baseline"])

    print("\n-0.02x(DT - baseline):")
    print(result["dt_minus_baseline_scaled"])

    print("\nCalculated COT (%):")
    print(result["calculated_cot_pct"])