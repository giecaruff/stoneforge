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
    - cot_depth_position: column position/index or name for measured COT depth in the CSV/text file
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
        cot_depth_position=None,
        lom=10.0,
        dt_baseline=100.0,
        logrt_baseline=0.0,
        skip=()
    ):
        self.csv_path = csv_path
        self.las_path = las_path
        self.mnemonics = mnemonics
        self.cot_position = cot_position
        self.cot_depth_position = cot_depth_position
        self.lom = lom
        self.dt_baseline = dt_baseline
        self.logrt_baseline = logrt_baseline
        self.skip = skip

        self.lab_df = None
        self.measured_cot = None
        self.measured_cot_depth = None
        self.log_data = None
        self.result = {}

    def load_lab_data(self):
        """Load the laboratory/measurement data from CSV or TXT, skipping specified row indices.

        :param skip: Tuple or list of row indices (0-indexed) to skip when
        reading.
        """
        # Convert tuple/iterable to list or callable for pandas
        if type(self.skip) == type((0,0)):
            skiprows = list(self.skip) if self.skip else None
        if type(self.skip) == type((0)):
            skiprows = int(self.skip)

        self.lab_df = pd.read_csv(
            self.csv_path, sep=None, engine="python", skiprows=skiprows
        )

        if self.cot_depth_position is None:
            if (
                "depth" in self.mnemonics
                and self.mnemonics["depth"] in self.lab_df.columns
            ):
                depth_column_name = self.mnemonics["depth"]
            else:
                depth_column_name = self.lab_df.columns[0]
        elif isinstance(self.cot_depth_position, int):
            depth_column_name = self.lab_df.columns[self.cot_depth_position]
        elif isinstance(self.cot_depth_position, str):
            depth_column_name = self.cot_depth_position
        else:
            raise TypeError("cot_depth_position must be None, an integer index, or a column name.")

        if self.cot_position is None:
            if (
                "toc" in self.mnemonics
                and self.mnemonics["toc"] in self.lab_df.columns
            ):
                column_name = self.mnemonics["toc"]
            else:
                column_name = self.lab_df.columns[1] if len(self.lab_df.columns) > 1 else self.lab_df.columns[0]
        elif isinstance(self.cot_position, int):
            column_name = self.lab_df.columns[self.cot_position]
        elif isinstance(self.cot_position, str):
            column_name = self.cot_position
        else:
            raise TypeError("cot_position must be None, an integer index, or a column name.")

        self.measured_cot_depth = self.lab_df[depth_column_name].astype(float).to_numpy()
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
        gr = np.asarray(self.log_data["gr"], dtype=float)
        logrt = np.asarray(self.log_data["logrt"], dtype=float)
        cali = np.asarray(self.log_data["cali"], dtype=float)

        # Avoid log(0) issues
        logrt = np.where(logrt > 0, np.log10(logrt), np.nan)

        # Keep only finite values
        mask = np.isfinite(depth) & np.isfinite(dt) & np.isfinite(logrt)
        depth = depth[mask]
        dt = dt[mask]
        gr = gr[mask]
        logrt = logrt[mask]
        cali= cali[mask]

        def passeymethod(dt, logrt, dtbaseline, logrtbaseline, lom):
                    dlogrt = (logrt - logrtbaseline) + 0.02*(dt - dtbaseline)
                    toc = dlogrt*10**(2.297 - 0.1688*lom)
                    return np.clip(toc, 0.0, 100.0)

        # To remove in future
        logrt_minus_baseline = logrt - self.logrt_baseline
        dt_minus_baseline_scaled = -0.02 * (dt - self.dt_baseline)

        #dlogrt = logrt_minus_baseline + dt_minus_baseline_scaled
        #calculated_cot = np.clip(
        #    dlogrt * 10 ** (2.297 - 0.1688 * self.lom),
        #    0.0,
        #    100.0,
        #)
        calculated_cot = passeymethod(dt, logrt, self.dt_baseline, self.logrt_baseline, self.lom)


        measured_cot_aligned = None
        if self.measured_cot is not None:
            if self.measured_cot_depth is not None and len(self.measured_cot_depth) == len(self.measured_cot):
                valid_lab = np.isfinite(self.measured_cot_depth) & np.isfinite(self.measured_cot)
                if np.any(valid_lab):
                    lab_depth = self.measured_cot_depth[valid_lab].astype(float)
                    lab_cot = self.measured_cot[valid_lab].astype(float)

                    if lab_depth.size > 0:
                        order = np.argsort(lab_depth)
                        lab_depth = lab_depth[order]
                        lab_cot = lab_cot[order]
                        measured_cot_aligned = np.interp(
                            depth,
                            lab_depth,
                            lab_cot,
                            left=np.nan,
                            right=np.nan,
                        )
            else:
                measured_cot_aligned = self.measured_cot[: len(calculated_cot)]

        self.result = { # ["depth", "dt", "logrt", "gr", "cali"]
            "depth": depth,
            "dt": dt,
            "gr":gr,
            "logrt": logrt,
            "cali": cali,
            "dt_baseline": np.array([self.dt_baseline]*len(depth)),
            "rt_baseline": np.array([self.logrt_baseline]*len(depth)),
            "logrt_minus_baseline": logrt_minus_baseline,
            "dt_minus_baseline_scaled": dt_minus_baseline_scaled,
            "calculated_cot_pct": calculated_cot,
            "measured_cot_pct": self.measured_cot[: len(calculated_cot)] if self.measured_cot is not None else None,
            "measured_cot_depth": self.measured_cot_depth,
            "measured_cot_pct_aligned": measured_cot_aligned,
        }

        return self.result

    def run(self):
        return self.calculate()