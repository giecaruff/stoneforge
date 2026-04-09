from dlisio import dlis
import numpy as np
import pandas as pd
import os

class DLISAccess:
    """Class to access and extract data from DLIS files using dlisio library."""

    def __init__(self, filename):
        """Class to access and parse DLIS files with optional GUI for selecting mnemonics.
        
        Parameters
        ----------
        filename : str
            Path to the DLIS file to be accessed.
        gui : bool, optional
            If True, a GUI with checkboxes will be displayed for selecting mnemonics. Default is True.

        Example
        -------
        >>> from stoneforge.io.dlisio_r import DLISAccess
        >>> dlis_manager = DLISAccess("path/to/dlis_file.dlis") # Initialize checkbox interface
        """

        self.filename = filename
        self.header = self._scan_header()

    # -------------------------------------------------
    # STEP 1 — FAST HEADER SCAN (no curve loading)
    # -------------------------------------------------

    def _scan_header(self):
        """Scan the DLIS file header to extract metadata about frames and channels without loading curve data."""

        rows = []

        with dlis.load(self.filename) as files:

            for f in files:

                logical = str(f)[12:-1]

                for frame in f.frames:

                    frame_name = frame.name

                    for ch in frame.channels:

                        rows.append(
                            (
                                logical,
                                frame_name,
                                ch.name,
                                ch.units,
                                ch.long_name,
                            )
                        )

        return pd.DataFrame(
            rows,
            columns=[
                "logical_file",
                "frame",
                "mnemonic",
                "unit",
                "description",
            ],
        )

    # -------------------------------------------------
    # STEP 2 — SHOW HEADER
    # -------------------------------------------------

    def show_header(self):
        """Display the scanned header information as a DataFrame."""

        return self.header
    
    # -------------------------------------------------
    # STEP 2.1 — SHOW MNEMONICS
    # -------------------------------------------------
    
    def mnemonics(self):
        """Return a sorted list of unique mnemonics available in the DLIS file."""

        if not hasattr(self, "_mnemonics"):

            self._mnemonics = sorted(
                self.header["mnemonic"].unique().tolist()
            )

        return self._mnemonics

    # -------------------------------------------------
    # STEP 3 — EXTRACT SELECTED MNEMONICS
    # -------------------------------------------------

    def extract(self, mnemonics=None):
        """Extract curve data for the specified mnemonics from the DLIS file."""

        if mnemonics is not None:
            mnemonics = set(mnemonics)

        data = {}

        with dlis.load(self.filename) as files:

            for f in files:

                logical = str(f)[12:-1]

                data[logical] = {}

                for frame in f.frames:

                    frame_name = frame.name

                    frame_dict = {}

                    for ch in frame.channels:

                        mnemonic = ch.name

                        if mnemonics and mnemonic not in mnemonics:
                            continue

                        values = np.array(ch.curves())

                        if values.dtype.kind == "f":
                            values[values <= -999] = np.nan

                        unit = ch.units

                        # channel-level container
                        frame_dict.setdefault(mnemonic, {})

                        frame_dict[mnemonic] = {
                            "values": values,
                            "unit": unit,
                        }

                    if frame_dict:
                        data[logical][frame_name] = frame_dict

        return data
    
    def _sanitize(self, name):

        return os.name.replace("/", "_").replace("\\", "_")
    
    def export_csv(self, file_data, output_path="."):

        """
        Export extracted DLIS data into CSV files.

        Structure:

        output_path/
            digital_file/
                frame.csv

        CSV format:

        row 1 -> mnemonics
        row 2 -> units
        row 3..n -> curve values

        Parameters
        ----------
        file_data : dict
            Output from self.extract()

        output_path : str
            Directory where CSV files will be saved
        """

        for digital_file in file_data:

            digital_path = os.path.join(output_path, self._sanitize(digital_file))
            os.makedirs(digital_path, exist_ok=True)


            for frame in file_data[digital_file]:

                frame_dict = file_data[digital_file][frame]

                mnemonics = []
                units = []
                arrays = []


                for mnemonic in frame_dict:

                    mnemonics.append(mnemonic)
                    units.append(frame_dict[mnemonic]["unit"])
                    arrays.append(frame_dict[mnemonic]["data"])


                df = pd.DataFrame({m: a for m, a in zip(mnemonics, arrays)})

                units_row = pd.DataFrame([units], columns=mnemonics)

                df_out = pd.concat([units_row, df], ignore_index=True)


                file_name = self._sanitize(frame) + ".csv"

                df_out.to_csv(
                    os.path.join(digital_path, file_name),
                    index=False
                )
    
#####################################

#data = dl.extract(["MD", "BS", "TVD", "TVDSS","CS","DTCO", "DCAL"])
#dl.export_csv(data, output_path="./output")