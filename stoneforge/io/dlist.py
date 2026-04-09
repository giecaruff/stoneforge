import numpy as np
import pandas as pd
import os

from dlispy import parse

class DLISAccess:


    def __init__(self, filename):

        self.filename = filename

        _, self.logical_files = parse(filename, eflr_only=False)

        self.header = self._scan_header()


    # -------------------------------------------------
    # FAST HEADER SCAN
    # -------------------------------------------------

    def _scan_header(self):

        rows = []

        for lf in self.logical_files:

            logical_id = lf.id.strip()

            for frame_key, frame in lf.simpleFrames.items():

                frame_name = frame_key.identifier

                for jj, ch in enumerate(frame.ChannelNames):

                    mnemonic = ch.identifier
                    unit = frame.Channels[jj].Units

                    rows.append(
                        (
                            logical_id,
                            frame_name,
                            mnemonic,
                            unit
                        )
                    )

        return pd.DataFrame(
            rows,
            columns=[
                "logical_file",
                "frame",
                "mnemonic",
                "unit"
            ]
        )


    # -------------------------------------------------
    # HEADER DISPLAY
    # -------------------------------------------------

    def show_header(self):

        return self.header


    # -------------------------------------------------
    # UNIQUE MNEMONICS
    # -------------------------------------------------

    def mnemonics(self):

        if not hasattr(self, "_mnemonics"):

            self._mnemonics = sorted(
                self.header["mnemonic"].unique().tolist()
            )

        return self._mnemonics


    # -------------------------------------------------
    # HIERARCHICAL DATA EXTRACTION
    # -------------------------------------------------

    def extract(self, mnemonics=None):

        if mnemonics is not None:

            mnemonics = set(mnemonics)

        data = {}

        for lf in self.logical_files:

            logical_id = lf.id.strip()

            data[logical_id] = {}

            simple_frames_keys = list(lf.simpleFrames.keys())

            ii = 0

            for frame_name, frame_data_list in lf.frameDataDict.items():

                sfk = simple_frames_keys[ii]

                frame_id = (
                    str(frame_name.identifier)
                    if hasattr(frame_name, "identifier")
                    else str(frame_name)
                )

                frame_dict = {}

                channel_names = lf.simpleFrames[sfk].ChannelNames
                channel_units = lf.simpleFrames[sfk].Channels


                for jj in range(len(channel_names)):

                    mnemonic = channel_names[jj].identifier

                    if mnemonics and mnemonic not in mnemonics:

                        continue


                    unit = channel_units[jj].Units


                    values = np.array(
                        [
                            fd.slots[jj]
                            for fd in frame_data_list
                        ]
                    )


                    if values.dtype.kind == "f":

                        values[values <= -999] = np.nan


                    frame_dict[mnemonic] = {

                        "values": values,
                        "unit": unit
                    }


                if frame_dict:

                    data[logical_id][frame_id] = frame_dict


                ii += 1


        return data


    # -------------------------------------------------
    # CSV EXPORT (same structure as dlisio version)
    # -------------------------------------------------

    def export_csv(self, file_data, output_path="."):

        for digital_file in file_data:

            digital_path = os.path.join(
                output_path,
                self._sanitize(digital_file)
            )

            os.makedirs(digital_path, exist_ok=True)


            for frame in file_data[digital_file]:

                frame_dict = file_data[digital_file][frame]


                mnemonics = []
                units = []
                arrays = []


                for mnemonic in frame_dict:

                    mnemonics.append(mnemonic)

                    units.append(
                        frame_dict[mnemonic]["unit"]
                    )

                    arrays.append(
                        frame_dict[mnemonic]["data"]
                    )


                if "DEPTH" in mnemonics:

                    depth_index = mnemonics.index("DEPTH")

                    mnemonics.insert(
                        0,
                        mnemonics.pop(depth_index)
                    )

                    units.insert(
                        0,
                        units.pop(depth_index)
                    )

                    arrays.insert(
                        0,
                        arrays.pop(depth_index)
                    )


                df = pd.DataFrame(
                    {m: a for m, a in zip(mnemonics, arrays)}
                )


                units_row = pd.DataFrame(
                    [units],
                    columns=mnemonics
                )


                df_out = pd.concat(
                    [units_row, df],
                    ignore_index=True
                )


                file_name = self._sanitize(frame) + ".csv"


                df_out.to_csv(
                    os.path.join(
                        digital_path,
                        file_name
                    ),
                    index=False
                )


    # -------------------------------------------------
    # SAFE FILENAMES
    # -------------------------------------------------

    def _sanitize(self, name):

        return name.replace("/", "_").replace("\\", "_")