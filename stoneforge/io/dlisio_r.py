import numpy as np
from dlisio import dlis  # Correct library import

# ---------------------------------------------------------------------------------------- #
    
def _dict_to_dataframe(data):
    rows = []
    for digital_file, frames in data.items():
        for frame_name, mnemonics in frames.items():
            for mnemonic in mnemonics:  # Each mnemonic gets its own row
                rows.append([digital_file, frame_name, mnemonic])
    df = pd.DataFrame(rows, columns=["Digital File", "Frame Name", "Mnemonics"])
    return df

def _dataframe_to_dict(df):
    data = {}
    for digital_file, frame_group in df.groupby("Digital File"):
        data[digital_file] = {}
        for frame_name, mnemonics_group in frame_group.groupby("Frame Name"):
            data[digital_file][frame_name] = mnemonics_group["Mnemonics"].tolist()
    return data

# ---------------------------------------------------------------------------------------- #

# Function to inspect the structure of a DLIS file
def inspect_dlis_structure(dlis_file, verbose=True):
    with dlis.load(dlis_file) as file:
        if verbose:
            print(f"DLIS File: {dlis_file}")
            print(f"Logical Files Found: {len(file)}")
        logical_info = {}
        for lf in file:
            if verbose:
                print("\n==========================")
                print(f"Logical File: {lf}")
            logical_info[str(lf)] = {}
            
            frames = lf.frames
            if verbose:
                print(f"Frames Found: {len(frames)}")
            for frame in frames:
                logical_info[str(lf)][frame.name] = []
                if verbose:
                    print("\n--------------------------")
                    print(f"Frame Name: {frame.name}")
                    print(f"Channels Found: {len(frame.channels)}")
                for ch in frame.channels:
                    logical_info[str(lf)][frame.name].append(ch.name)
                    if verbose:
                        print(f"   * Mnemonic: {ch.name}, \t Units: {ch.units}, \t Description: {ch.long_name}")
                  
    if not verbose:  
        return logical_info
    
# ---------------------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------------------- #
               
# Extract data from a DLIS file     
def parse_dlis(file_path, data_access):
    extracted_data = {}

    with dlis.load(file_path) as files:
        for f in files:
            logical_file_id = str(f)  # Unique identifier for each logical file
            
            if logical_file_id not in data_access:
                continue  # Skip if the logical file is not in the selection

            extracted_data[logical_file_id] = {}

            simple_frames = {frame.name: frame for frame in f.frames}

            for frame_name, mnemonics in data_access[logical_file_id].items():
                if frame_name not in simple_frames:
                    continue  # Skip if the frame is not found

                frame = simple_frames[frame_name]
                extracted_data[logical_file_id][frame_name] = {}

                for channel in frame.channels:
                    if channel.name in mnemonics:
                        mnemonic = channel.name
                        unit = channel.units
                        values = np.array(channel.curves())  # Convert to numpy array

                        extracted_data[logical_file_id][frame_name][mnemonic] = {
                            'unit': unit,
                            'values': values
                        }

    return extracted_data