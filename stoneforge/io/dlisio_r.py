import numpy as np
from dlisio import dlis  # Correct library import

# Function to inspect the structure of a DLIS file
def inspect_dlis_structure(dlis_file):
    with dlis.load(dlis_file) as file:
        print(f"Logical Files Found: {len(file)}")
        for lf in file:
            print("\n==========================")
            print(f"Logical File: {lf}")
            
            # List all frames and their respective channels
            frames = lf.frames
            print(f"Total Frames: {len(frames)}")
            for frame in frames:
                print(f" - Frame Name: {frame.name}, Channels: {len(frame.channels)}")
                for ch in frame.channels:
                    print(f"   * Mnemonic: {ch.name}, \t Units: {ch.units}, \t Description: {ch.long_name}")
               
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