import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
from dlisio import dlis  # Correct library import
import pandas as pd

class DLISAccess:
    def __init__(self, filename, vis=True):
        self.filename = filename
        self.data = None
        self.metadata = None
        
        self.dlis_dict_headers = self._dlis_info(filename, verbose=False)
        self.dlis_dataframe_headers = self._dict_to_dataframe(self.dlis_dict_headers)
        if vis:
            self._preview_data(self.dlis_dataframe_headers)
        
    def get_info(self):
        return self.dlis_dict_headers
        
    def get_data(self):
        selected_rows = [i for i, checked in enumerate(ALL_CHECKBOX_STATES) if checked]
        selected_table = self.dlis_dataframe_headers.iloc[selected_rows]
        dict_data_info = self._dataframe_to_dict(selected_table)
        data = self._parse_dlis(self.filename, dict_data_info)
        return data
        
    def preview(self):
        selected_rows = [i for i, checked in enumerate(ALL_CHECKBOX_STATES) if checked]
        selected_table = self.dlis_dataframe_headers.iloc[selected_rows]
        dict_data_info = self._dataframe_to_dict(selected_table)
        print(f"Selected {len(selected_table)} rows")
        print("Selected rows data:")
        print(dict_data_info)
        print()
        
    # ==================================================================== #
    
    def _dict_to_dataframe(self, data):
        rows = []
        for digital_file, frames in data.items():
            for frame_name, mnemonics in frames.items():
                for mnemonic, details in mnemonics.items():
                    if isinstance(details, dict):
                        unit = details.get('unit', 'N/A')
                        dim = details.get('dim', 'N/A')
                        min_val = details.get('min', 'N/A')
                        max_val = details.get('max', 'N/A')
                        long_name = details.get('long_name', 'N/A')
                    else:
                        unit = 'N/A'
                        dim = 'N/A'

                    rows.append([
                    digital_file,
                    frame_name,
                    mnemonic,
                    unit,
                    dim + 'D',
                    min_val,
                    max_val,
                    long_name
                ])
        return pd.DataFrame(
            rows,
            columns=["Digital File", "Frame Name", "Mnemonics", "Unit", "Dimension", "Min", "Max", "Long Name"]
        )
        
    # ==================================================================== #
    
    def _dataframe_to_dict(self, df):
        data = {}
        for digital_file, frame_group in df.groupby("Digital File"):
            data[digital_file] = {}
            for frame_name, mnemonics_group in frame_group.groupby("Frame Name"):
                data[digital_file][frame_name] = mnemonics_group["Mnemonics"].tolist()
        return data
    
    # ==================================================================== #
        
    def _dlis_info(self, file_path, verbose=False):
        all_data = {}
        logical_info = {}  # For backward compatibility with old structure

        with dlis.load(file_path) as files:
            if verbose:
                print(f"DLIS File: {file_path}")
                print(f"Logical Files Found: {len(files)}")
            
            for f in files:
                logical_file_id = str(f)[12:-1]  # Cleaned logical file ID
                all_data[logical_file_id] = {}
                logical_info[logical_file_id] = {}  # Old-style structure
                
                if verbose:
                    print("\n==========================")
                    print(f"Logical File: {logical_file_id}")
                
                frames = f.frames
                if verbose:
                    print(f"Frames Found: {len(frames)}")
                
                for frame in frames:
                    frame_id = frame.name
                    all_data[logical_file_id][frame_id] = {}
                    logical_info[logical_file_id][frame_id] = []  # Old-style structure
                    
                    if verbose:
                        print("\n--------------------------")
                        print(f"Frame Name: {frame_id}")
                        print(f"Channels Found: {len(frame.channels)}")
                    
                    for channel in frame.channels:
                        mnemonic = channel.name
                        unit = channel.units
                        values = np.array(channel.curves())
                        dim = str(values.ndim)
                        if values.dtype == 'float16' or values.dtype == 'float32' or values.dtype == 'float64':
                            values[values <= -999.] = np.nan
                        min_val = np.nanmin(values)
                        max_val = np.nanmax(values)
                        l_name = channel.long_name
                        
                        # New structure with units and dimensions
                        all_data[logical_file_id][frame_id][mnemonic] = {
                            'unit': unit,
                            'dim': dim,
                            'min': min_val,
                            'max': max_val,
                            'long_name': l_name
                        }
                        
                        # Old-style structure (just mnemonics)
                        logical_info[logical_file_id][frame_id].append(mnemonic)
                        
                        if verbose:
                            print(f"   * Mnemonic: {mnemonic}")
                            print(f"     Units: {unit}")
                            print(f"     Dimension: {dim}D")
                            if hasattr(channel, 'long_name'):
                                print(f"     Description: {channel.long_name}")
        
        if verbose:
            print("\n==========================")
            print("Inspection complete")
        else:
            return all_data
        
    # ==================================================================== #
    
    def _preview_data(self, table, ROWS_PER_PAGE = 30, COLOR_SCHEME = ("#ffd5c2","#ff9868")):
        
        global ALL_CHECKBOX_STATES
        
        LIGHT_COLOR = COLOR_SCHEME[0]
        DARK_COLOR = COLOR_SCHEME[1]

        # Calculate number of pages needed
        total_rows = len(table)
        num_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE  # Ceiling division

        # Create figure with appropriate height
        fig_height = max(6, min(ROWS_PER_PAGE, total_rows) * 0.3)
        fig, ax = plt.subplots(figsize=(15, fig_height))  # Adjust size
        
        # Add title and subtitle
        fig.suptitle("DLIS data Access", fontsize=16, fontweight='bold', y=0.98, x=0.2)
        fig.text(0.35, 0.94, "Structure: digital file | frame | mnemonic (min value | max value) [unit] - dimension | description", ha='center', fontsize=10, style='italic')

        plt.subplots_adjust(left=0.2)  # Make room for the vertical slider
        ax.axis('off')

        # Create vertical slider axis
        global page_slider
        global slider_ax

        slider_ax = plt.axes([0.08, 0.13, 0.03, 0.73])  # (left, bottom, width, height)
        page_slider = Slider(slider_ax, 'Page', valmin = 1, valmax = num_pages, valinit=num_pages, valstep=1, orientation='vertical', color=DARK_COLOR)

        # Create checkbox axis (will be updated)
        checkbox_ax = plt.axes([0.05, 0.1, 0.5, 0.8])  # (left, bottom, width, height)
        checkbox_ax.set_axis_off()

        # Store all checkbox states and labels globally
        ALL_CHECKBOX_STATES = [False] * total_rows
        self.checkbox_labels_all = [
            f"{row[0]} | {row[1]} | {row[2]} ( {row[5]:.2f} | {row[6]:.2f} ) [ {row[3]} ] - {row[4]} | {row[7]}"
            for row in table.values
        ]
        current_checkboxes = None

        def update_checkboxes(page):
            global current_checkboxes
            
            # Clear previous checkboxes
            checkbox_ax.clear()
            checkbox_ax.set_axis_off()
            
            # Calculate current page range (reversed order)
            page_idx = int(num_pages - page)  # This reverses the page order
            start_idx = page_idx * ROWS_PER_PAGE
            end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
            
            # Get current page labels and states
            current_labels = self.checkbox_labels_all[start_idx:end_idx]
            current_states = ALL_CHECKBOX_STATES[start_idx:end_idx]
            
            # Create new checkboxes
            current_checkboxes = CheckButtons(checkbox_ax, current_labels, current_states)
            
            # Apply alternating row colors
            for i, label in enumerate(current_checkboxes.labels):
                if i % 2 == 1:  # Apply light blue to every second row
                    label.set_backgroundcolor(LIGHT_COLOR)
                else:
                    label.set_backgroundcolor('white')  # Keep other rows white
                label.set_color('black')  # Set text color to black for contrast

            def toggle_row(label):
                full_index = self.checkbox_labels_all.index(label)
                ALL_CHECKBOX_STATES[full_index] = not ALL_CHECKBOX_STATES[full_index]
                print(f'{label} is {"checked" if ALL_CHECKBOX_STATES[full_index] else "unchecked"}')

            current_checkboxes.on_clicked(toggle_row)
            plt.draw()

        # Initialize first page
        update_checkboxes(1)

        # Connect slider to update function
        page_slider.on_changed(update_checkboxes)

        plt.show()
    
    # ==================================================================== #
    
    def _parse_dlis(self, file_path, data_access):
        extracted_data = {}

        with dlis.load(file_path) as files:
            for f in files:
                logical_file_id = str(f)[12:-1]  # Unique identifier for each logical file
                
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