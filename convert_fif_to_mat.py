import mne
import scipy.io as sio
import numpy as np
import os
import pathlib

# --- Configuration ---
# 1. Replace with your actual file path
fif_file_path = pathlib.Path("measure-2025-11-26_12-27-16-raw.fif") 

# --- Load the .fif file ---
if not os.path.exists(fif_file_path):
    print(f"Error: File not found at {fif_file_path}")
else:
    # Read the raw data (use read_raw_fif for continuous data, read_epochs_fif for epoched data)
    raw = mne.io.read_raw_fif(fif_file_path, preload=True, verbose='error') 

    # --- Data Extraction ---
    
    # 1. Extract the data array
    # MNE data is typically (channels x time) or (channels x samples x trials)
    data_matrix = raw.get_data() 
    
    # 2. Extract the sampling rate
    srate = raw.info['sfreq']
    
    # 3. Extract the channel names
    # raw.ch_names returns a list of strings
    channel_names = raw.ch_names 
    
    # --- Prepare for .mat file saving ---
    
    # MATLAB uses a different format for cell arrays/strings. 
    # Convert the Python list of strings to a NumPy object array for MATLAB compatibility.
    channel_names_np = np.array(channel_names, dtype=object)

    # Dictionary containing all data needed for EEGLAB
    mat_data = {
        'data': data_matrix,          # EEG data (MNE format: Channels x Time/Samples [x Trials])
        'srate': srate,               # Sampling rate
        'chan_names': channel_names_np # Channel labels (crucial for locations)
    }

    # Save to .mat file
    math_output_path = fif_file_path.with_suffix(".mat")
    sio.savemat(math_output_path, mat_data)
    
    print(f"Successfully converted and saved data to {math_output_path}")