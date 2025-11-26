import mne
import scipy.io as sio

rename_dict = {  # channel renaming map: old_name -> new_name
    "0": "FP1",
    "1": "Fp2",
    "2": "Oz",
    "3": "T5",
    "4": "T6",
    "5": "T4",
    "6": "T3",
    "7": "CZ",
    # add as many as you like...
}

epochs = mne.read_epochs("BRIa_measurement-2025-10-31_12:12:22.fif")
info: mne.Info = epochs.info
print(f" ch names: {info['ch_names']}")
epochs.rename_channels(rename_dict)


data = epochs.get_data().reshape(
    len(info["ch_names"]), -1
)  # flatten epochs into continuous
raw = mne.io.RawArray(data, info)
mne.export.export_raw("my_data.set", raw, fmt="eeglab", overwrite=True)


# sio.savemat("my_data.mat", {"data": data, "times": times, "ch_names": ch_names})
