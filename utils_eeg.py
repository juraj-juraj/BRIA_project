import mne

def load_data(file_name: str, use_fif: bool=True) -> mne.Epochs:
    if use_fif:
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.set_eeg_reference("average")
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            raw,                # The Raw data object
            events,             # The events array derived from annotations
            event_id=event_id,  # The ID mapping derived from annotations
            # tmin=T_MIN,         # Start time of the epoch
            # tmax=T_MAX,         # End time of the epoch
            preload=True,       # Load all epoched data into memory
        )
        del raw
    else:
        epochs = mne.read_epochs(file_name, preload=True)
    return epochs
