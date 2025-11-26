import numpy as np
import mne
import sounddevice as sd
import time
import datetime

from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds


def beep(frequency=440, duration=0.5, samplerate=44100):
    t = np.linspace(0, duration, int(samplerate * duration), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    sd.play(tone, samplerate)
    sd.wait()


def main():
    BoardShim.enable_dev_board_logger()  # enable logger when developing to catch relevant logs
    params = MindRoveInputParams()
    board_id = BoardIds.MINDROVE_WIFI_BOARD
    board_shim = BoardShim(board_id, params)

    board_shim.prepare_session()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    accel_channels = BoardShim.get_accel_channels(board_id)
    board_description = BoardShim.get_board_descr(board_id)
    # eeg_channel_names = BoardShim.get_eeg_names(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    print("Board Description:", board_description)
    print("EEG Channels:", eeg_channels)
    # print("EEG Channel Names:", eeg_channel_names)
    print("Accelerometer Channels:", accel_channels)
    print("Sampling Rate:", sampling_rate)

    window_size = 1  # seconds
    num_points = window_size * sampling_rate

    event_dict = {"open_eye": 1, "closed_eye": 2}
    time_points = [
        (5, 0),
        (20, 1),
        (5, 0),
        (20, 2),
    ]  # (duration in seconds, event code)
    raw_epochs = []

    for duration, event_code in time_points:
        print(f"Starting phase: {event_code} for {duration} seconds.")
        board_shim.start_stream(num_samples=80000)
        current_num_points = 0

        start_time = int(time.time())
        current_epoch_data = []
        while (int(time.time()) - start_time) < duration:
            if board_shim.get_board_data_count() >= (current_num_points + num_points):
                current_num_points = board_shim.get_board_data_count()
                data = board_shim.get_current_board_data(num_points)
                eeg_data = data[
                    eeg_channels
                ]  # Shape: (num_eeg_channels=8, num_points=sampling_rate)
                current_epoch_data.append(eeg_data)
        board_shim.stop_stream()
        beep(800, 0.2)
        if event_code != 0:
            # Concatenate chunks along time axis
            # Shape: (8, total_samples_in_epoch)
            epoch_data = np.concatenate(current_epoch_data, axis=1)
            
            # Rescale each channel to [-50, 50] range
            rescaled_data = np.zeros_like(epoch_data)
            for ch_idx in range(epoch_data.shape[0]):
                ch_data = epoch_data[ch_idx, :]
                ch_min = ch_data.min()
                ch_max = ch_data.max()
                # Normalize to [0, 1] then scale to [-50, 50]
                if ch_max - ch_min > 0:
                    rescaled_data[ch_idx, :] = ((ch_data - ch_min) / (ch_max - ch_min)) * 100 - 50
                else:
                    rescaled_data[ch_idx, :] = 0
            
            raw_epochs.append(rescaled_data)

    board_shim.release_session()
    
    # Concatenate all epochs into continuous data
    # Shape: (8, total_samples_all_epochs)
    all_data = np.concatenate(raw_epochs, axis=1)
    
    # Create info with proper channel names
    ch_names = ['fp1', 'fp2', 'oz', 't5', 't6', 't4', 't3', 'cz']
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sampling_rate,
        ch_types=['eeg'] * len(ch_names),
    )
    
    # Create RawArray for continuous data
    raw = mne.io.RawArray(all_data, info)
    
    # Add events as annotations
    # Shape: (num_epochs,) - sample indices where each epoch starts
    event_samples = np.cumsum([0] + [epoch.shape[1] for epoch in raw_epochs[:-1]]).astype(int)
    # Shape: (num_epochs,) - time in seconds where each epoch starts
    onsets = event_samples / sampling_rate  # Convert to seconds
    # Shape: (num_epochs,) - duration in seconds of each epoch
    durations = [epoch.shape[1] / sampling_rate for epoch in raw_epochs]
    descriptions = ['open_eye', 'closed_eye']
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(annotations)
    
    # Save as raw FIF (use - instead of : for valid filename)
    file_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw.save(f"measure-{file_id}-raw.fif", overwrite=True)
    print(f"Data saved to measure-{file_id}-raw.fif")


if __name__ == "__main__":
    main()
