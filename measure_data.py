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
        while (int(time.time()) - start_time) <= duration + 1:
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
            raw_data = np.concatenate(current_epoch_data, axis=1)
            # raw_epochs.append(raw_data)
            # Rescale each channel to [-50, 50] range
            rescaled_data = np.zeros_like(raw_data)
            for ch_idx in range(raw_data.shape[0]):
                ch_data = raw_data[ch_idx, :]
                ch_min = ch_data.min()
                ch_max = ch_data.max()
                # Normalize to [0, 1] then scale to [-50, 50]
                if ch_max - ch_min > 0:
                    rescaled_data[ch_idx, :] = ((ch_data - ch_min) / (ch_max - ch_min)) * 100 - 50
                else:
                    rescaled_data[ch_idx, :] = 0
            
            print(f"{rescaled_data.shape = }")
            rescaled_data = rescaled_data[:, :duration * sampling_rate]
            raw_epochs.append(rescaled_data)

    board_shim.release_session()

    # Create info with proper channel names
    ch_names = ['Fp1', 'Fp2', 'Oz', 'T5', 'T6', 'T4', 'T3', 'Cz']
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sampling_rate,
        ch_types=['eeg'] * len(ch_names),
    )
    
    # Set standard 10-20 montage for proper channel locations
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)


    event_samples = np.cumsum(
        [0] + [epoch.shape[1] for epoch in raw_epochs[:-1]]
    ).astype(int)
    events = np.column_stack(
        (
            event_samples,
            np.zeros(len(raw_epochs), dtype=int),
            np.array([1, 2], dtype=int),
        )
    )

    raw_epochs = np.stack(raw_epochs, axis=0)
    epochs = mne.EpochsArray(
        raw_epochs, info, events=events, event_id=event_dict, tmin=0
    )

    file_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    epochs.save(f"measure-{file_id}-epo.fif", overwrite=True)


if __name__ == "__main__":
    main()
