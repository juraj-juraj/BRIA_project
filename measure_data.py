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
                ]  # output of shape (num_eeg_channels, num_of_samples)
                current_epoch_data.append(eeg_data)
        board_shim.stop_stream()
        beep(800, 0.2)
        if event_code != 0:
            raw_epochs.append(np.concatenate(current_epoch_data, axis=1))

    board_shim.release_session()
    info = mne.create_info(
        len(eeg_channels),
        sfreq=sampling_rate,
        ch_types=["eeg"] * len(eeg_channels),
    )
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
    # epochs.plot(events=events, event_id=event_dict, scalings="auto", show=True)
    file_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    epochs.save(f"measure-{file_id}.fif", overwrite=True)


if __name__ == "__main__":
    main()
