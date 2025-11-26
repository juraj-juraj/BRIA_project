from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds

BoardShim.enable_dev_board_logger() # enable logger when developing to catch relevant logs
params = MindRoveInputParams()
board_id = BoardIds.MINDROVE_WIFI_BOARD
board_shim = BoardShim(board_id, params)

board_shim.prepare_session()
board_shim.start_stream()

eeg_channels = BoardShim.get_eeg_channels(board_id)
accel_channels = BoardShim.get_accel_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)

window_size = 2 # seconds
num_points = window_size * sampling_rate

while True:
  if board_shim.get_board_data_count() >= num_points:
    data = board_shim.get_current_board_data(num_points) # Note that using this command, there will be an overlap in the data obtained in the current cycle and the previous cycle(s)
    eeg_data = data[eeg_channels] # output of shape (8, num_of_samples) ## Beware that depending on the electrode configuration, some channels can be *inactive*, resulting in all-zero data for that particular channel
    accel_data = data[accel_channels] # output of shape (3, num_of_samples)
    # process data, or print it out
    print(eeg_data)
