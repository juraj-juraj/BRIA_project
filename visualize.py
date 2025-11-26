import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# -----------------------------------------------------
# 1. Load epochs
# -----------------------------------------------------hs.plot(scalings="auto", show=True)
# Ensure the matplotlib event loop runs and block until the user closes the window
epochs = mne.read_epochs("BRIa_measurement-2025-11-05_20:19:21.fif", preload=True)
epochs.plot(scalings="auto", show=True)
# Example mapping (adjust if your file already stores event_id)
event_dict = {"open_eye": 1, "closed_eye": 2}

# -----------------------------------------------------
# 2. Keep only desired event types
# -----------------------------------------------------
epochs = epochs["open_eye", "closed_eye"]


# -----------------------------------------------------
# 4. Define frequency bands
# -----------------------------------------------------
bands = {
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

sfreq = epochs.info["sfreq"]
ch_names = epochs.info["ch_names"]
data = epochs.get_data()  # shape = (n_epochs, n_channels, n_times)

# Apply a 0.5–40 Hz bandpass to all epochs/channels (time axis = 2)
# l_freq, h_freq = 0.5, 40.0
# data = mne.filter.filter_data(data, sfreq, l_freq, h_freq, axis=2, fir_design='firwin')

# -----------------------------------------------------
# 5. Loop through epochs individually
# -----------------------------------------------------
for ep_idx, ep_data in enumerate(data):
    ep_event = epochs.events[ep_idx, 2]
    event_name = [k for k, v in event_dict.items() if v == ep_event][0]

    print(f"Plotting epoch {ep_idx+1}/{len(data)} ({event_name})")

    # ---- Spectrogram per channel ----
    fig_spec, axs_spec = plt.subplots(
        len(ch_names), 1, figsize=(10, 2 * len(ch_names)), constrained_layout=True
    )

    for i, ch in enumerate(ch_names):
        f, t, Sxx = spectrogram(
            ep_data[i], fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq)
        )
        axs_spec[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
        axs_spec[i].set_title(f"{event_name} - {ch}")
        axs_spec[i].set_ylabel("Freq [Hz]")
        axs_spec[i].set_xlabel("Time [s]")
        axs_spec[i].set_ylim(0, 100)
    fig_spec.suptitle(f"Spectrograms - Epoch {ep_idx+1} ({event_name})")
    plt.show()

    # ---- Band power over time ----
    fig_band, axs_band = plt.subplots(
        len(ch_names), 1, figsize=(10, 2 * len(ch_names)), constrained_layout=True
    )

    for i, ch in enumerate(ch_names):
        f, t, Sxx = spectrogram(
            ep_data[i], fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq)
        )
        band_powers = {}
        for band, (fmin, fmax) in bands.items():
            mask = (f >= fmin) & (f <= fmax)
            band_powers[band] = np.mean(Sxx[mask, :], axis=0)

        axs_band[i].plot(t, band_powers["alpha"], label="Alpha (8–13 Hz)")
        axs_band[i].plot(t, band_powers["beta"], label="Beta (13–30 Hz)")
        axs_band[i].plot(t, band_powers["gamma"], label="Gamma (30–45 Hz)")
        axs_band[i].set_title(f"{event_name} - {ch}")
        axs_band[i].set_xlabel("Time [s]")
        axs_band[i].set_ylabel("Power")
        axs_band[i].legend(loc="upper right")

    fig_band.suptitle(f"Band Power - Epoch {ep_idx+1} ({event_name})")
    plt.show()
