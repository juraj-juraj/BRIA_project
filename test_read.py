import mne

# === Load EEGLAB dataset (.set) ===
raw = mne.io.read_raw_eeglab("my_data.set", preload=True)

# === Plot a short time window (for example, 10 seconds) ===
fig = raw.plot(start=0, duration=10, scalings="auto", show=True)

# === Save the plot to file ===
fig.savefig("raw_plot.png", dpi=300, bbox_inches="tight")

print("Channel names:", raw.ch_names)
print("Sampling rate:", raw.info["sfreq"])

# === Check for events ===
events, event_id = mne.events_from_annotations(raw)
print("Found events:", events[:5])
print("Event IDs:", event_id)

# === Create epochs ===
# Adjust tmin/tmax according to your desired window around the event (in seconds)
epochs = mne.Epochs(
    raw,
    events=events,
    event_id=event_id,
    tmin=-0.2,  # 200 ms before event
    tmax=0.8,  # 800 ms after event
    baseline=(None, 0),
    preload=True,
)

print(epochs)
print("Epoch shape:", epochs.get_data().shape)
