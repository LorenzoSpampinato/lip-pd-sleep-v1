import mne
import numpy as np
from preprocess.preprocess import preprocess
from lidpd_main import get_args
import matplotlib.pyplot as plt

# Define the file path
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Load the file with preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)
raw.pick_channels(['E8', 'E9', 'E10', 'E11'])

# Store the original raw data before filtering and preprocessing
original_raw_data = raw.copy()

# Apply filtering to the raw signal for channel E8
raw_e8 = raw.copy().pick_channels(['E8'])
raw_filtered_e8 = raw_e8.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

# Apply preprocessing (detrending, filtering, etc.)
args = get_args()
processed_raw = preprocess(raw, args.run_preprocess, args.run_bad_interpolation)

# Set up the time for visualization
start_time = 0  # Start of visualization (you can change this value)
duration = 10   # Duration of visualization in seconds
sfreq = original_raw_data.info['sfreq']  # Sampling frequency
n_samples = int(duration * sfreq)  # Number of samples to plot

# Extract the time vector and data for channel 'E8' from original, filtered, and processed_raw
times = original_raw_data.times[:n_samples]  # Time vector for the first 'duration' seconds
raw_data_e8 = original_raw_data.get_data(picks='E8')[0, :n_samples]  # Raw data for channel 'E8'
filtered_data_e8 = raw_filtered_e8.get_data()[0, :n_samples]  # Filtered data for channel 'E8'
processed_data_e8 = processed_raw.get_data(picks='E8')[0, :n_samples]  # Processed data for channel 'E8'

# Create a figure for subplots
plt.figure(figsize=(10, 8))

# Subplot 1: Original Raw Data
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
plt.plot(times, raw_data_e8, color='blue')
plt.title('Raw Data - Canale E8')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.xlim(start_time, start_time + duration)
plt.grid()

# Subplot 2: Filtered Data
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
plt.plot(times, filtered_data_e8, color='green')
plt.title('Filtered Data - Canale E8')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.xlim(start_time, start_time + duration)
plt.grid()

# Subplot 3: Processed Data
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
plt.plot(times, processed_data_e8, color='red')
plt.title('Processed Data - Canale E8')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.xlim(start_time, start_time + duration)
plt.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

# Check if the raw and processed signals are equal
are_signals_equal = np.array_equal(raw_data_e8, processed_data_e8)
if are_signals_equal:
    print("I segnali pre e post elaborazione sono uguali.")
else:
    print("I segnali pre e post elaborazione sono diversi.")
