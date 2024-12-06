import mne
import numpy as np
import os
from scipy.signal import detrend
from mne_icalabel import label_components
import matplotlib.pyplot as plt
from utilities import EEGRegionsDivider
from scipy.io import loadmat
import time
from multiprocessing import Pool

def process_epoch(epoch, i, epoch_plot_base_dir):
    """
    Apply ICA to a single epoch, remove components, and save plots.
    If an error occurs, save and return the unmodified epoch.

    Parameters:
    - epoch: Single epoch as an MNE Epochs object.
    - i: Index of the epoch.
    - epoch_plot_base_dir: Directory to save epoch-specific plots.

    Returns:
    - Corrected epoch data (or unmodified if an error occurs).
    """
    print(f"[Epoch {i + 1}] Processing epoch {i + 1}...")

    # Start time tracking
    start_time = time.time()

    # Ensure the epoch is an MNE Epochs object
    if not isinstance(epoch, mne.Epochs):
        raise TypeError(f"[Epoch {i + 1}] Expected mne.Epochs, got {type(epoch)}")

    try:
        # Initialize ICA
        ica = mne.preprocessing.ICA(n_components=0.85, method='fastica', random_state=0, verbose=False)
        print(f"[Epoch {i + 1}] Fitting ICA...")

        # Fit ICA
        ica.fit(epoch)
        print(f"[Epoch {i + 1}] ICA fitting completed.")

        # Identify components to exclude using ICLabel
        ic_labels = label_components(epoch, ica, method="iclabel")
        exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]

        # Apply ICA to remove undesired components
        epoch_corrected = ica.apply(epoch.copy(), exclude=exclude_idx, verbose=False)
        print(f"[Epoch {i + 1}] ICA correction completed.")

    except Exception as e:
        print(f"[Epoch {i + 1}] Error during ICA processing: {e}")
        print(f"[Epoch {i + 1}] Returning unprocessed epoch as fallback.")

        # Save the problematic epoch for later analysis
        error_dir = os.path.join(epoch_plot_base_dir, "errors")
        os.makedirs(error_dir, exist_ok=True)
        error_data_path = os.path.join(error_dir, f"epoch_{i + 1}_data.npy")
        np.save(error_data_path, epoch.get_data())
        print(f"[Epoch {i + 1}] Saved problematic epoch data to: {error_data_path}")

        # Generate a Matplotlib plot of the first 4 channels of the problematic epoch
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        for ch_idx, ax in enumerate(axes):
            if ch_idx < epoch.get_data().shape[1]:  # Ensure we don't exceed available channels
                ax.plot(epoch.get_data()[0, ch_idx, :].T)  # Plot data of the current channel
                ax.set_title(f"Channel {ch_idx + 1}")
                ax.set_ylabel("Amplitude")
        axes[-1].set_xlabel("Time (samples)")
        fig.suptitle(f"Problematic Epoch {i + 1}", fontsize=16)
        # Save the plot
        error_plot_path = os.path.join(error_dir, f"epoch_{i + 1}_plot.png")
        plt.savefig(error_plot_path)
        plt.close(fig)
        print(f"[Epoch {i + 1}] Saved problematic epoch plot to: {error_plot_path}")
        epoch_corrected=epoch
        return epoch_corrected.get_data()

    # Create a directory for epoch-specific plots
    epoch_plot_dir = os.path.join(epoch_plot_base_dir, f"Epoch_{i + 1}")
    os.makedirs(epoch_plot_dir, exist_ok=True)

    # Plot ICA components
    component_figures = mne.viz.plot_ica_components(ica, show=False)
    if isinstance(component_figures, list):
        for j, fig in enumerate(component_figures):
            fig.savefig(os.path.join(epoch_plot_dir, f"ica_components_{j + 1}.png"))
            plt.close(fig)
    elif isinstance(component_figures, plt.Figure):  # Check if it's a single matplotlib figure
        component_figures.savefig(os.path.join(epoch_plot_dir, "ica_components_1.png"))
        plt.close(component_figures)


    fig_sources = ica.plot_sources(epoch, show=False)
    fig_sources_path = os.path.join(epoch_plot_dir, f"ica_sources.png")
    fig_sources.savefig(fig_sources_path)
    plt.close(fig_sources)


    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    print(f"[Epoch {i + 1}] Processing completed in {elapsed_time:.2f} seconds.")

    # Return the corrected data of the epoch
    return epoch_corrected.get_data()


'''L'errore che stai ottenendo è dovuto al fatto che la funzione process_epoch
 è una funzione locale definita all'interno del metodo apply_ica. 
 In Python, le funzioni locali non possono essere "pickle" (ovvero
  non possono essere serializzate), il che è necessario quando
   si utilizzano librerie di parallelismo come multiprocessing.
    Le funzioni locali, come quella che hai definito dentro apply_ica,
     non possono essere trasferite tra i processi.'''


class EEGPreprocessor:
    def __init__(self, raw, data_path, label_path, save_path, run_preprocess=True, run_bad_interpolation=True, only_class=None, only_patient=None):
        """
        Initializes the EEGPreprocessor object with preprocessing parameters.

        Parameters:
        - raw: RawMff object containing PSG data
        - data_path: Path to the original data files
        - save_path: Path where preprocessed files will be saved
        - run_preprocess: flag to run the preprocessing
        - run_bad_interpolation: flag to run bad channel interpolation
        """
        self.raw = raw
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.run_preprocess = run_preprocess
        self.run_bad_interpolation = run_bad_interpolation
        self.ica = None
        self.ic_labels = None
        self.only_class = only_class
        self.only_patient = only_patient

        # Initialize the EEG region divider
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()

    def preprocess(self, file_name_base, overwrite=True):
        if not self.run_preprocess:
            return self.raw

        print("Loading bad channels from .mat file")
        matlab_bad_channels = self.load_matlab_bad_channels()

        print("Identifying bad channels automatically")
        self.automatic_bad_channels(matlab_bad_channels)

        print("Interpolating bad channels")
        self.interpolate_bad_channels()

        print("Applying custom reference")
        self.apply_custom_reference()

        print("Removing trend")
        self.remove_trend()

        print("Filtering data")
        self.filter_data()

        print("Segmenting data into 30-second epochs")
        self.segment_epochs()

        print("Applying ICA")
        self.apply_ica()

        self.save_to_fif(file_name_base, overwrite=overwrite)

        return self.raw

    def load_matlab_bad_channels(self):
        """
        Load visually identified bad channels from a .mat file.
        """
        subject_folder = os.path.join(self.label_path, self.only_class)
        mat_file_path = os.path.join(subject_folder, f"{self.only_patient}.mat")

        if os.path.exists(mat_file_path):
            mat_data = loadmat(mat_file_path)
            if 'badchannelsNdx' in mat_data:
                bad_channels = mat_data['badchannelsNdx'].squeeze()
                if bad_channels.ndim == 0:
                    bad_channels = np.array([bad_channels])
                bad_names = [f"E{int(bad)}" for bad in bad_channels if f"E{int(bad)}" in self.raw.ch_names]
                print(f"Visually identified bad channels: {bad_names}")
                return bad_names
            else:
                print("Field 'badchannelsNdx' not found in the .mat file.")
        else:
            print(f".mat file not found for patient: {mat_file_path}")

        return []

    def automatic_bad_channels(self, matlab_bad_channels=None):
        """
        Identifies bad channels using epoch-based analysis and updates the raw.info['bads'] attribute.
        """
        # Identify bad channels by epochs
        bad_channels_epochs = self.identify_bad_channels()
        print(f"Bad channels by epochs: {bad_channels_epochs}")

        # Combine results from all sources
        combined_bad_channels = set(bad_channels_epochs)
        if matlab_bad_channels:
            combined_bad_channels.update(matlab_bad_channels)
        print(f"Combined bad channels: {combined_bad_channels}")

        # Update bad channels in raw.info
        self.raw.info['bads'] = list(combined_bad_channels)


    def identify_bad_channels(self, epoch_duration=30.0, max_bad_epochs_ratio=0.2):
        """
        Identifies bad channels in the EEG data by analyzing 30-second epochs and detecting anomalies in the signal's
        spectral properties using the Interquartile Range (IQR) method.

        Steps:
        1. Segments the EEG data into 30-second epochs.
        2. Calculates the Power Spectral Density (PSD) for each channel in each epoch.
        3. Computes the mean PSD for each channel across all epochs.
        4. Uses the IQR method to detect outlier PSD values, setting an upper threshold based on the 75th percentile (Q3)
           and IQR.
        5. Checks if the ratio of "bad" epochs (where the PSD exceeds the upper threshold) exceeds a predefined limit
           (`max_bad_epochs_ratio`).
        6. If a channel has too many bad epochs, it is flagged as a bad channel.

        Parameters:
        - epoch_duration: Duration of each epoch in seconds (default is 30.0 seconds).
        - max_bad_epochs_ratio: Maximum allowable ratio of bad epochs for a channel to be considered bad (default is 0.2).

        Returns:
        - List of bad channels that show anomalous spectral activity.
        """
        self.segment_epochs(epoch_duration=epoch_duration)
        epochs = self.epochs
        bad_channels = []

        for ch_idx, ch_name in enumerate(self.raw.ch_names):
            psd_means = []
            for epoch_idx in range(len(epochs)):
                epoch_data = epochs.get_data(item=epoch_idx, picks=[ch_idx])[0]
                psd = mne.time_frequency.psd_array_welch(epoch_data,sfreq=self.raw.info['sfreq'],fmin=0.3,fmax=35,n_fft=int(self.raw.info['sfreq'] * 2), verbose=False)
                psd_mean = psd[0].mean()
                psd_means.append(psd_mean)

            # Compute IQR and threshold
            Q1 = np.percentile(psd_means, 25)
            Q3 = np.percentile(psd_means, 75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR

            # Check bad epochs ratio
            num_bad_epochs = sum([1 for psd_mean in psd_means if psd_mean > upper_bound])
            bad_epochs_ratio = num_bad_epochs / len(epochs)
            if bad_epochs_ratio > max_bad_epochs_ratio:
                bad_channels.append(ch_name)
                print(f"Channel {ch_name} has {num_bad_epochs} bad epochs out of {len(epochs)} ({bad_epochs_ratio:.2f})")

        return bad_channels

    def interpolate_bad_channels(self):
        if self.raw.info['bads'] and self.run_bad_interpolation:
            self.raw.interpolate_bads(method=dict(eeg="spline"), verbose=False)


    def apply_custom_reference(self):
        """
        Apply a common average reference excluding 'Vertex Reference' and set 'Vertex Reference'
        to the negative of the computed average.
        """
        # Find the index of 'Vertex Reference'
        vertex_idx = self.raw.ch_names.index('Vertex Reference') if 'Vertex Reference' in self.raw.ch_names else None

        # Select all EEG channels except 'Vertex Reference'
        picks = mne.pick_types(self.raw.info, eeg=True, exclude=['Vertex Reference'])

        # Calculate the average signal across selected EEG channels
        avg_signal = self.raw.get_data(picks=picks).mean(axis=0)

        # Subtract the calculated average from each EEG channel (except 'Vertex Reference')
        for ch_idx in picks:
            self.raw._data[ch_idx] -= avg_signal

        # Set 'Vertex Reference' to the negative of the average
        if vertex_idx is not None:
            self.raw._data[vertex_idx] = -avg_signal

        return self.raw

    def remove_trend(self):
        data, _ = self.raw[:, :]
        self.raw._data = detrend(data, axis=1)

    def filter_data(self, l_freq=0.30, h_freq=35):
        """
        Apply bandpass filtering to the EEG data and save plots of the filtered signal with separate subplots for each channel.
        """

        self.raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_window='hamming', fir_design='firwin',
                        verbose=False)

        # Create a directory to save plots
        plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(plot_dir, exist_ok=True)

        print("Saving filtered EEG data plot with subplots using matplotlib...")

        # Define channels for visualization and the time range
        channels_to_plot = ['E27', 'E224', 'E59', 'E55', 'E76', 'E116']
        start_time = 60*60+60  # seconds
        duration = 60  # seconds
        fs = int(self.raw.info['sfreq'])  # Sampling frequency

        # Calculate start and end indices
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)

        # Extract data for selected channels and time range
        data, times = self.raw.copy().pick_channels(channels_to_plot).get_data(return_times=True)
        data = data[:, start_idx:end_idx]
        times = times[start_idx:end_idx]

        # Create subplots
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 2 * len(channels_to_plot)), sharex=True)

        for i, ax in enumerate(axes):
            ax.plot(times, data[i] * 1e6)  # Scale signal to µV
            ax.set_title(f"Channel: {channels_to_plot[i]}")
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True)

        # Set the x-label for the last subplot
        axes[-1].set_xlabel("Time (s)")

        # Adjust layout and save the figure
        plt.tight_layout()
        filtered_plot_path = os.path.join(plot_dir, "filtered_data_plot_subplots.png")
        plt.savefig(filtered_plot_path)
        plt.close()

        print(f"Filtered EEG data plot saved to {filtered_plot_path}")

        return self.raw

    def segment_epochs(self, epoch_duration=30.0):
        """Segment the EEG data into fixed-length epochs."""
        events = mne.make_fixed_length_events(self.raw, duration=epoch_duration)
        self.epochs = mne.Epochs(self.raw, events, tmin=0.0, tmax=epoch_duration, baseline=None, preload=True, verbose=False)
        num_epochs = len(self.epochs)
        print(f"Number of epochs created: {num_epochs}")
        return self.epochs

    def apply_ica(self):
        """
        Apply ICA to each epoch individually using parallel processing, generate plots,
        recombine corrected data into a single Raw object, and save results.
        """
        # Set up directories for plots
        patient_plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(patient_plot_dir, exist_ok=True)
        epoch_plot_base_dir = os.path.join(patient_plot_dir, "ICA_individual_epochs")
        os.makedirs(epoch_plot_base_dir, exist_ok=True)


        # Process the central epochs with ICA
        with Pool(processes=None) as pool:
            print("Applying ICA to each epoch using parallel processing...")
            processed_data = pool.starmap(
                process_epoch,
                [(self.epochs[i:i + 1], i, epoch_plot_base_dir)
                 for i in range(len(self.epochs))]
            )

        # Convert processed data to NumPy array
        processed_data = np.concatenate(processed_data, axis=2).squeeze(axis=0)

        # Combine all data (initial, processed, final)
        all_data = processed_data

        # Create a new Raw object with the combined data
        original_info = self.raw.info.copy()
        self.raw = mne.io.RawArray(all_data, original_info)

        # Save concatenated raw EEG data plot
        print("Saving concatenated raw EEG data plot...")
        channels_to_plot = ['E27', 'E224', 'E59', 'E55', 'E76', 'E116']
        start_time = 60 * 60 + 60  # seconds
        duration = 60  # seconds
        fs = int(self.raw.info['sfreq'])  # Sampling frequency
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)

        data, times = self.raw.copy().pick_channels(channels_to_plot).get_data(return_times=True)
        data = data[:, start_idx:end_idx]
        times = times[start_idx:end_idx]

        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 2 * len(channels_to_plot)), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(times, data[i] * 1e6)  # Scale to µV
            ax.set_title(f"Channel: {channels_to_plot[i]}")
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        concatenated_plot_path = os.path.join(patient_plot_dir, "preprocessed_data.png")
        plt.savefig(concatenated_plot_path)
        plt.close()

        print(f"Concatenated raw EEG data plot saved to {concatenated_plot_path}")

        # Return the modified Raw object with the concatenated data
        return self.raw

    def save_to_fif(self, file_name_base, overwrite=True):
        """
        Save the processed raw EEG data to a .fif file.

        Parameters:
        - file_name_base: Base name for the .fif file
        - overwrite: Whether to overwrite an existing file
        """
        # Define the export directory and ensure it exists
        export_dir = os.path.join(self.save_path, "Preprocessed_ICA")
        os.makedirs(export_dir, exist_ok=True)  # Create the full path if it doesn't exist

        # Create the full path for the .fif file
        fif_file_path = os.path.join(export_dir, f"{os.path.basename(file_name_base)}.fif")

        # Attempt to save the file
        try:
            self.raw.save(fif_file_path, overwrite=overwrite)
            print(f"Processed raw EEG data saved to {fif_file_path}")
        except Exception as e:
            print(f"Error saving processed EEG data: {e}")



