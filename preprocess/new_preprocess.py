import mne
import numpy as np
import os
from scipy.signal import detrend
from mne_icalabel import label_components
import matplotlib.pyplot as plt
from utilities import EEGRegionsDivider
import time
from multiprocessing import Pool
from scipy.interpolate import CubicSpline

def process_epoch(epoch, i, epoch_plot_base_dir):
    """
    Apply ICA to a single epoch, remove components, and save plots.
    If an error occurs, retry ICA with n_components=5 and exclude non-brain components.
    Save plots of data before and after correction.

    Parameters:
    - epoch: Single epoch as an MNE Epochs object.
    - i: Index of the epoch.
    - epoch_plot_base_dir: Directory to save epoch-specific plots.

    Returns:
    - Corrected epoch data (or unmodified if an error occurs).
    """
    print(f"[Epoch {i}] Processing epoch {i}...")

    # Start time tracking
    start_time = time.time()

    # Ensure the epoch is an MNE Epochs object
    if not isinstance(epoch, mne.Epochs):
        raise TypeError(f"[Epoch {i}] Expected mne.Epochs, got {type(epoch)}")

    # Define channels to plot
    channels_to_plot = ['E36', 'E224', 'E59', 'E183', 'E116']

    def plot_and_save(data, title, save_path, channels_to_plot):
        """
        Helper function to plot and save raw data with time in seconds and amplitude in microvolts.

        Parameters:
        - data: MNE Epochs object containing the data to plot.
        - title: Title for the plot.
        - save_path: Path where the plot will be saved.
        - channels_to_plot: List of channel names to plot.
        """
        # Get the time array in seconds
        time = data.times  # Directly get time in seconds

        # Create the figure and axes
        fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 8), sharex=True)

        for ch_idx, ax in enumerate(axes):
            ch_name = channels_to_plot[ch_idx]
            if ch_name in data.info['ch_names']:
                # Get the data for the specific channel
                ch_idx_in_data = data.info['ch_names'].index(ch_name)
                ch_data = data.get_data()[:, ch_idx_in_data, :]  # Shape: (n_epochs, n_channels, n_times)

                # Convert data from volts to microvolts
                ch_data_in_microvolts = ch_data * 1e6

                # Plot the first epoch's data
                ax.plot(time, ch_data_in_microvolts[0].T)
                ax.set_title(f"Channel {ch_name}")
                ax.set_ylabel("Amplitude (µV)")  # Update unit to microvolts

        # Set x-axis label for the last subplot
        axes[-1].set_xlabel("Time (s)")

        # Add overall title and save the figure
        fig.suptitle(title, fontsize=16)
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to: {save_path}")

    def save_ica_plots(ica, epoch, save_dir):
        """
        Helper function to plot and save ICA components and sources.
        """
        # Plot ICA components
        component_figures = mne.viz.plot_ica_components(ica, show=False)
        if isinstance(component_figures, list):
            for j, fig in enumerate(component_figures):
                fig.savefig(os.path.join(save_dir, f"ica_components_{j + 1}.png"))
                plt.close(fig)
        elif isinstance(component_figures, plt.Figure):
            component_figures.savefig(os.path.join(save_dir, "ica_components_1.png"))
            plt.close(component_figures)

        # Plot ICA sources
        fig_sources = ica.plot_sources(epoch, show=False)
        fig_sources_path = os.path.join(save_dir, f"ica_sources.png")
        fig_sources.savefig(fig_sources_path)
        plt.close(fig_sources)
        print(f"Saved ICA sources plot to: {fig_sources_path}")

    try:
        # Create a directory for epoch-specific plots
        epoch_plot_dir = os.path.join(epoch_plot_base_dir, f"Epoch_{i}")
        os.makedirs(epoch_plot_dir, exist_ok=True)

        # Plot and save raw signal before ICA correction
        before_plot_path = os.path.join(epoch_plot_dir, f"epoch_{i}_before_ica.png")
        plot_and_save(epoch, f"Epoch {i} (Before ICA)", before_plot_path, channels_to_plot)

        # Initialize ICA
        ica = mne.preprocessing.ICA(n_components=0.85, method='fastica', random_state=0, verbose=False)
        print(f"[Epoch {i}] Fitting ICA...")

        # Fit ICA
        ica.fit(epoch)
        print(f"[Epoch {i}] ICA fitting completed.")

        # Identify components to exclude using ICLabel
        ic_labels = label_components(epoch, ica, method="iclabel")
        labels = ic_labels["labels"]
        print(f"[Epoch {i}] ICLabel Component Labels: {labels}")
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]

        # Apply ICA to remove undesired components
        epoch_corrected = ica.apply(epoch.copy(), exclude=exclude_idx, verbose=False)
        print(f"[Epoch {i}] ICA correction completed.")

        # Plot and save raw signal after ICA correction
        after_plot_path = os.path.join(epoch_plot_dir, f"epoch_{i}_after_ica.png")
        plot_and_save(epoch_corrected, f"Epoch {i} (After ICA)", after_plot_path, channels_to_plot)

        # Save ICA plots
        save_ica_plots(ica, epoch, epoch_plot_dir)

    except Exception as e:
        print(f"[Epoch {i}] Error during ICA processing: {e}")
        print(f"[Epoch {i}] Retrying ICA with n_components=6...")

        try:

            # Create the error directory
            error_dir = os.path.join(epoch_plot_base_dir, "errors", f"Epoch_{i}")
            os.makedirs(error_dir, exist_ok=True)

            # Plot and save raw signal before ICA correction
            retry_before_path = os.path.join(error_dir, f"epoch_{i}_before_ica.png")
            plot_and_save(epoch, f"Epoch {i} (Before Retry ICA)", retry_before_path, channels_to_plot)

            # Retry with n_components=6 and exclude non-brain components
            ica = mne.preprocessing.ICA(n_components=6, method='fastica', random_state=0, verbose=False)
            ica.fit(epoch)
            print(f"[Epoch {i}] ICA fitting with n_components=6 completed.")

            # Identify components to exclude using ICLabel
            ic_labels = label_components(epoch, ica, method="iclabel")
            labels = ic_labels["labels"]
            print(f"[Epoch {i}] ICLabel Component Labels (Retry): {labels}")
            exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain"]]

            # Apply ICA to remove undesired components
            epoch_corrected = ica.apply(epoch.copy(), exclude=exclude_idx, verbose=False)
            print(f"[Epoch {i}] ICA correction with n_components=6 completed.")

            # Plot and save raw signal before ICA correction
            retry_after_path = os.path.join(error_dir, f"epoch_{i}_after_ica.png")
            plot_and_save(epoch_corrected, f"Epoch {i} (After Retry ICA)", retry_after_path, channels_to_plot)

            # Save ICA plots in error directory
            save_ica_plots(ica, epoch, error_dir)

        except Exception as fallback_error:
            print(f"[Epoch {i}] Retry ICA also failed: {fallback_error}")
            # Save the raw epoch data for troubleshooting
            error_data_path = os.path.join(error_dir, f"epoch_{i}_data.npy")
            np.save(error_data_path, epoch.get_data())
            print(f"[Epoch {i}] Saved problematic epoch data to: {error_data_path}")
            return epoch.get_data()

    # Calculate time elapsed
    elapsed_time = time.time() - start_time
    print(f"[Epoch {i}] Processing completed in {elapsed_time:.2f} seconds.")

    return epoch_corrected.get_data()


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

        print("Removing trend")
        self.remove_trend()

        print("Filtering data")
        self.filter_data()

        print("Resampling the data")
        self.resample_data(new_sfreq=128)

        print("Identifying bad channels")
        self.identify_bad_epochs_and_channels_combined(amplitude_threshold=0.0006, epoch_duration=30.0,
                                                  max_good_channels_bad=4, max_bad_epochs_ratio=0.2)

        print("Interpolating bad channels")
        self.interpolate_bad_channels()

        print("Clipping signal and applying custom reference")
        self.apply_custom_reference()

        print("Saving pre-ICA files")
        #self.save_to_fif_and_set(file_name_base, stage="pre_ICA", overwrite=overwrite)

        print("Segmenting data into 30-second epochs")
        #self.segment_epochs()

        print("Applying ICA")
        #self.apply_ica()

        print("Saving post-ICA files")
        self.save_to_fif_and_set(file_name_base, stage="_BIN", overwrite=overwrite)

        return self.raw

    def identify_bad_epochs_and_channels_combined(self, amplitude_threshold=0.0006, epoch_duration=30.0,
                                                  max_good_channels_bad=4, max_bad_epochs_ratio=0.2):
        """
        Combines amplitude- and PSD-based analyses to identify bad epochs and bad channels.

        Parameters:
        - amplitude_threshold: Threshold in volts to classify a channel as bad in an epoch (default is 0.0006 V).
        - epoch_duration: Duration of each epoch in seconds (default is 30.0 seconds).
        - max_good_channels_bad: Maximum number of good channels allowed to be bad in an epoch (default is 4).
        - max_bad_epochs_ratio: Maximum allowable ratio of bad epochs for a channel to be considered bad (default is 0.2).

        Returns:
        - results: A dictionary with:
            - "bad_epochs_info": Epochs flagged as bad and their bad channels (amplitude-based).
            - "bad_channels_amplitude": Bad channels identified using amplitude analysis.
            - "bad_channels_psd": Bad channels identified using PSD analysis.
            - "combined_bad_channels": Final list of bad channels combining amplitude and PSD results.
            - "bad_epochs_good_channels": Epochs marked as bad due to issues in good channels.
        """
        print("Starting combined identification of bad epochs and channels...\n")

        # Segment the data into epochs
        self.segment_epochs(epoch_duration=epoch_duration)

        # Step 1: Identify bad epochs and channels using amplitude analysis
        print("Running amplitude-based analysis...\n")
        bad_epochs_info, bad_channels_amplitude, bad_epochs_good_channels = self.identify_bad_epochs_and_channels_amplitude(
            amplitude_threshold=amplitude_threshold,
            epoch_duration=epoch_duration,
            max_good_channels_bad=max_good_channels_bad
        )

        # Step 2: Identify bad channels using PSD analysis
        print("\nRunning PSD-based analysis...\n")
        bad_channels_psd = self._identify_bad_channels_psd(max_bad_epochs_ratio=max_bad_epochs_ratio)

        # Step 3: Combine results from both analyses
        combined_bad_channels = list(set(bad_channels_amplitude + bad_channels_psd))

        # Update raw.info['bads'] with combined bad channels
        self.raw.info['bads'] = combined_bad_channels
        print(f"\nFinal bad channels updated in raw.info: {self.raw.info['bads']}\n")

        # Return all results
        return {
            "bad_epochs_info": bad_epochs_info,
            "bad_channels_amplitude": bad_channels_amplitude,
            "bad_channels_psd": bad_channels_psd,
            "combined_bad_channels": combined_bad_channels,
            "bad_epochs_good_channels": bad_epochs_good_channels
        }

    def _identify_bad_channels_psd(self, max_bad_epochs_ratio=0.2):
        """
        Identifies bad channels using Power Spectral Density (PSD) analysis.

        Parameters:
        - max_bad_epochs_ratio: Maximum allowable ratio of bad epochs for a channel to be considered bad.

        Returns:
        - bad_channels: List of channels identified as bad based on PSD analysis.
        """
        print("Identifying bad channels using PSD analysis...\n")
        bad_channels = []

        # Iterate over channels and compute PSD
        for ch_idx, ch_name in enumerate(self.raw.ch_names):
            psd_means = []

            for epoch_data in self.epochs.get_data():
                psd = mne.time_frequency.psd_array_welch(epoch_data[ch_idx], sfreq=self.raw.info['sfreq'],
                                                         fmin=0.3, fmax=35, verbose=False)
                psd_means.append(psd[0].mean())

            # Calculate IQR for PSD means
            q25, q75 = np.percentile(psd_means, [25, 75])
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr

            # Count bad epochs for the channel
            bad_epochs = sum(1 for psd_mean in psd_means if psd_mean > upper_bound)
            if bad_epochs / len(psd_means) > max_bad_epochs_ratio:
                bad_channels.append(ch_name)
                print(f"Channel {ch_name} has {bad_epochs} bad epochs ({bad_epochs / len(psd_means):.2f} ratio)")

        return bad_channels

    def identify_bad_epochs_and_channels_amplitude(self, amplitude_threshold=0.0006, epoch_duration=30.0,
                                                   max_good_channels_bad=4):
        """
        Identifies bad epochs and bad channels using amplitude thresholds.

        Parameters:
        - amplitude_threshold: Threshold in volts to classify a channel as bad in an epoch.
        - epoch_duration: Duration of each epoch in seconds.
        - max_good_channels_bad: Maximum number of good channels allowed to be bad in an epoch.

        Returns:
        - bad_epochs_info: Dictionary of epochs marked as bad and their bad channels.
        - bad_channels: List of bad channels based on amplitude analysis.
        - bad_epochs_good_channels: List of bad epochs caused by issues in good channels.
        """
        print("Identifying bad epochs and channels based on amplitude...\n")

        # Segment the data into epochs
        self.segment_epochs(epoch_duration=epoch_duration)

        bad_epochs_info = {}
        bad_channel_count = {}

        # Get EEG channel indices
        eeg_channel_indices = mne.pick_types(self.raw.info, eeg=True)

        # Iterate through epochs and check for bad channels
        for epoch_idx, epoch_data in enumerate(self.epochs.get_data()):
            bad_channels = []

            for ch_idx in eeg_channel_indices:
                if np.any(np.abs(epoch_data[ch_idx, :]) > amplitude_threshold):
                    bad_channels.append(self.raw.ch_names[ch_idx])
                    bad_channel_count[self.raw.ch_names[ch_idx]] = bad_channel_count.get(self.raw.ch_names[ch_idx],
                                                                                         0) + 1

            if bad_channels:
                bad_epochs_info[epoch_idx] = bad_channels
                print(f"Epoch {epoch_idx} is bad. Bad channels: {bad_channels}")

        # Identify bad channels based on bad epoch counts
        total_epochs = len(self.epochs)
        bad_channel_percentages = {ch: (count / total_epochs) * 100 for ch, count in bad_channel_count.items()}

        # Use IQR to identify bad channels
        median = np.median(list(bad_channel_percentages.values()))
        q25 = np.percentile(list(bad_channel_percentages.values()), 25)
        q75 = np.percentile(list(bad_channel_percentages.values()), 75)
        iqr = q75 - q25

        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        bad_channels = [ch for ch, percentage in bad_channel_percentages.items() if percentage > upper_bound]

        # Check bad epochs caused by good channels
        good_channels = [ch for ch in self.raw.ch_names if ch not in bad_channels]
        bad_epochs_good_channels = []

        for epoch_idx, epoch_data in enumerate(self.epochs.get_data()):
            bad_good_channels = [
                ch for ch in good_channels if
                np.any(np.abs(epoch_data[self.raw.ch_names.index(ch), :]) > amplitude_threshold)
            ]
            if len(bad_good_channels) > max_good_channels_bad:
                bad_epochs_good_channels.append(epoch_idx)
                print(f"Epoch {epoch_idx} marked as bad due to {len(bad_good_channels)} bad good channels.")

        # Define the export directory structure using label_path
        export_dir = os.path.join(
            self.label_path,  # Base path
            "clinical_scorings",  # Top-level folder for clinical annotations
            self.only_class,  # Group (e.g., control, healthy, etc.)
            self.only_patient  # Subject/Patient identifier
        )

        # Ensure the directory exists
        os.makedirs(export_dir, exist_ok=True)

        # Define file paths for saving bad epochs
        bad_epochs_file = os.path.join(export_dir, f"{self.only_patient}EEG_bad_epochs_HP 0.8.npy")

        # Save the bad epochs to the structured directory
        np.save(bad_epochs_file, np.array(bad_epochs_good_channels))
        print(f"Saved bad epochs to: {bad_epochs_file}")

        return bad_epochs_info, bad_channels, bad_epochs_good_channels

    def interpolate_bad_channels(self):
        if self.raw.info['bads'] and self.run_bad_interpolation:
            self.raw.interpolate_bads(method=dict(eeg="spline"), verbose=False)

    def apply_custom_reference(self):
        """
        Apply a common average reference excluding 'Vertex Reference' and set 'Vertex Reference'
        to the negative of the computed average. Plot and save the epochs for specified channels
        before and after re-referencing, with clipping applied after pre-re-referencing plots.
        """
        print("Applying custom reference...")

        # Define channels to plot
        channels_to_plot = ['E36', 'E224', 'E59', 'E183', 'E116']

        # Directory setup for pre-re-referencing plots
        patient_plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(patient_plot_dir, exist_ok=True)

        epoch_plot_base_dir_pre = os.path.join(patient_plot_dir, "BIN_0.8_Before_re-referencing")
        os.makedirs(epoch_plot_base_dir_pre, exist_ok=True)

        # Segment epochs (before re-referencing)
        self.segment_epochs()

        def plot_and_save(epoch_data, save_path, channels_to_plot, epoch_idx):
            """
            Helper function to plot and save raw data with time in seconds and amplitude in microvolts.

            Parameters:
            - epoch_data: Array of shape (n_channels, n_times) for the current epoch.
            - save_path: Path where the plot will be saved.
            - channels_to_plot: List of channel names to plot.
            - epoch_idx: Index of the epoch being plotted.
            """
            # Get the time array in seconds
            time = self.epochs.times

            # Create the figure and axes
            fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 8), sharex=True)

            for ch_idx, ax in enumerate(axes):
                ch_name = channels_to_plot[ch_idx]
                if ch_name in self.epochs.info['ch_names']:
                    # Get the data for the specific channel
                    ch_idx_in_data = self.epochs.info['ch_names'].index(ch_name)
                    ch_data = epoch_data[ch_idx_in_data, :]  # Shape: (n_times,)

                    # Convert data from volts to microvolts
                    ch_data_in_microvolts = ch_data * 1e6

                    # Plot the data
                    ax.plot(time, ch_data_in_microvolts.T)
                    ax.set_title(f"Channel {ch_name}")
                    ax.set_ylabel("Amplitude (µV)")

            # Set x-axis label for the last subplot
            axes[-1].set_xlabel("Time (s)")

            # Add overall title
            fig.suptitle(f"Epoch {epoch_idx}", fontsize=16)

            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved plot to: {save_path}")

        # Plot each epoch (before re-referencing)
        for epoch_idx, epoch_data in enumerate(self.epochs.get_data()):
            save_path = os.path.join(epoch_plot_base_dir_pre, f"epoch_{epoch_idx}.png")
            plot_and_save(epoch_data, save_path, channels_to_plot, epoch_idx)

        print("Pre-re-referencing plotting completed.")

        # Apply clipping to data
        clip_min = -0.0003  # Minimum allowable amplitude in volts
        clip_max = 0.0003 # Maximum allowable amplitude in volts

        # Clip the raw EEG data
        self.raw._data = np.clip(self.raw._data, clip_min, clip_max)
        print(f"Clipping applied: values limited to range [{clip_min}, {clip_max}] volts.")

        # Find the index of 'Vertex Reference'
        vertex_idx = self.raw.ch_names.index('E257') if 'E257' in self.raw.ch_names else None

        # Select all EEG channels except 'Vertex Reference'
        picks = mne.pick_types(self.raw.info, eeg=True, exclude=['E257'])

        # Calculate the average signal across selected EEG channels
        avg_signal = self.raw.get_data(picks=picks).mean(axis=0)

        # Subtract the calculated average from each EEG channel (except 'Vertex Reference')
        for ch_idx in picks:
            self.raw._data[ch_idx] -= avg_signal

        # Set 'Vertex Reference' to the negative of the average
        if vertex_idx is not None:
            self.raw._data[vertex_idx] = -avg_signal

        # Directory setup for post-re-referencing plots
        epoch_plot_base_dir_post = os.path.join(patient_plot_dir, "BIN_0.8_After_re-referencing")
        os.makedirs(epoch_plot_base_dir_post, exist_ok=True)

        # Segment epochs (after re-referencing)
        self.segment_epochs()

        # Plot each epoch (after re-referencing)
        for epoch_idx, epoch_data in enumerate(self.epochs.get_data()):
            save_path = os.path.join(epoch_plot_base_dir_post, f"epoch_{epoch_idx}.png")
            plot_and_save(epoch_data, save_path, channels_to_plot, epoch_idx)

        print("Re-referencing and plotting completed.")

    def remove_trend(self):
        data, _ = self.raw[:, :]
        self.raw._data = detrend(data, axis=1)

    def filter_data(self, l_freq=0.8, h_freq=35, l_trans_bandwidth=0.3):
        """
        Apply bandpass filtering to the EEG data and save plots of the filtered signal with separate subplots for each channel.
        """

        self.raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', l_trans_bandwidth=l_trans_bandwidth, fir_window='hamming', fir_design='firwin',
                        verbose=False)

        # Create a directory to save plots
        plot_dir = os.path.join(self.save_path, "PLOT", self.only_class, self.only_patient)
        os.makedirs(plot_dir, exist_ok=True)

        print("Saving filtered EEG data plot with subplots using matplotlib...")

        # Define channels for visualization and the time range
        channels_to_plot = ['E36', 'E224', 'E59', 'E183','E116']
        start_time = 10710  # seconds
        duration = 30  # seconds
        fs = int(self.raw.info['sfreq'])  # Sampling frequency

        # Calculate start and end indices
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)
        data, times = self.raw.copy().pick(channels_to_plot).get_data(return_times=True)
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

    def resample_data(self, new_sfreq):
        """
        Resample the EEG data to a new sampling frequency.

        Parameters:
        - new_sfreq: Target sampling frequency (e.g., 256 Hz).
        """
        print(f"Resampling the data to {new_sfreq} Hz...")
        self.raw.resample(new_sfreq, npad="auto", verbose=False)
        print(f"Resampling completed. New sampling frequency: {self.raw.info['sfreq']} Hz")
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
        epoch_plot_base_dir = os.path.join(patient_plot_dir, "ICA0.5_individual_epochs")
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

        print("Interpolating boundaries between epochs...")
        print("Shape of processed data:", processed_data.shape)

        # Assumiamo che processed_data sia (n_channels, n_samples)
        n_channels, n_samples = processed_data.shape
        sfreq = int(self.raw.info['sfreq'])  # Sampling frequency
        epoch_duration = 30.0  # Duration of each epoch
        epoch_length = int(epoch_duration * sfreq)  # Samples per epoch
        n_epochs = n_samples // epoch_length

        # ------------------- Boundary Interpolation -------------------
        # Interpolate boundaries
        for epoch_idx in range(1, n_epochs):
            start_prev = (epoch_idx - 1) * epoch_length
            end_prev = start_prev + epoch_length
            start_curr = epoch_idx * epoch_length
            end_curr = start_curr + epoch_length

            # Extract boundary data
            boundary_data1 = processed_data[:, end_prev - 10:end_prev]  # Last 10 samples of the previous epoch
            boundary_data2 = processed_data[:, start_curr:start_curr + 10]  # First 10 samples of the current epoch

            # Interpolation for each channel
            for ch in range(n_channels):
                boundary_times = np.linspace(0, 1, 20)  # Create interpolation points
                spline = CubicSpline(boundary_times, np.concatenate([boundary_data1[ch], boundary_data2[ch]]))
                interpolated_data = spline(boundary_times)

                # Replace the boundary data with interpolated data
                processed_data[ch, end_prev - 10:end_prev] = interpolated_data[:10]
                processed_data[ch, start_curr:start_curr + 10] = interpolated_data[10:]

        print("Boundary interpolation completed.")

        # ------------------- End of Boundary Interpolation -------------------

        # Combine all data (initial, processed, final)
        all_data = processed_data

        # Create a new Raw object with the combined data
        original_info = self.raw.info.copy()
        self.raw = mne.io.RawArray(all_data, original_info)

        # Save concatenated raw EEG data plot
        print("Saving concatenated raw EEG data plot...")
        channels_to_plot = ['E36', 'E224', 'E59', 'E183', 'E116']
        start_time = 60 * 60 + 60  # seconds
        duration = 60  # seconds
        fs = int(self.raw.info['sfreq'])  # Sampling frequency
        start_idx = int(start_time * fs)
        end_idx = start_idx + int(duration * fs)
        data, times = self.raw.copy().pick(channels_to_plot).get_data(return_times=True)
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


    def convert_to_microvolt(self, raw=None):
        """
        Convert the EEG data from volts to microvolts and update metadata accordingly.
        Parameters:
        - raw: The Raw object to convert. If None, use self.raw.
        """
        if raw is None:
            raw = self.raw

        print("Converting EEG data from volts to microvolts...")

        # Moltiplica i dati per 1e6 per trasformarli in microvolt
        raw._data *= 1e6

        # Aggiorna i metadati per specificare che ora l'unità è microvolt
        for ch in raw.info['chs']:
            ch['unit'] = 201  # Codice 201 corrisponde ai microvolt (µV) in MNE

        print("Conversion to microvolts completed. Metadata updated.")

    def save_to_fif_and_set(self, file_name_base, stage, overwrite=True):
        """
        Save the processed raw EEG data to both .fif and .set formats,
        using a copy of the raw data in the pre_ICA stage and the original in post_ICA.

        Parameters:
        - file_name_base: Base name for the saved files (without extension)
        - stage: A string that indicates the stage ('pre_ICA' or 'post_ICA')
        - overwrite: Whether to overwrite existing files
        """
        # Define the export directory structure: group -> patient -> stage
        export_dir = os.path.join(
            self.save_path,
            f"Preprocessing_{stage}",  # Use the 'stage' parameter to determine the preprocessing phase
            self.only_class,
            self.only_patient
        )
        os.makedirs(export_dir, exist_ok=True)  # Create the full path if it doesn't exist

        # Use a copy of raw for pre_ICA, otherwise use self.raw
        if stage == "pre_ICA":
            raw_to_process = self.raw.copy()  # Create a copy for pre_ICA
            print("Working on a copy of raw for pre_ICA")
        elif stage == "_BIN":
            raw_to_process = self.raw  # Use self.raw directly for post_ICA
            print("Working on self.raw for post_ICA")
        else:
            raise ValueError(f"Unknown stage: {stage}. Must be 'pre_ICA' or 'post_ICA'.")

        print("Converting to microvolts")
        self.convert_to_microvolt(raw=raw_to_process)  # Apply conversion to the copy

        # Save as .fif
        fif_file_path = os.path.join(export_dir, f"{os.path.basename(file_name_base)}.fif")
        try:
            raw_to_process.save(fif_file_path, overwrite=overwrite)
            print(f"Processed raw EEG data saved to {fif_file_path}")
        except Exception as e:
            print(f"Error saving processed EEG data to .fif: {e}")
            return  # Exit early if saving to .fif fails


        # Reload the .fif file to ensure compatibility with EEGLAB export
        try:
            raw_from_fif = mne.io.read_raw_fif(fif_file_path, preload=True)
            print(f"Reloaded .fif file from {fif_file_path} for conversion to .set")
        except Exception as e:
            print(f"Error reloading .fif file: {e}")
            return

        # Define the path for .set file
        set_file_path = os.path.join(export_dir, f"{os.path.basename(file_name_base)}.set")

        # Save as .set
        try:
            # MNE has a built-in function to export data to EEGLAB format (.set + .fdt)
            mne.export.export_raw(set_file_path, raw_from_fif, fmt="eeglab", overwrite=overwrite)
            print(f"Processed raw EEG data saved to {set_file_path} (.set and .fdt)")
        except Exception as e:
            print(f"Error saving processed EEG data to .set: {e}")






