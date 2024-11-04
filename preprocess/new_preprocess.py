import mne
import numpy as np
import os
from scipy.signal import detrend
from mne_icalabel import label_components
import matplotlib.pyplot as plt

class EEGPreprocessor111:
    def __init__(self, raw, data_path, save_path, run_preprocess=True, run_bad_interpolation=True):
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
        self.save_path = save_path
        self.run_preprocess = run_preprocess
        self.run_bad_interpolation = run_bad_interpolation
        self.ica = None
        self.ic_labels = None

    def preprocess(self):
        if not self.run_preprocess:
            return self.raw
        print("interpolating bad channels")
        self.interpolate_bad_channels()
        print("setting average reference")
        self.set_average_reference()
        print("removing trend")
        self.remove_trend()
        print("filtering data")
        self.filter_data()
        print("applying ICA")
        self.apply_ica()
        print("saving preprocessed data")
        original_file_path = self.raw.filenames[0]
        preproc_save_path_bin = self._get_save_path(original_file_path)

        os.makedirs(os.path.dirname(preproc_save_path_bin), exist_ok=True)

        self.raw.get_data().astype(np.float64).tofile(preproc_save_path_bin)
        print(f"Preprocessed data saved in BIN format: {preproc_save_path_bin}")

        return self.raw

    def interpolate_bad_channels(self):
        if self.raw.info['bads'] and self.run_bad_interpolation:
            self.raw.interpolate_bads(method=dict(eeg="spline"), verbose=False)

    def set_average_reference(self):
        self.raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg', verbose=False)

    def remove_trend(self):
        data, _ = self.raw[:, :]
        self.raw._data = detrend(data, axis=1)

    def filter_data(self, l_freq=0.30, h_freq=35):
        self.raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)

    def apply_ica(self):
        # Create and fit ICA
        self.ica = mne.preprocessing.ICA(n_components=None, method='fastica', random_state=0, verbose=False)
        self.ica.fit(self.raw, verbose=False)

        # Save ICA sources plot
        print("Saving ICA sources plot...")
        fig_sources = self.ica.plot_sources(self.raw, start=7200, stop= 7230, title='Ica Plot sources PD011', show=False, show_scrollbars=False)
        fig_sources.savefig(os.path.join(self.save_path, "ica_sources.png"))
        plt.close(fig_sources)

        # Save ICA component plots using plot_ica_components
        print("Saving ICA component plots...")
        component_figures = mne.viz.plot_ica_components(self.ica, show=False)
        if isinstance(component_figures, list):
            for i, fig in enumerate(component_figures):
                fig.savefig(os.path.join(self.save_path, f"ica_component_{i}.png"))
                plt.close(fig)


        # Label ICA components and exclude non-brain-related components
        print("Labeling ICA components...")
        self.ic_labels = label_components(self.raw, self.ica, method="iclabel")
        exclude_idx = [idx for idx, label in enumerate(self.ic_labels["labels"]) if label not in ["brain", "other"]]

        # Save overlay plot for the excluded components
        print("Saving ICA overlay plot...")
        fig_overlay = self.ica.plot_overlay(self.raw, exclude=exclude_idx[:1], picks="eeg", show=False)
        fig_overlay.savefig(os.path.join(self.save_path, "ica_overlay.png"))
        plt.close(fig_overlay)

        # Apply ICA with the exclusion of non-brain components
        self.raw = self.ica.apply(self.raw, exclude=exclude_idx, verbose=False)

    def _get_save_path(self, original_file_path):
        if not original_file_path.endswith('.bin'):
            preproc_save_path_bin = original_file_path.replace(self.data_path, os.path.join(self.save_path, 'Preprocessed_BIN'))
            return os.path.splitext(preproc_save_path_bin)[0] + '.bin'
        else:
            return original_file_path.replace(self.data_path, os.path.join(self.save_path, 'Preprocessed_BIN'))
