import mne
from scipy.signal import detrend
from mne_icalabel import label_components


class EEGPreprocessor:
    def __init__(self, raw, run_preprocess=True, run_bad_interpolation=True):
        """
        Initializes the EEGPreprocessor object with preprocessing parameters.

        Parameters:
        - raw: RawMff object containing PSG data
        - run_preprocess: flag to run the preprocessing
        - run_bad_interpolation: flag to run bad channel interpolation
        """
        self.raw = raw
        self.run_preprocess = run_preprocess
        self.run_bad_interpolation = run_bad_interpolation
        self.ica = None
        self.ic_labels = None

    def interpolate_bad_channels(self):
        """Interpolates bad channels using the spherical spline method."""
        # Bad channels interpolation using "spherical spline method"
        # --> Sensor locations are projected onto a unit sphere before signals at the bad sensor locations are
        # interpolated based on the signals at the good locations.
        if self.raw.info['bads'] and self.run_bad_interpolation:
            self.raw.interpolate_bads(method=dict(eeg="spline"), verbose=False)

    def set_average_reference(self):
        """Sets average EEG reference for the channels."""
        self.raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg', verbose=False)

    def remove_trend(self):
        """Removes the linear trend from the EEG data."""
        data, _ = self.raw[:, :]
        self.raw._data = detrend(data, axis=1)

    def filter_data(self):
        """Applies a band-pass FIR filter to the EEG data (0.35-40 Hz)."""
        self.raw.filter(l_freq=0.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)
        # To plot filter mask
        # filt_pars = mne.filter.create_filter(data=None, sfreq=fs, l_freq=.1, h_freq=40,
        #                                      fir_window='hamming', fir_design='firwin')
        # mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')

    def apply_ica(self):
        """Performs ICA and removes artifacts based on ICA labels."""
        self.ica = mne.preprocessing.ICA(n_components=None, method='fastica', verbose=False, random_state=0)
        self.ica.fit(self.raw, verbose=False)

        # Possible labels for ICA: ‘brain’, ‘muscle artifact’, ‘eye blink’, ‘heart beat’, ‘line noise’,
        # ‘channel noise’, ‘other’
        # --> "other" = these ICs primarily fall into 2 categories i.e., ICs containing 1) indeterminate noise or
        # 2) multiple signals that ICA could not separate well
        # In HD-EEG recordings (64 channels and above), the majority of ICs typically falls into this category
        # --> “other” = catch-all for non-classifiable components, thus it is picked to stay on the side of caution

        # Label ICA components
        self.ic_labels = label_components(self.raw, self.ica, method="iclabel")
        exclude_idx = [idx for idx, label in enumerate(self.ic_labels["labels"]) if label not in ["brain", "other"]]

        # Apply ICA excluding artifact components
        self.raw = self.ica.apply(self.raw, exclude=exclude_idx, verbose=False)

    def preprocess(self):
        """Runs the preprocessing pipeline if the flag is set."""
        if not self.run_preprocess:
            return self.raw

        # 1. Interpolate bad channels
        self.interpolate_bad_channels()

        # 2. Set average reference
        self.set_average_reference()

        # 3. Remove linear trend
        self.remove_trend()

        # 4. Apply filtering
        self.filter_data()

        # 5. Remove artifacts using ICA
        self.apply_ica()

        return self.raw

