import mne
from scipy.signal import detrend
from mne_icalabel import label_components

def preprocess(raw, run_preprocess, run_bad_interpolation):
    # raw: RawMff object with PSG data
    # run_preprocess: flag to run pre-processing
    # run_bad_interpolation: flag to run bad channels interpolation

    if not run_preprocess: return

    # ----------------------------------  1. Bad channels interpolation  -----------------------------------
    # Bad channels interpolation using "spherical spline method"
    # --> Sensor locations are projected onto a unit sphere before signals at the bad sensor locations are
    # interpolated based on the signals at the good locations.
    if raw.info['bads'] and run_bad_interpolation: raw.interpolate_bads(method=dict(eeg="spline"), verbose=False)

    # -------------------------------------  2. Average re-reference  --------------------------------------
    raw.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg', verbose=False)

    # -----------------------------------------  3. Trend removal  -----------------------------------------
    data, _ = raw[:, :]
    raw._data = detrend(data, axis=1)

    # -------------------------------------------  4. Filtering  -------------------------------------------
    # Band-pass filter (FIR) [0.1 - 40 Hz]
    raw.filter(l_freq=.35, h_freq=40, method='fir', fir_window='hamming', fir_design='firwin', verbose=False)
    # To plot filter mask
    # filt_pars = mne.filter.create_filter(data=None, sfreq=fs, l_freq=.1, h_freq=40,
    #                                      fir_window='hamming', fir_design='firwin')
    # mne.viz.plot_filter(filt_pars, sfreq=fs, freq=None, gain='both')

    # --------------------  5. Artifact Removal: Independent Component Analysis (ICA)  ---------------------
    ica = mne.preprocessing.ICA(n_components=None, method='fastica', verbose=False, random_state=0)
    ica.fit(raw, verbose=False)

    # Possible labels for ICA: ‘brain’, ‘muscle artifact’, ‘eye blink’, ‘heart beat’, ‘line noise’,
    # ‘channel noise’, ‘other’
    # --> "other" = these ICs primarily fall into 2 categories i.e., ICs containing 1) indeterminate noise or
    # 2) multiple signals that ICA could not separate well
    # In HD-EEG recordings (64 channels and above), the majority of ICs typically falls into this category
    # --> “other” = catch-all for non-classifiable components, thus it is picked to stay on the side of caution
    ic_labels = label_components(raw, ica, method="iclabel")
    exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]
    processed_raw = ica.apply(raw, exclude=exclude_idx, verbose=False)

    return processed_raw