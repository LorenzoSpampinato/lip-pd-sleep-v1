
import numpy as np
import mne
from scipy.io import loadmat


class EEGDataLoader:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path

    def load_and_prepare_data(self, file_path):
        raw = self.load_raw_data(file_path)
        self.mark_bad_channels(raw, file_path)
        return raw

    def load_raw_data(self, file_path):
        print(f"Loading EEG data from: {file_path}")
        raw = mne.io.read_raw_egi(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)
        return raw

    def mark_bad_channels(self, raw, sub_fold):
        bads_file = sub_fold.replace('.mff', '').replace(self.data_path, self.label_path) + '.mat'
        bads = loadmat(bads_file)['badchannelsNdx'].squeeze()
        bads = bads.tolist() if np.size(bads) > 1 else [bads]
        bad_names = [raw.ch_names[bad] for bad in bads if raw.ch_names[bad] in raw.ch_names]
        raw.info['bads'].extend(bad_names)