import os
import glob
import time
import numpy as np
import mne
from scipy.io import loadmat
from feature_extraction import ChannelConnectivityFeatureExtractor, NetworkFeatureExtractor, \
    SingleChannelFeatureExtractor
from preprocess import EEGPreprocessor
from utilities import EEGRegionsDivider


class EEGFeatureExtractor:
    def __init__(self, data_path, label_path, save_path, run_preprocess=True, run_bad_interpolation=True):
        # data_path: absolute path to the PSG data i.e., folders for the different disease stages, each with PSG data
        # label_path: absolute path to the hypnograms i.e., folders for the different disease stages, each with stage files
        # save_path: absolute path where to save the results i.e., feature matrices
        # run_preprocess: flag to run pre-processing
        # run_bad_interpolation: flag to run bad channels interpolation

        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        self.run_preprocess = run_preprocess
        self.run_bad_interpolation = run_bad_interpolation

        # Initialize the EEG region divider
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()

        # Channel names and reference setup
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.names[self.names.index('E257')] = 'Vertex Reference'

    def _load_raw_data(self, file_path):
        # Load raw EEG data from the file
        print(f"Loading EEG data from: {file_path}")
        raw = mne.io.read_raw_egi(file_path + '.mff', preload=True, verbose=False)
        raw.pick_types(eeg=True, verbose=False)  # Pick EEG channels only
        return raw

    def _mark_bad_channels(self, raw, sub_fold):
        # Mark bad channels from the associated .mat file
        bads_file = sub_fold.replace('.mff', '').replace(self.data_path, self.label_path) + '.mat'
        bads = loadmat(bads_file)['badchannelsNdx'].squeeze()
        bads = bads.tolist() if np.size(bads) > 1 else [bads]
        bad_names = [raw.ch_names[bad] for bad in bads if raw.ch_names[bad] in self.names]
        raw.info['bads'].extend(bad_names)

    def _preprocess_data(self, raw):
        # Preprocess the EEG data
        preprocessor = EEGPreprocessor(raw, run_preprocess=self.run_preprocess,
                                       run_bad_interpolation=self.run_bad_interpolation)
        return preprocessor.preprocess()

    def _segment_epochs(self, raw):
        # Segment the EEG data into 30-second epochs
        events = mne.make_fixed_length_events(raw, duration=30.0)
        return mne.Epochs(raw=raw, events=events, tmin=0.0, tmax=30.0, baseline=None, preload=True, verbose=False)

    def _extract_features(self, epoched_data, fs):
        # Extract single-channel, channel-connectivity, and network features
        print("Extracting features...")

        # Single-channel features
        sc_extractor = SingleChannelFeatureExtractor(epochs=epoched_data, fs=fs, ch_reg=sorted(self.regions))
        feats_m_sc, feats_sc = sc_extractor.extract_features()

        # Channel-connectivity features
        cc_extractor = ChannelConnectivityFeatureExtractor(epochs=epoched_data, fs=fs, ch_reg=sorted(self.regions))
        feats_m_cc, feats_cc = cc_extractor.extract_features()

        # Network features
        net_extractor = NetworkFeatureExtractor(epochs=epoched_data, ch_reg=sorted(self.regions))
        feats_m_na, feats_na = net_extractor.extract_features()

        # Concatenate all features: [Number of brain regions X Tot number of features X Number of epochs]
        all_feats = np.concatenate([feats_m_sc, feats_m_cc, feats_m_na], axis=1)
        sorted_feats = sorted(feats_sc + feats_cc + feats_na, key=lambda x: int(x.split()[0]))

        return all_feats, sorted_feats

    def _save_features(self, all_feats, sorted_feats, sub_fold):
        # Save the extracted features to a .npz file
        brain_regions = [reg.split('_')[1] for reg in sorted(self.regions)]
        res_sub_fold = sub_fold.replace(self.data_path, os.path.join(self.save_path, 'Features')).replace('.mff', '')
        os.makedirs(res_sub_fold, exist_ok=True)

        np.savez(os.path.join(res_sub_fold, os.path.basename(res_sub_fold) + '_all_feats.npz'),
                 data=all_feats, feats=sorted_feats, regions=brain_regions)

    def extract_features_for_all_subjects(self, only_class=None, only_patient=None , run_features=True):
        if not run_features: return
        # Extract features for all subjects or specific class/patient if specified
        if only_class and only_patient:
            sub_folds = [os.path.join(self.data_path, only_class, only_patient)]
        else:
            sub_folds = glob.glob(self.data_path + '/*/*')

        for sub_fold in sub_folds:
            if not sub_fold.endswith('.mff'):
                os.rename(sub_fold, sub_fold + '.mff')

            print(f"Processing subject: {sub_fold}")

            # Start timing the process
            start_time = time.time()

            # Load raw data
            raw = self._load_raw_data(sub_fold)

            # Mark bad channels
            self._mark_bad_channels(raw, sub_fold)

            # Pick selected channels
            raw.pick(picks=self.names, verbose=False)

            # Preprocess data
            processed_raw = self._preprocess_data(raw)

            # Segment the data into 30-second epochs
            epoched_data = self._segment_epochs(processed_raw)

            # Extract features
            all_feats, sorted_feats = self._extract_features(epoched_data, fs=raw.info['sfreq'])

            # Save the extracted features
            self._save_features(all_feats, sorted_feats, sub_fold)

            # Restore original folder name (remove '.mff')
            if os.path.exists(sub_fold + '.mff'):
                os.rename(sub_fold + '.mff', sub_fold)

            # End timing the process
            end_time = time.time()
            print(f"Finished processing {sub_fold} in {end_time - start_time:.2f} seconds")
