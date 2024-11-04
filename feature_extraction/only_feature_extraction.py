import os
import numpy as np
import mne
from feature_extraction import ChannelConnectivityFeatureExtractor, NetworkFeatureExtractor, SingleChannelFeatureExtractor
import time
from utilities import EEGRegionsDivider

class EEGFeatureExtractor111:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        # Initialize the EEG region divider
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()

        # Channel names and reference setup
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.names[self.names.index('E257')] = 'Vertex Reference'

    def process_and_save_features(self, raw, sub_fold):
        """Process the raw data to extract features and save them."""
        # Segment the data into epochs
        epoched_data = self.segment_epochs(raw)

        # Extract features
        all_feats, sorted_feats = self.extract_features(epoched_data, fs=raw.info['sfreq'])

        # Save the extracted features
        self.save_features(all_feats, sorted_feats, sub_fold)

    def segment_epochs(self, raw):
        """Segment the EEG data into 30-second epochs."""
        events = mne.make_fixed_length_events(raw, duration=30.0)
        return mne.Epochs(raw=raw, events=events, tmin=0.0, tmax=30.0, baseline=None, preload=True, verbose=False)


    def extract_features(self, epoched_data, fs):
        """Extract single-channel, channel-connectivity, and network features with timing for each step."""
        print("Extracting features...")

        # Dictionary to store processing times for each feature extraction step
        processing_times = {}

        # Single-channel feature extraction
        start_time = time.time()
        sc_extractor = SingleChannelFeatureExtractor(epochs=epoched_data, fs=fs, ch_reg=sorted(self.regions))
        feats_m_sc, feats_sc = sc_extractor.extract_features()
        processing_times['Single-Channel Feature Extraction'] = time.time() - start_time

        # Channel connectivity feature extraction
        start_time = time.time()
        cc_extractor = ChannelConnectivityFeatureExtractor(epochs=epoched_data, fs=fs, ch_reg=sorted(self.regions))
        feats_m_cc, feats_cc = cc_extractor.extract_features()
        processing_times['Channel-Connectivity Feature Extraction'] = time.time() - start_time

        # Network analysis feature extraction
        start_time = time.time()
        net_extractor = NetworkFeatureExtractor(epochs=epoched_data, ch_reg=sorted(self.regions))
        feats_m_na, feats_na = net_extractor.extract_features()
        processing_times['Network Analysis Feature Extraction'] = time.time() - start_time

        # Combine all features
        all_feats = np.concatenate([feats_m_sc, feats_m_cc, feats_m_na], axis=1)
        sorted_feats = sorted(feats_sc + feats_cc + feats_na, key=lambda x: int(x.split()[0]))

        # Print or log the processing times
        for step, duration in processing_times.items():
            print(f"{step} took {duration:.2f} seconds.")

        return all_feats, sorted_feats

    def save_features(self, all_feats, sorted_feats, sub_fold):
        """Save the extracted features to a .npz file."""
        brain_regions = [reg.split('_')[1] for reg in sorted(self.regions)]
        res_sub_fold = sub_fold.replace(self.data_path, os.path.join(self.save_path, 'Features')).replace('.mff', '')
        os.makedirs(res_sub_fold, exist_ok=True)

        np.savez(os.path.join(res_sub_fold, os.path.basename(res_sub_fold) + '_features.npz'),
                 data=all_feats, feats=sorted_feats, regions=brain_regions)
        print(f"Features saved for {sub_fold} in {res_sub_fold}")



