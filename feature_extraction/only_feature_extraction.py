import os
import numpy as np
import mne
from feature_extraction import ChannelConnectivityFeatureExtractor, NetworkFeatureExtractor, SingleChannelFeatureExtractor
import time
from utilities import EEGRegionsDivider

class EEGFeatureExtractor:
    def __init__(self, data_path, label_path, save_path, only_stage=None, only_patient=None):
        self.data_path = data_path
        self.label_path = label_path
        self.save_path = save_path
        # Initialize the EEG region divider
        self.divider = EEGRegionsDivider()
        self.regions = self.divider.get_all_regions()
        self.idx_chs = self.divider.get_index_channels()

        # Channel names and reference setup
        self.names = ['E' + str(idx_ch) for idx_ch in self.idx_chs]
        self.names[self.names.index('E257')] = 'Vertex Reference'
        self.only_stage = only_stage
        self.only_patient = only_patient


    def process_and_save_features(self, raw, sub_fold):
        """Process the raw data to extract features and save them."""
        # Segment the data into epochs
        epoched_data = self.segment_epochs(raw)

        epoched_filtered_data = self.filter_epochs(epoched_data, raw)

        # Extract features
        all_feats, sorted_feats = self.extract_features(epoched_filtered_data, fs=raw.info['sfreq'])

        # Save the extracted features
        self.save_features(all_feats, sorted_feats, sub_fold)

    def segment_epochs(self, raw):
        """Segment the EEG data into 30-second epochs."""
        events = mne.make_fixed_length_events(raw, duration=30.0)
        return mne.Epochs(raw=raw, events=events, tmin=0.0, tmax=30.0, baseline=None, preload=True, verbose=False)

    def filter_epochs(self, epoched_data, raw):
        """
        Filter the epochs based on clinical scorings using self.only_patient and exclude bad epochs.

        Parameters:
            epoched_data (mne.Epochs): The segmented EEG data.
            raw (mne.io.Raw): The raw EEG data containing bad epochs information.

        Returns:
            mne.Epochs: Filtered epochs with clinical score '3', excluding bad epochs.
        """
        # Construct the path to the clinical scoring file
        clinical_scoring_path = os.path.join(self.label_path, 'clinical_scorings', f'{self.only_patient}.npy')
        print('clinical_scoring_path:', clinical_scoring_path)

        # Ensure the file exists
        if not os.path.exists(clinical_scoring_path):
            raise FileNotFoundError(f"Clinical scoring file not found: {clinical_scoring_path}")

        # Load clinical scores
        clinical_scores = np.load(clinical_scoring_path)
        #print("clinical_scores: ", clinical_scores[0])
        #print("epoched_data: ", epoched_data[0])
        # Ensure the number of epochs matches the clinical scoring
        if len(clinical_scores)-1 != len(epoched_data):
            raise ValueError(
                f"Mismatch between number of epochs ({len(epoched_data)}) and clinical scoring entries ({len(clinical_scores)})")

        # Select only the epochs corresponding to clinical score '3'
        selected_indices = np.where(clinical_scores == 3)[0]
        print("length pre bad epochs: ", len(selected_indices))
        print("selected_indices (clinical score '3'):", selected_indices)

        print(raw._bad_epochs)

        # Check for bad epochs in raw._bad_epochs if available
        if hasattr(raw, '_bad_epochs'):
            bad_epochs = set(raw._bad_epochs)  # Convert to set for efficient lookup
            print("Bad epochs from raw._bad_epochs:", bad_epochs)

            # Exclude bad epochs from the selected indices
            selected_indices = [idx for idx in selected_indices if idx not in bad_epochs]
            print("selected_indices (after excluding bad epochs):", selected_indices)

        # Filter the epoched data to keep only the selected indices
        filtered_epoched_data = epoched_data[selected_indices]

        print(f"Selected {len(selected_indices)} epochs with clinical score '3', excluding bad epochs.")

        return filtered_epoched_data, selected_indices

    def extract_features(self, epoched_filtered_data, fs):
        """Extract single-channel, channel-connectivity, and network features with timing for each step."""

        print("Extracting features...")

        # Dictionary to store processing times for each feature extraction step
        processing_times = {}

        # Single-channel feature extraction
        start_time = time.time()
        sc_extractor = SingleChannelFeatureExtractor(epochs=epoched_filtered_data, fs=fs, ch_reg=sorted(self.regions), save_path=self.save_path, only_stage=self.only_stage, only_patient=self.only_patient)
        feats_m_sc, feats_sc = sc_extractor.extract_features()
        processing_times['Single-Channel Feature Extraction'] = time.time() - start_time

        # Channel connectivity feature extraction
        start_time = time.time()
        cc_extractor = ChannelConnectivityFeatureExtractor(epochs=epoched_filtered_data, fs=fs, ch_reg=sorted(self.regions))
        feats_m_cc, feats_cc = cc_extractor.extract_features()
        processing_times['Channel-Connectivity Feature Extraction'] = time.time() - start_time

        # Network analysis feature extraction
        start_time = time.time()
        net_extractor = NetworkFeatureExtractor(epochs=epoched_filtered_data, ch_reg=sorted(self.regions))
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



