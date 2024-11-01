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

        ## Save the raw data in EDF format
        #edf_save_path = file_path.replace(self.data_path, os.path.join(self.save_path, 'Raw_EDF')) + '.edf'
        #os.makedirs(os.path.dirname(edf_save_path), exist_ok=True)
        #raw.export(edf_save_path, fmt='edf')
        #print(f"Raw data saved in EDF format: {edf_save_path}")

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
        processed_raw = preprocessor.preprocess()
        #Need to fix it
        # Save the preprocessed data in both .bin and .edf formats
        #preproc_save_path_edf = raw.filenames[0].replace(self.data_path, os.path.join(self.save_path, 'Preprocessed_EDF')) + '.edf'
        #os.makedirs(os.path.dirname(preproc_save_path_edf), exist_ok=True)
        #processed_raw.export(preproc_save_path_edf, fmt='edf')
        #print(f"Preprocessed data saved in EDF format: {preproc_save_path_edf}")

        # Ottieni il percorso originale del file preprocessato
        original_file_path = raw.filenames[0]

        # Verifica se il file ha già l'estensione .bin
        if not original_file_path.endswith('.bin'):
            # Cambia l'estensione in .bin se non è già .bin
            preproc_save_path_bin = original_file_path.replace(self.data_path,
                                                               os.path.join(self.save_path, 'Preprocessed_BIN'))
            preproc_save_path_bin = os.path.splitext(preproc_save_path_bin)[0] + '.bin'  # Cambia estensione in .bin
        else:
            # Mantieni il file .bin senza fare ulteriori modifiche
            preproc_save_path_bin = original_file_path.replace(self.data_path,
                                                               os.path.join(self.save_path, 'Preprocessed_BIN'))

        # Creazione delle directory se non esistono
        os.makedirs(os.path.dirname(preproc_save_path_bin), exist_ok=True)

        # Salva il file preprocessato in formato .bin
        processed_raw.get_data().astype(np.float64).tofile(preproc_save_path_bin)
        print(f"Preprocessed data saved in BIN format: {preproc_save_path_bin}")

        return processed_raw

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

        np.savez(os.path.join(res_sub_fold, os.path.basename(res_sub_fold) + 'prova_all_feats.npz'),
                 data=all_feats, feats=sorted_feats, regions=brain_regions)

    def extract_features_for_all_subjects(self, only_class=None, only_patient=None, run_features=True):
        if not run_features:
            return

        # Extract features for all subjects or specific class/patient if specified
        if only_class and only_patient:
            sub_folds = [os.path.join(self.data_path, only_class, only_patient)]
        else:
            sub_folds = glob.glob(self.data_path + '/*/*')

        for sub_fold in sub_folds:
            print(f"Start processing: {sub_fold}")  # Print start of processing

            if not sub_fold.endswith('.mff'):
                print(f"Renaming {sub_fold} to {sub_fold + '.mff'}...")  # Print before renaming
                os.rename(sub_fold, sub_fold + '.mff')
                print(f"End renaming: {sub_fold} to {sub_fold + '.mff'}")  # Print after renaming

            print(f"Processing subject: {sub_fold}")

            # Start timing the process
            start_time = time.time()

            # Load raw data
            start_time_loading = time.time()
            print(f"Start loading raw data of: {sub_fold}...")  # Print before loading
            raw = self._load_raw_data(sub_fold)
            end_time_loading = time.time()
            print(f"End loading raw data of: {sub_fold}...")  # Print after loading
            print(f"Time taken to load raw data: {end_time_loading - start_time_loading:.2f} seconds")

            # Mark bad channels
            start_time_marking = time.time()
            print(f"Start marking bad channels for: {sub_fold}...")  # Print before marking bad channels
            self._mark_bad_channels(raw, sub_fold)
            end_time_marking = time.time()
            print(f"End marking bad channels for: {sub_fold}...")  # Print after marking bad channels
            print(f"Time taken to mark bad channels: {end_time_marking - start_time_marking:.2f} seconds")

            # Pick selected channels
            start_time_picking = time.time()
            print(f"Start picking selected channels for: {sub_fold}...")  # Print before picking channels
            raw.pick(picks=self.names, verbose=False)
            end_time_picking = time.time()
            print(f"End picking selected channels for: {sub_fold}...")  # Print after picking channels
            print(f"Time taken to pick selected channels: {end_time_picking - start_time_picking:.2f} seconds")

            # Preprocess data
            start_time_preprocessing = time.time()
            print(f"Start preprocessing data for: {sub_fold}...")  # Print before preprocessing
            processed_raw = self._preprocess_data(raw)
            end_time_preprocessing = time.time()
            print(f"End preprocessing data for: {sub_fold}...")  # Print after preprocessing
            print(f"Time taken to preprocess data: {end_time_preprocessing - start_time_preprocessing:.2f} seconds")

            # Segment the data into 30-second epochs
            start_time_segmenting = time.time()
            print(f"Start segmenting data into epochs for: {sub_fold}...")  # Print before segmenting
            epoched_data = self._segment_epochs(processed_raw)
            end_time_segmenting = time.time()
            print(f"End segmenting data into epochs for: {sub_fold}...")  # Print after segmenting
            print(f"Time taken to segment data into epochs: {end_time_segmenting - start_time_segmenting:.2f} seconds")

            # Extract features
            start_time_extracting = time.time()
            print(f"Start extracting features for: {sub_fold}...")  # Print before extracting features
            all_feats, sorted_feats = self._extract_features(epoched_data, fs=raw.info['sfreq'])
            end_time_extracting = time.time()
            print(f"End extracting features for: {sub_fold}...")  # Print after extracting features
            print(f"Time taken to extract features: {end_time_extracting - start_time_extracting:.2f} seconds")

            # Save the extracted features
            start_time_saving = time.time()
            print(f"Start saving features for: {sub_fold}...")  # Print before saving features
            self._save_features(all_feats, sorted_feats, sub_fold)
            end_time_saving = time.time()
            print(f"End saving features for: {sub_fold}...")  # Print after saving features
            print(f"Time taken to save features: {end_time_saving - start_time_saving:.2f} seconds")

            # Restore original folder name (remove '.mff')
            if os.path.exists(sub_fold + '.mff'):
                print(f"Renaming {sub_fold + '.mff'} back to {sub_fold}...")  # Print before renaming back
                os.rename(sub_fold + '.mff', sub_fold)
                print(f"End renaming back: {sub_fold + '.mff'} to {sub_fold}")  # Print after renaming back

            # End timing the process
            end_time = time.time()
            print(f"Finished processing {sub_fold} in {end_time - start_time:.2f} seconds")  # Print end of processing

