import os
import glob
import time
import numpy as np
import mne
from scipy.io import loadmat
from feature_extraction import feature_extraction_channel_connectivity
from feature_extraction import feature_extraction_network_analysis
from feature_extraction import feature_extraction_single_channel
from preprocess import preprocess
from utilities import divide_to_regions


def features_extraction(data_path, label_path, save_path, run_features, run_preprocess, run_bad_interpolation,
                        only_class=None, only_patient=None):
    # data_path: absolute path to the PSG data i.e., folders for the different disease stages, each with PSG data
    # label_path: absolute path to the hypnograms i.e., folders for the different disease stages, each with stage files
    # save_path: absolute path where to save the results i.e., feature matrices
    # run_features: flag to run feature extraction
    # run_preprocess: flag to run pre-processing
    # run_bad_interpolation: flag to run bad channels interpolation
    # only_class: class i.e., disease stage only for which features are extracted
    # only_patient: patient's name only for which features are extracted

    if not run_features: return

    feat_save_path = os.path.join(save_path, 'Features')

    ################################################################################################################
    # --------------------------------------------  Channel selection  ---------------------------------------------
    ################################################################################################################
    # Picking only 150 scalp-EEG channels from all available ones
    regs, idx_chs = divide_to_regions()
    names = ['E' + str(idx_ch) for idx_ch in idx_chs]
    names[names.index('E257')] = 'Vertex Reference'

    # Cycle on classes i.e., PD = {DNV (de-novo), ADV (advanced), DYS (dyskinetic)} vs CTL (control)
    # --> assign numbers to labels of PD disease stage
    if only_class and only_patient: sub_folds = [os.path.join(data_path, only_class, only_patient)]
    else: sub_folds = glob.glob(data_path + '/*/*')

    for sub_fold in sub_folds:
        res_sub_fold = sub_fold.replace(data_path, feat_save_path).replace('.mff', '')
        if not os.path.exists(res_sub_fold): os.makedirs(res_sub_fold, exist_ok=True)

        # Changing the folder name by adding .mff at the end if not present
        if not os.path.exists(sub_fold.replace('.mff', '') + '.mff'):
            os.rename(sub_fold.replace('.mff', ''), sub_fold.replace('.mff', '') + '.mff')

        # Start time: before loading the signal
        step0_end = time.time()

        # Loading data (when running on cluster)
        raw = mne.io.read_raw_egi(sub_fold.replace('.mff', '') + '.mff', preload=True, verbose=False)

        # Time for loading data
        step1_end = time.time()
        print(f"Time for loading data: {step1_end - step0_end:.2f} seconds")

        # Extracting useful information
        #############################
        # Montage: GSN-HydroCel-257 #
        #############################
        # The unit of measurements of hd-EEG signals is Volt, [V] (by looking at the plots)
        # In the .mff files --> unit = uV (microVolt)
        fs = raw.info['sfreq']
        raw.pick_types(eeg=True, verbose=False)

        # Marking bad channels
        if not raw.info['bads']:
            bads = loadmat(sub_fold.replace('.mff', '').replace(data_path, label_path) +
                           '.mat')['badchannelsNdx'].squeeze()
            if np.size(bads) == 1: bads = [bads]
            else: bads = bads.tolist()
            bad_names = [raw.ch_names[bad] for bad in bads if raw.ch_names[bad] in names]
            raw.info['bads'].extend(bad_names)

        # Picking selected channels
        raw.pick(picks=names, verbose=False)

        ########################################################################################################
        # -------------------------------------------  Pre-processing  -----------------------------------------
        ########################################################################################################
        processed_raw = preprocess(raw, run_preprocess, run_bad_interpolation)

        ########################################################################################################
        # --------------------------  Segmenting data into consecutive 30-s segments  --------------------------
        ########################################################################################################
        # Note: important that segmentation is performed relative to the sleep scoring i.e., using 30-s epochs
        events = mne.make_fixed_length_events(processed_raw, duration=30.0)
        epoched_data = mne.Epochs(raw=processed_raw, events=events, tmin=0.0, tmax=30.0,
                                  baseline=None, preload=True, verbose=False)

        # Time for pre-processing data
        step2_end = time.time()
        print(f"Time for pre-processing data: {step2_end - step1_end:.2f} seconds")

        ########################################################################################################
        # ----------------------------------------  Feature extraction  ----------------------------------------
        ########################################################################################################
        # Single-channel features
        feats_m_sc, feats_sc = feature_extraction_single_channel(epochs=epoched_data, fs=fs, ch_reg=sorted(regs))

        # Time for extracting single-channel features
        step3_end = time.time()
        print(f"Time for extracting all single-channel features: {step3_end - step2_end:.2f} seconds")

        # Channel-connectivity features
        feats_m_cc, feats_cc = feature_extraction_channel_connectivity(epochs=epoched_data, fs=fs, ch_reg=sorted(regs))

        # Time for extracting channel-connectivity features
        step4_end = time.time()
        print(f"Time for extracting all channel-connectivity features: {step4_end - step3_end:.2f} seconds")

        # Network analysis features
        feats_m_na, feats_na = feature_extraction_network_analysis(epochs=epoched_data, ch_reg=sorted(regs))

        # Time for extracting network analysis features
        step5_end = time.time()
        print(f"Time for extracting all network analysis features: {step5_end - step4_end:.2f} seconds")

        # ------------------------------------------------------------------------------------------------------
        # For each subject, each feature matrix is [Number of brain regions X Number of features X Number of Epochs]
        # There are 45 single-channel features, 12 channel-connectivity features, and 5 network analysis features
        # There are 6 brain regions --> feature matrices: [6, 45, epochs], [6, 12, epochs], [6, 5, epochs]
        # Aggregating all the features --> [Number of brain regions X Tot number of features X Number of epochs]
        # --> all_feats is [6, 62, N_epochs]
        all_feats = np.concatenate([feats_m_sc, feats_m_cc, feats_m_na], axis=1)
        brain_regions = [reg.split('_')[1] for reg in sorted(regs)]
        sorted_feats = sorted(feats_sc + feats_cc + feats_na, key=lambda x: int(x.split()[0]))
        np.savez(os.path.join(res_sub_fold, os.path.basename(res_sub_fold) + '_all_feats.npz'),
                 data=all_feats, feats=sorted_feats, regions=brain_regions)

        # Restoring the proper name for data folders i.e., removing the suffix '.mff'
        if os.path.exists(sub_fold.replace('.mff', '') + '.mff'):
            os.rename(sub_fold.replace('.mff', '') + '.mff', sub_fold.replace('.mff', ''))