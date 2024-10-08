import os
import glob
import re
import numpy as np
import pandas as pd

def dataframe_generation(label_path, save_path, run_aggregation, aggregate_labels):
    # label_path: absolute path to the hypnograms
    # save_path: absolute path where the results are stored
    # run_aggregation: flag to run data aggregation for dataframe definition (for each subject separately)
    # aggregate_labels: flag to whether consider labels during dataframe definition

    if not run_aggregation: return

    # Pattern to extract PD group
    pattern_1 = r'(' + '|'.join(os.listdir(label_path)) + ')'
    # Pattern to extract subject id
    pattern_2 = r'(PD\d{3})'
    # Cycle on features files
    for feat_m in glob.glob(os.path.join(save_path, 'Features') + '/*/*/*.npz'):
        # Loading features data --> dimensions [N_brain_regions X N_features X N_epochs]
        fm3d = np.load(feat_m)['data'].squeeze()
        # -----------------------------------------------  Data  -----------------------------------------------
        # Aggregating features from different brain regions --> dimensions [N_features X (N_brain_regions * N_epochs)]
        fm2d = np.concatenate(fm3d, axis=1)
        # --------------------------------------------  Row labels  --------------------------------------------
        # Extracting PD group
        group = [re.search(pattern_1, feat_m).group(0)]
        # Extracting subject id
        sub_id = [re.search(pattern_2, feat_m).group(0)]
        # Extracting brain regions names
        regs = np.load(feat_m)['regions'].tolist()

        # If sleep stages must be considered for defining the pandas DataFrame
        if aggregate_labels:
            # Extracting sleep stages labels
            directory, _ = os.path.split(feat_m.replace(os.path.join(save_path, 'Features'), label_path))
            stages = np.load(directory + '.npy').squeeze().astype(str).tolist()
            stages = list(map(lambda x: x.replace('0', 'W').replace('1', 'N1').replace('2', 'N2')
                              .replace('3', 'N3').replace('4', 'REM'), stages))
        else: stages = ['/'] * np.size(fm3d, 2)
        multi_index = pd.MultiIndex.from_product([group, sub_id, regs, stages],
                                                 names=['Group', 'Subject', 'Brain region', 'Stage'])
        # ------------------------------------------  Column labels  -------------------------------------------
        # Extracting features names
        feats = np.load(feat_m)['feats'].tolist()
        # -----------------------------------------  Pandas DataFrame  -----------------------------------------
        df_fm2d = pd.DataFrame(fm2d.T, index=multi_index, columns=feats)
        # Saving DataFrame as CSV file
        df_fm2d.to_csv(os.path.splitext(feat_m)[0] + '.csv', index=True, header=True, na_rep='NaN')
        ########################################################################################################
        # Note: when loading the dataframe use:
        # --> "pd.read_csv(os.path.splitext(feat_m)[0] + '.csv', header=0)"