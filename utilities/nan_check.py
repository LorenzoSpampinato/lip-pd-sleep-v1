import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

def nan_check(save_path, run_nan_check):
    # save_path: absolute path where the results are stored
    # run_nan_check: flag to run check on NaN values among extracted features

    if not run_nan_check: return

    # Pattern to extract subject id
    pattern = r'(PD\d{3})'
    # Cycle on features DataFrame
    for feat_m in glob.glob(os.path.join(save_path, 'Features') + '/*/*/*.csv'):
        fm2d = pd.read_csv(feat_m, header=0).iloc[:, 4:]
        regs = pd.read_csv(feat_m, header=0).iloc[:, 2].values
        n_epochs = len(np.where(regs == np.unique(regs)[0])[0])
        if np.isnan(fm2d.values).any():
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.suptitle(re.search(pattern, feat_m).group(0), fontsize=20)
            msno.matrix(fm2d, ax=ax)
            fig.tight_layout()
            for i in range((len(fm2d) // n_epochs) + 1):
                ax.axhline(i * n_epochs, color='red', linestyle='--', linewidth=2)
            fig.savefig(os.path.splitext(feat_m)[0] + '.jpg', dpi=300)
            plt.close()