import numpy as np
import pandas as pd
from utilities import EEGDataFrameGenerator

# Path to your .npz file
npz_file_path = r"D:\TESI\lid-data-samples\lid-data-samples\Results\Features\DYS\PD012\\PD012prova_all_feats.npz"
label_path = r"D:\TESI\lid-data-samples\lid-data-samples\Labels"
save_path = r"D:\TESI\lid-data-samples\lid-data-samples\Results"
aggregate_labels = True
run_aggregation = True
generator = EEGDataFrameGenerator(label_path, save_path, aggregate_labels)
generator.generate_dataframe(run_aggregation)