import mne

full_file_path= "Z:\home\lorenzo.spampinato\MEDITECH\BSP\data\lid_pd\Dataset_set\ADV\PD002\PD002.set"
raw = mne.io.read_raw_eeglab(full_file_path, preload=True, verbose=False)
print(raw.info['ch_names'])