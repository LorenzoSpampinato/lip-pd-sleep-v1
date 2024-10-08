import mne
import matplotlib.pyplot as plt

# Definisci il percorso del file
percorso_file = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/PD012_parziale.fif"

# Carica il file con preload=True
raw = mne.io.read_raw_fif(percorso_file, preload=True, verbose=False)

# Visualizza la disposizione degli elettrodi
raw.plot_sensors(show=True)
plt.title("Disposizione degli elettrodi")
plt.show()
