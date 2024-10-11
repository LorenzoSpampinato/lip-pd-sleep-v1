import mne

# Definisci il percorso del file e il percorso per salvare il file croppato
percorso_cartella = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff"
percorso_file_croppato = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012_cropped.fif"

# Carica il file MFF con preload=True
raw = mne.io.read_raw_egi(percorso_cartella, preload=True, verbose=False)

# Definisci la terza ora in secondi (3 ore = 3 * 60 * 60 secondi)
terza_ora_inizio = 3 * 60 * 60
mezz_ora = 0.5 * 60 * 60
# Croppa il file a partire dalla terza ora fino alla fine
raw_cropped = raw.crop(tmin=terza_ora_inizio, tmax= (terza_ora_inizio + mezz_ora))

# Salva il file croppato in formato .fif, mantenendo le informazioni originali
raw_cropped.save(percorso_file_croppato, overwrite=True)
