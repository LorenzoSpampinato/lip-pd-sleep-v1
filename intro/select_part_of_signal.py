import os
import struct
import matplotlib.pyplot as plt
import mne
import numpy as np

#I have problems with the .bin format but with .fif I can extract smaller pieces of signal.
# Then I extract 5% of the signal into .fif
print("inizio")
# Definisci il percorso del file
percorso_cartella = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff"

# Carica tutto il file con preload=True
raw = mne.io.read_raw_egi(percorso_cartella, preload=True, verbose=False)

# Ottieni il numero totale di campioni
n_campioni_totali = raw.n_times

# Calcola il 5% dei campioni
n_campioni_5_percento = int(n_campioni_totali * 1)

# Estrai i primi 5% del segnale (fino a n_campioni_5_percento)
dati_parziali = raw.get_data()[:, :n_campioni_5_percento]

# Crea un nuovo oggetto RawArray mantenendo le stesse info di raw
info = raw.info  # Mantieni le informazioni originali
raw_parziale = mne.io.RawArray(dati_parziali, info)

# Salva i dati parziali in un file .fif
percorso_output_fif = 'D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/signal1.fif'
raw_parziale.save(percorso_output_fif, overwrite=True)

# Salva i dati parziali anche in un file .bin (usando float64)
#percorso_output_bin = 'D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012_parziale.bin'
#dati_parziali.astype('float64').tofile(percorso_output_bin)

print(f"I primi 5% dei dati sono stati salvati con successo in {percorso_output_fif} ")



