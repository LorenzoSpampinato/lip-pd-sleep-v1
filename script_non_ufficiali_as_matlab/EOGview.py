import numpy as np
import matplotlib.pyplot as plt
import random

# Definisci il percorso del file EOG (file binario)
percorso_file_eog = "D:/TESI/lid-data-samples/lid-data-samples/Dataset/DYS/PD012.mff/signal2.bin"

# Carica il segnale EOG (presumendo che il formato sia float32)
eog_data = np.fromfile(percorso_file_eog, dtype=np.float32)

# Definisci la frequenza di campionamento (sfreq)
sfreq = 250  # Modifica con la frequenza di campionamento corretta se diversa

# Durata del segnale in secondi
durata_segnale = len(eog_data) / sfreq
print(len(eog_data))
# Lunghezza del segmento in secondi (60 secondi)
segmento_durata = 60

# Seleziona casualmente un inizio del segmento
inizio_segmento = random.randint(0, int(durata_segnale - segmento_durata))

# Converti l'inizio del segmento in campioni
inizio_campione = int(inizio_segmento * sfreq)
fine_campione = int((inizio_segmento + segmento_durata) * sfreq)

# Estrai il segmento dal segnale EOG
eog_segment = eog_data[inizio_campione:fine_campione]

# Crea l'asse temporale per il segmento
time = np.arange(inizio_campione, fine_campione) / sfreq

# Plot del segmento EOG
plt.figure(figsize=(10, 4))
plt.plot(time, eog_segment, label="EOG Segment", color="b")
plt.title(f"EOG Signal - Random 60 seconds starting at {inizio_segmento} seconds")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
