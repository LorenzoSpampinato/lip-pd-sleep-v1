import mne
import matplotlib.pyplot as plt

# Specifica il percorso del file FIF
file_path = r'C:\Users\Lorenzo\Desktop\PD045.fif'  # Sostituisci con il percorso del tuo file

# Carica i dati dal file FIF
raw = mne.io.read_raw_fif(file_path, preload=True)
print(raw.info)

# Estrai i dati per tutti i canali
data, times = raw[:, :]  # Ottieni tutti i dati per tutti i canali

# Ottieni l'indice corrispondente a 180 secondi
sfreq = raw.info['sfreq']  # Frequenza di campionamento
end_idx = int(60 * sfreq)  # Indice corrispondente ai primi 180 secondi

# Ciclo per i primi 3 canali
for i, ch_name in enumerate(raw.info['ch_names'][:50]):  # Solo i primi 3 canali
    plt.figure(figsize=(12, 6))  # Crea una nuova figura per ogni canale
    plt.plot(times[:end_idx], data[i, :end_idx].T)  # Plotta i dati del canale per i primi 180 secondi
    plt.title(f'Segno del Canale: {ch_name}')  # Titolo con il nome del canale
    plt.xlabel('Tempo (s)')
    plt.ylabel('Ampiezza')
    plt.grid()
    plt.show()
