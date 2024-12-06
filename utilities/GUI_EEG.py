import mne
import numpy as np
import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QSlider, QComboBox, QHBoxLayout, \
    QPushButton
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys


class EEGViewer(QMainWindow):
    def __init__(self, fif_file,ica_file):
        super().__init__()
        self.setWindowTitle("EEG Viewer - Preprocessato")

        # Caricamento del file preprocessato e ICA
        self.raw_preprocessed = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

        self.ica = mne.preprocessing.read_ica(ica_file)
        start_sample = int(120 * 60 * self.raw_preprocessed.info['sfreq'])
        end_sample = start_sample + int(60 * 60 * self.raw_preprocessed.info['sfreq'])
        self.raw_preprocessed = self.raw_preprocessed.crop(tmin=start_sample / self.raw_preprocessed.info['sfreq'],
                                                           tmax=end_sample / self.raw_preprocessed.info['sfreq'])

        # Layout principale
        main_layout = QVBoxLayout()

        # Menu a tendina per la selezione del canale
        self.channel_selector = QComboBox()
        self.channel_selector.addItems(self.raw_preprocessed.ch_names)
        self.channel_selector.currentIndexChanged.connect(self.update_plots)
        if self.channel_selector.count() > 0:
            self.channel_selector.setCurrentIndex(0)

        # Slider per navigare tra segmenti di 30 secondi
        self.epoch_slider = QSlider(Qt.Horizontal)
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setMaximum(int(60 * 60 / 30) - 3)
        self.epoch_slider.valueChanged.connect(self.update_plots)

        # Label per l'epoca corrente
        self.epoch_label = QLabel("Segmento: 0")

        # Pulsanti per aprire finestre di topografie, ICA, PSD, e finestra raw
        calc_topo_button = QPushButton("Visualizza Topografie")
        calc_topo_button.clicked.connect(self.open_topography_window)
        view_ica_button = QPushButton("Visualizza Componenti ICA")
        view_ica_button.clicked.connect(self.open_ica_window)
        view_psd_button = QPushButton("Visualizza PSD con Bande")
        view_psd_button.clicked.connect(self.open_psd_window)
        view_raw_button = QPushButton("Visualizza Segnale Raw")
        view_raw_button.clicked.connect(self.open_raw_window)

        # Layout per i controlli
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Seleziona Canale"))
        control_layout.addWidget(self.channel_selector)
        control_layout.addWidget(QLabel("Naviga nei segmenti"))
        control_layout.addWidget(self.epoch_slider)
        control_layout.addWidget(self.epoch_label)
        control_layout.addWidget(calc_topo_button)
        control_layout.addWidget(view_ica_button)
        control_layout.addWidget(view_psd_button)
        control_layout.addWidget(view_raw_button)

        # Canvas per il plot dei segnali
        self.fig = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.axs = [self.fig.add_subplot(3, 1, i + 1) for i in range(3)]
        self.fig.tight_layout(pad=3.0)

        # Aggiunta dei layout
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas)

        # Impostazione del widget centrale
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Variabili per visualizzazione
        self.topography_window = None
        self.ica_window = None
        self.psd_window = None
        self.raw_window = None
        self.update_plots()

    def update_plots(self):
        selected_channel = self.channel_selector.currentText()
        channel_index = self.raw_preprocessed.ch_names.index(selected_channel)
        start_time_offset = 0
        start_segment = self.epoch_slider.value() * 30

        for i in range(3):
            segment_start = int((start_segment + i * 30) * self.raw_preprocessed.info['sfreq'])
            segment_end = int(segment_start + 30 * self.raw_preprocessed.info['sfreq'])
            data, times = self.raw_preprocessed[channel_index, segment_start:segment_end]

            self.axs[i].cla()
            self.axs[i].plot(times + start_time_offset, data.T)
            self.axs[i].set_title(f"Segmento {self.epoch_slider.value() + i} - Canale {selected_channel} (Preprocessato)")
            self.axs[i].set_xlabel("Tempo (s)")
            self.axs[i].set_ylabel("Ampiezza")
            self.axs[i].grid(True)

        self.canvas.draw()

    def open_topography_window(self):
        if self.topography_window is None:
            self.topography_window = TopographyWindow(self.raw_preprocessed)
        self.topography_window.show()

    def open_ica_window(self):
        if self.ica_window is None:
            self.ica_window = ICAViewer(self.raw_preprocessed, self.ica)
        self.ica_window.show()

    def open_psd_window(self):
        if self.psd_window is None:
            self.psd_window = PSDWindow(self.raw_preprocessed)
        self.psd_window.show()

    def open_raw_window(self):
        if self.raw_window is None:
            self.raw_window = RawEEGWindow(self.raw_epoch)
        self.raw_window.show()


class RawEEGWindow(QMainWindow):
    def __init__(self, raw_epoch):
        super().__init__()
        self.setWindowTitle("EEG Viewer - Segnale Raw")

        # Caricamento del segnale raw
        self.raw = raw_epoch

        # Layout principale
        main_layout = QVBoxLayout()

        # Menu a tendina per la selezione del canale
        self.channel_selector = QComboBox()
        self.channel_selector.addItems(self.raw.ch_names)
        self.channel_selector.currentIndexChanged.connect(self.update_plot)
        if self.channel_selector.count() > 0:
            self.channel_selector.setCurrentIndex(0)

        # Canvas per il plot del segnale raw
        self.fig = plt.figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.fig.tight_layout(pad=2.0)

        # Aggiunta dei layout
        main_layout.addWidget(QLabel("Seleziona Canale"))
        main_layout.addWidget(self.channel_selector)
        main_layout.addWidget(self.canvas)

        # Impostazione del widget centrale
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Aggiornamento iniziale del plot
        self.update_plot()

    def update_plot(self):
        selected_channel = self.channel_selector.currentText()
        channel_index = self.raw.ch_names.index(selected_channel)
        data, times = self.raw[channel_index, :]

        # Pulisce e aggiorna il plot
        self.ax.cla()
        self.ax.plot(times, data.T)
        self.ax.set_title(f"Canale {selected_channel} (Raw - Epoca)")
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Ampiezza")
        self.ax.grid(True)

        self.canvas.draw()


class TopographyWindow(QMainWindow):
    def __init__(self, raw_data):
        super().__init__()
        self.setWindowTitle("Topografie delle Bande")
        self.raw = raw_data
        layout = QVBoxLayout()
        self.fig, self.axs = plt.subplots(3, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.calculate_topographies()

    def calculate_topographies(self):
        print("Canali disponibili per il calcolo della PSD:", self.raw.ch_names)

        frequency_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha1': (8, 10),
            'Alpha2': (10, 12),
            'Beta': (15, 30),
            'Gamma': (30, 40)
        }
        band_positions = [
            ('Delta', 0, 0), ('Theta', 1, 0), ('Alpha1', 2, 0),
            ('Alpha2', 0, 1), ('Beta', 1, 1), ('Gamma', 2, 1)
        ]
        psd_min, psd_max = float('inf'), -float('inf')

        for band_name, row, col in band_positions:
            fmin, fmax = frequency_bands[band_name]
            psd, freqs = self.raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)
            psd_band = np.mean(psd[:, (freqs >= fmin) & (freqs <= fmax)], axis=1)
            psd_min = min(psd_min, psd_band.min())
            psd_max = max(psd_max, psd_band.max())
            self.axs[row][col].cla()
            mne.viz.plot_topomap(psd_band, self.raw.info, axes=self.axs[row][col], show=False, cmap='viridis')
            self.axs[row][col].set_title(f"Banda {band_name}")

        cbar_ax = self.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=psd_min, vmax=psd_max)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        self.fig.colorbar(sm, cax=cbar_ax, orientation="vertical", label="Power Spectral Density (µV²/Hz)")

        self.canvas.draw()


class ICAViewer(QMainWindow):
    def __init__(self, raw, ica):
        super().__init__()
        self.setWindowTitle("Componenti ICA")
        self.raw = raw
        self.ica = ica

        layout = QVBoxLayout()
        self.comp_selector = QComboBox()
        self.groups = [f"Componenti {i}-{i + 9}" for i in range(0, ica.n_components_, 10)]
        self.comp_selector.addItems(self.groups)
        self.comp_selector.currentIndexChanged.connect(self.plot_components_group)

        layout.addWidget(QLabel("Seleziona gruppo di componenti ICA"))
        layout.addWidget(self.comp_selector)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.fig, self.axs = plt.subplots(2, 5, figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.plot_components_group()

    def plot_components_group(self):
        group_idx = self.comp_selector.currentIndex() * 10
        components_to_plot = list(range(group_idx, min(group_idx + 10, self.ica.n_components_)))

        for ax in self.axs.flat:
            ax.cla()
        self.ica.plot_components(picks=components_to_plot, show=False, axes=self.axs)
        self.canvas.draw()


class PSDWindow(QMainWindow):
    def __init__(self, raw_data):
        super().__init__()
        self.setWindowTitle("Power Spectral Density")
        self.raw = raw_data

        layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.plot_psd_with_bands()

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def plot_psd_with_bands(self):
        psd, freqs = self.raw.compute_psd().get_data(return_freqs=True)
        psd_mean = psd.mean(axis=0)

        self.ax.plot(freqs, psd_mean, color='pink', label="PSD Media")
        bands = {
            'Delta': (0.5, 4, 'black'),
            'Theta': (4, 8, 'blue'),
            'Alpha': (8, 12, 'green'),
            'Beta': (12, 30, 'yellow'),
            'Gamma': (30, 40, 'red')
        }

        for band, (fmin, fmax, color) in bands.items():
            self.ax.axvspan(fmin, fmax, color=color, alpha=0.3, label=band)

        self.ax.set_xlim(0, 40)
        self.ax.set_title("Power Spectral Density")
        self.ax.set_xlabel("Frequenza (Hz)")
        self.ax.set_ylabel("Power (µV²/Hz)")
        self.ax.legend(loc="upper right")
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    fif_file = r"C:\Users\Lorenzo\Desktop\PD002.fif"
    ica_file = r"C:\Users\Lorenzo\Desktop\PD005-ica.fif"
    viewer = EEGViewer(fif_file, ica_file)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()