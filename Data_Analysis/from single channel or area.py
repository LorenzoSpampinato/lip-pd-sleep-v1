from scipy.stats import f_oneway, wilcoxon, kruskal, normaltest, levene, bartlett, fligner, ttest_rel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


def analyze_and_plot(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=None):
    results = {}
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)
    filtered_data = data[data['Phase_Assigned'].isin(['Early', 'Late'])]

    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        filtered_data_all = filtered_data
    else:
        filtered_data_selected = pd.DataFrame()
        filtered_data_all = filtered_data

    subject_means_all = filtered_data_all.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[
        feature_name].mean().reset_index()
    channel_means_all = subject_means_all.groupby(['Group', 'Channel', 'Phase_Assigned'], observed=True)[
        feature_name].mean().reset_index()
    overall_means_all = channel_means_all.groupby(['Group', 'Phase_Assigned'], observed=True)[
        feature_name].mean().reset_index()

    if selected_channels is not None:
        subject_means_selected = \
        filtered_data_selected.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[
            feature_name].mean().reset_index()
        channel_means_selected = subject_means_selected.groupby(['Group', 'Channel', 'Phase_Assigned'], observed=True)[
            feature_name].mean().reset_index()
        overall_means_selected = channel_means_selected.groupby(['Group', 'Phase_Assigned'], observed=True)[
            feature_name].mean().reset_index()
    else:
        overall_means_selected = pd.DataFrame()

    def statistical_tests(channel_means, label):
        for phase in ['Early', 'Late']:
            phase_data = channel_means[channel_means['Phase_Assigned'] == phase]
            groups = phase_data['Group'].unique()
            data = [phase_data[phase_data['Group'] == group][feature_name].dropna() for group in groups]

            print(f"\n===== {label} - {phase} =====")
            normality = {group: normaltest(group_data)[1] for group, group_data in zip(groups, data)}
            print("Test di normalità (p-value):", normality)

            if all(p > 0.05 for p in normality.values()):
                levene_p = levene(*data)[1]
                bartlett_p = bartlett(*data)[1]
                var_test = "Levene" if levene_p < 0.05 else "Bartlett"
                var_p = levene_p if levene_p < 0.05 else bartlett_p
            else:
                fligner_p = fligner(*data)[1]
                var_test = "Fligner-Killeen"
                var_p = fligner_p

            print(f"Test di varianza: {var_test} (p={var_p:.2e})")

            if all(p > 0.05 for p in normality.values()):
                if var_p > 0.05:
                    stat, p = f_oneway(*data)
                    test_type = "ANOVA"
                else:
                    stat, p = f_oneway(*data)
                    test_type = "Welch's ANOVA"
            else:
                stat, p = kruskal(*data)
                test_type = "Kruskal-Wallis"

            print(f"Test statistico: {test_type} (stat={stat:.3f}, p={p:.2e})")

            if p < 0.05:
                if test_type in ["ANOVA", "Welch's ANOVA"]:
                    tukey = pairwise_tukeyhsd(endog=phase_data[feature_name], groups=phase_data['Group'], alpha=0.05)
                    print("\nTest post hoc Tukey HSD:\n", tukey)
                elif test_type == "Kruskal-Wallis":
                    dunn_results = sp.posthoc_dunn(phase_data, val_col=feature_name, group_col="Group",
                                                   p_adjust="bonferroni")
                    print("\nTest post hoc Dunn con correzione Bonferroni:\n", dunn_results)

    def compare_early_late(data, feature_name):
        results = {}

        for group in data['Group'].unique():
            early_data = data[(data['Group'] == group) & (data['Phase_Assigned'] == 'Early')][feature_name].dropna()
            late_data = data[(data['Group'] == group) & (data['Phase_Assigned'] == 'Late')][feature_name].dropna()

            if len(early_data) > 0 and len(late_data) > 0:
                # Test di normalità
                normality_early_p = normaltest(early_data)[1]
                normality_late_p = normaltest(late_data)[1]

                # Test di omogeneità della varianza
                levene_p = levene(early_data, late_data)[1]
                bartlett_p = bartlett(early_data, late_data)[1]

                # Scelta del test statistico
                if normality_early_p > 0.05 and normality_late_p > 0.05:  # Dati normali
                    if levene_p > 0.05:
                        stat, p = ttest_rel(early_data, late_data)
                        test_type = "Paired t-test (Repeated Measures ANOVA)"
                    else:
                        stat, p = ttest_rel(early_data, late_data, alternative='two-sided')
                        test_type = "Welch’s t-test"
                else:
                    stat, p = wilcoxon(early_data, late_data)
                    test_type = "Wilcoxon Test"

                results[group] = {
                    "test_type": test_type,
                    "statistic": stat,
                    "p_value": p,
                    "normality_early_p": normality_early_p,
                    "normality_late_p": normality_late_p,
                    "levene_p": levene_p,
                    "bartlett_p": bartlett_p
                }

                print(f"{group}: {test_type} (stat={stat:.3f}, p={p:.2e})")
                print("Test di normalità (p-value): Early =", normality_early_p, "Late =", normality_late_p)
                print("Test di omogeneità della varianza: Levene p-value:", levene_p, "Bartlett p-value:",
                        bartlett_p)

        return results

    statistical_tests(channel_means_all, "Tutti i Canali")
    statistical_tests(channel_means_selected, "Canali Selezionati")
    compare_early_late(channel_means_all, feature_name)
    compare_early_late(channel_means_selected, feature_name)

    def plot_data():
        color_map = {'Early': 'red', 'Late': 'blue'}
        offset = {'Early': -0.2, 'Late': 0.2}
        plt.figure(figsize=(12, 7))

        for phase in ["Early", "Late"]:
            phase_data_all = channel_means_all[channel_means_all['Phase_Assigned'] == phase]
            plt.scatter(
                [group_order.index(g) + offset[phase] for g in phase_data_all['Group']],
                phase_data_all[feature_name],
                color=color_map[phase],
                label=f"{phase} - Tutti i canali",
                alpha=0.6,
                edgecolors='black',
                s=100
            )

        # Creiamo il grafico scatter per i canali selezionati
        if selected_channels is not None:
            for phase in ["Early", "Late"]:
                phase_data_selected = channel_means_selected[channel_means_selected['Phase_Assigned'] == phase]
                plt.scatter(
                    [group_order.index(g) + offset[phase] for g in phase_data_selected['Group']],
                    phase_data_selected[feature_name],
                    color='green',  # Colore per i canali selezionati
                    label=f"{phase} - Canali selezionati",
                    alpha=0.6,
                    edgecolors='black',
                    s=100
                )

            # Aggiungiamo una linea orizzontale per la media dei canali selezionati per ogni gruppo
            if not overall_means_selected.empty:
                for phase in ["Early", "Late"]:
                    phase_means_selected = overall_means_selected[overall_means_selected['Phase_Assigned'] == phase]
                    if not phase_means_selected.empty:
                        plt.hlines(
                            y=phase_means_selected[feature_name],
                            xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                            xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                            colors='green',  # Colore per la linea tratteggiata dei canali selezionati
                            linestyles='dashed',
                            linewidth=2,
                            label=f"Media Canali Selezionati {phase}"
                        )

        # Aggiungiamo una linea orizzontale per la media di tutti i canali per ogni gruppo
        if not overall_means_all.empty:
            for phase in ["Early", "Late"]:
                phase_means_all = overall_means_all[overall_means_all['Phase_Assigned'] == phase]
                if not phase_means_all.empty:
                    plt.hlines(
                        y=phase_means_all[feature_name],
                        xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                        xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                        colors=color_map[phase],
                        linestyles='dashed',
                        linewidth=2,
                        label=f"Media Tutti i Canali {phase}"
                    )

        plt.xlabel("Gruppi", fontsize=12)  # Font più piccolo per le etichette degli assi
        plt.ylabel(feature_name, fontsize=12)  # Font più piccolo per le etichette degli assi
        plt.title(f"Media delle Medie per Early e Late per Canale ({feature_name})",
                  fontsize=14)  # Font più piccolo per il titolo
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(ticks=range(len(group_order)), labels=group_order, rotation=45, fontsize=10)

        # Posizioniamo la legenda fuori dal grafico con testo più piccolo
        plt.legend(title="Legenda", bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=8)  # Font più piccolo per la legenda

        # Mostriamo il grafico
        plt.tight_layout()  # Aggiunge un padding per evitare che la legenda venga tagliata
        plt.show()

    plot_data()
    return results


def plot_selected_channels_mean_per_group(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],
                                          selected_channels=None):
    """
    Plotta un grafico scatter con la media della feature selezionata per i canali selezionati,
    raggruppata per 'Early' e 'Late' per ogni gruppo, calcolando prima la media per soggetto e canale.

    Parametri:
    - data: DataFrame con i dati.
    - feature_name: Nome della feature da plottare (colonna y).
    - group_order: Lista opzionale per ordinare i gruppi sull'asse x.
    - selected_channels: Lista di canali da includere nel grafico. Se None, non vengono plottati.
    """
    # Assicura che la colonna 'Group' sia categorica con l'ordine desiderato
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filtra i dati per le fasi "Early" e "Late"
    filtered_data = data[data['Phase_Assigned'].isin(["Early", "Late"])]

    # Se sono selezionati canali specifici, filtra i dati per quei canali
    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
    else:
        print("Devi selezionare dei canali per vedere il grafico!")
        return

    # Calcola la media delle epoche per ogni soggetto e canale
    subject_means_selected = filtered_data_selected.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Calcola la media dei soggetti per ogni gruppo, fase e canale
    channel_means_selected = subject_means_selected.groupby(['Group', 'Channel', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Calcola la media complessiva tra i canali per ogni gruppo e fase
    overall_means_selected = channel_means_selected.groupby(['Group', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Mappa colori per le fasi
    color_map = {'Early': 'red', 'Late': 'blue'}

    plt.figure(figsize=(8, 6))

    # Creiamo il grafico scatter
    for phase in ["Early", "Late"]:
        phase_data = overall_means_selected[overall_means_selected['Phase_Assigned'] == phase]
        plt.scatter(
            [group_order.index(g) for g in phase_data['Group']],
            phase_data[feature_name],
            color=color_map[phase],
            label=f"{phase}",
            s=100,
            edgecolors='black'
        )

    # Etichette e titoli
    plt.xlabel("Gruppi", fontsize=12)
    plt.ylabel(f"Media {feature_name}", fontsize=12)
    plt.title(f"Media di {feature_name} per Early e Late nei Canali Selezionati", fontsize=14)
    plt.xticks(ticks=range(len(group_order)), labels=group_order, rotation=45, fontsize=10)

    # Aggiungiamo la legenda
    plt.legend(title="Fasi", fontsize=10)

    # Mostriamo il grafico
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_early_and_late_per_channel(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=None):
    """
    Plotta un unico grafico scatter in cui ogni punto rappresenta un canale,
    con valore medio calcolato per ogni gruppo e fase ('Early' e 'Late').
    Aggiunge una linea orizzontale per ogni gruppo che rappresenta la media dei canali.
    Dispone i punti di 'Early' leggermente a sinistra e quelli di 'Late' leggermente a destra.
    Inoltre, aggiunge la media per i canali selezionati come linea tratteggiata.

    Parametri:
    - data: DataFrame con i dati.
    - feature_name: Nome della feature da plottare (colonna y).
    - group_order: Lista opzionale per ordinare i gruppi sull'asse x.
    - selected_channels: Lista di canali da includere nel grafico. Se None, include tutti i canali.
    """
    # Assicura che la colonna 'Group' sia categorica con l'ordine desiderato
    data['Group'] = pd.Categorical(data['Group'], categories=group_order)

    # Filtra i dati per le fasi "Early" e "Late"
    filtered_data = data[data['Phase_Assigned'].isin(["Early", "Late"])]

    # Se sono selezionati canali specifici, filtra i dati per quei canali
    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        filtered_data_all = filtered_data  # Tutti i canali
    else:
        filtered_data_selected = pd.DataFrame()  # Se non ci sono canali selezionati, nessun filtro
        filtered_data_all = filtered_data  # Tutti i canali

    # Calcola la media delle epoche per ogni soggetto e canale
    subject_means_all = filtered_data_all.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
    subject_means_selected = filtered_data_selected.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Calcola la media dei soggetti per ogni gruppo, fase e canale
    channel_means_all = subject_means_all.groupby(['Group', 'Channel', 'Phase_Assigned'], observed=True)[feature_name].mean().reset_index()
    channel_means_selected = subject_means_selected.groupby(['Group', 'Channel', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Calcola la media complessiva tra i canali per ogni gruppo e fase
    overall_means_all = channel_means_all.groupby(['Group', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()
    overall_means_selected = channel_means_selected.groupby(['Group', 'Phase_Assigned'],observed=True)[feature_name].mean().reset_index()

    # Mappa colori per le fasi
    color_map = {'Early': 'red', 'Late': 'blue'}
    offset = {'Early': -0.2, 'Late': 0.2}  # Offset per spostare i punti leggermente

    plt.figure(figsize=(10, 6))

    # Creiamo il grafico scatter per i singoli canali (Tutti i canali)
    for phase in ["Early", "Late"]:
        phase_data_all = channel_means_all[channel_means_all['Phase_Assigned'] == phase]
        plt.scatter(
            [group_order.index(g) + offset[phase] for g in phase_data_all['Group']],
            phase_data_all[feature_name],
            color=color_map[phase],
            label=f"{phase} - Tutti i canali",
            alpha=0.6,
            edgecolors='black',
            s=20
        )

    # Creiamo il grafico scatter per i canali selezionati
    if selected_channels is not None:
        for phase in ["Early", "Late"]:
            phase_data_selected = channel_means_selected[channel_means_selected['Phase_Assigned'] == phase]
            plt.scatter(
                [group_order.index(g) + offset[phase] for g in phase_data_selected['Group']],
                phase_data_selected[feature_name],
                color='green',  # Colore per i canali selezionati
                label=f"{phase} - Canali selezionati",
                alpha=0.6,
                edgecolors='black',
                s=20
            )

        # Aggiungiamo una linea orizzontale per la media dei canali selezionati per ogni gruppo
        if not overall_means_selected.empty:
            for phase in ["Early", "Late"]:
                phase_means_selected = overall_means_selected[overall_means_selected['Phase_Assigned'] == phase]
                if not phase_means_selected.empty:
                    plt.hlines(
                        y=phase_means_selected[feature_name],
                        xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                        xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                        colors='green',  # Colore per la linea tratteggiata dei canali selezionati
                        linestyles='dashed',
                        linewidth=2,
                        label=f"Media Canali Selezionati {phase}"
                    )

    # Aggiungiamo una linea orizzontale per la media di tutti i canali per ogni gruppo
    if not overall_means_all.empty:
        for phase in ["Early", "Late"]:
            phase_means_all = overall_means_all[overall_means_all['Phase_Assigned'] == phase]
            if not phase_means_all.empty:
                plt.hlines(
                    y=phase_means_all[feature_name],
                    xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                    xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                    colors=color_map[phase],
                    linestyles='dashed',
                    linewidth=2,
                    label=f"Media Tutti i Canali {phase}"
                )

    plt.xlabel("Gruppi", fontsize=12)  # Font più piccolo per le etichette degli assi
    plt.ylabel(feature_name, fontsize=12)  # Font più piccolo per le etichette degli assi
    plt.title(f"Media delle Medie per Early e Late per Canale ({feature_name})", fontsize=14)  # Font più piccolo per il titolo
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(ticks=range(len(group_order)), labels=group_order, rotation=45, fontsize=10)

    # Posizioniamo la legenda fuori dal grafico con testo più piccolo
    plt.legend(title="Legenda", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Font più piccolo per la legenda

    # Mostriamo il grafico
    plt.tight_layout()  # Aggiunge un padding per evitare che la legenda venga tagliata
    plt.show()

'''
def plot_early_and_late_per_channel(data, feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=None):
    """
    Plotta un unico grafico scatter in cui ogni punto rappresenta un canale,
    con valore medio calcolato per ogni gruppo e fase ('Early' e 'Late').
    Aggiunge una linea orizzontale per ogni gruppo che rappresenta la media dei canali.
    Dispone i punti di 'Early' leggermente a sinistra e quelli di 'Late' leggermente a destra.
    Inoltre, aggiunge la media per i canali selezionati come linea tratteggiata.

    Parametri:
    - data: DataFrame con i dati.
    - feature_name: Nome della feature da plottare (colonna y).
    - group_order: Lista opzionale per ordinare i gruppi sull'asse x.
    - selected_channels: Lista di canali da includere nel grafico. Se None, include tutti i canali.
    """
    # Assicura che la colonna 'Group' sia categorica con l'ordine desiderato
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filtra i dati per le fasi "Early" e "Late"
    filtered_data = data[data['Phase_Assigned'].isin(["Early", "Late"])]

    # Se sono selezionati canali specifici, filtra i dati per quei canali
    if selected_channels is not None:
        filtered_data_selected = filtered_data[filtered_data['Channel'].isin(selected_channels)]
        filtered_data_all = filtered_data  # Tutti i canali
    else:
        filtered_data_selected = pd.DataFrame()  # Se non ci sono canali selezionati, nessun filtro
        filtered_data_all = filtered_data  # Tutti i canali

    # Calcola la media delle epoche per ogni soggetto e canale
    subject_means_all = filtered_data_all.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'])[feature_name].mean().reset_index()
    subject_means_selected = filtered_data_selected.groupby(['Group', 'Channel', 'Subject', 'Phase_Assigned'])[feature_name].mean().reset_index()

    # Calcola la media dei soggetti per ogni gruppo, fase e canale
    channel_means_all = subject_means_all.groupby(['Group', 'Channel', 'Phase_Assigned'])[feature_name].mean().reset_index()
    channel_means_selected = subject_means_selected.groupby(['Group', 'Channel', 'Phase_Assigned'])[feature_name].mean().reset_index()

    # Calcola la media complessiva tra i canali per ogni gruppo e fase
    overall_means_all = channel_means_all.groupby(['Group', 'Phase_Assigned'])[feature_name].mean().reset_index()
    overall_means_selected = channel_means_selected.groupby(['Group', 'Phase_Assigned'])[feature_name].mean().reset_index()

    # Mappa colori per le fasi
    color_map = {'Early': 'red', 'Late': 'blue'}
    offset = {'Early': -0.2, 'Late': 0.2}  # Offset per spostare i punti leggermente

    plt.figure(figsize=(10, 6))

    # Creiamo il grafico scatter per i singoli canali (Tutti i canali)
    for phase in ["Early", "Late"]:
        phase_data_all = channel_means_all[channel_means_all['Phase_Assigned'] == phase]
        plt.scatter(
            [group_order.index(g) + offset[phase] for g in phase_data_all['Group']],
            phase_data_all[feature_name],
            color=color_map[phase],
            label=f"{phase} - Tutti i canali",
            alpha=0.6,
            edgecolors='black',
            s=80
        )

    # Creiamo il grafico scatter per i canali selezionati
    if selected_channels is not None:
        for phase in ["Early", "Late"]:
            phase_data_selected = channel_means_selected[channel_means_selected['Phase_Assigned'] == phase]
            plt.scatter(
                [group_order.index(g) + offset[phase] for g in phase_data_selected['Group']],
                phase_data_selected[feature_name],
                color='green',  # Colore per i canali selezionati
                label=f"{phase} - Canali selezionati",
                alpha=0.6,
                edgecolors='black',
                s=100
            )

        # Aggiungiamo una linea orizzontale per la media dei canali selezionati per ogni gruppo
        if not overall_means_selected.empty:
            for phase in ["Early", "Late"]:
                phase_means_selected = overall_means_selected[overall_means_selected['Phase_Assigned'] == phase]
                if not phase_means_selected.empty:
                    plt.hlines(
                        y=phase_means_selected[feature_name],
                        xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                        xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                        colors='green',  # Colore per la linea tratteggiata dei canali selezionati
                        linestyles='dashed',
                        linewidth=2,
                        label=f"Media Canali Selezionati {phase}"
                    )

    # Aggiungiamo una linea orizzontale per la media di tutti i canali per ogni gruppo
    if not overall_means_all.empty:
        for phase in ["Early", "Late"]:
            phase_means_all = overall_means_all[overall_means_all['Phase_Assigned'] == phase]
            if not phase_means_all.empty:
                plt.hlines(
                    y=phase_means_all[feature_name],
                    xmin=np.arange(len(group_order)) + offset[phase] - 0.1,
                    xmax=np.arange(len(group_order)) + offset[phase] + 0.1,
                    colors=color_map[phase],
                    linestyles='dashed',
                    linewidth=2,
                    label=f"Media Tutti i Canali {phase}"
                )

    plt.xlabel("Gruppi", fontsize=12)  # Font più piccolo per le etichette degli assi
    plt.ylabel(feature_name, fontsize=12)  # Font più piccolo per le etichette degli assi
    plt.title(f"Media delle Medie per Early e Late per Canale ({feature_name})", fontsize=14)  # Font più piccolo per il titolo
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(ticks=range(len(group_order)), labels=group_order, rotation=45, fontsize=10)

    # Posizioniamo la legenda fuori dal grafico con testo più piccolo
    plt.legend(title="Legenda", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Font più piccolo per la legenda

    # Mostriamo il grafico
    plt.tight_layout()  # Aggiunge un padding per evitare che la legenda venga tagliata
    plt.show()
'''

# Carica i dati
file_path = r"D:\TESI\prova statistica\N2N3FILTERS\_no_mean_N2N3FILTERS_specific_channels_150_aggregated_with_phases.csv"
data = pd.read_csv(file_path)
data = data[data['Stage'] == 3]

feature_to_analyze = ['0. Absolute Low Delta Power']

'''
# Ciclo per analizzare ogni caratteristica
for feature in feature_to_analyze:
    print(f"\n### Analisi per caratteristica: {feature} ###")
    results = analyze_feature(data, feature)
'''
# Seleziona solo alcuni canali
selected_channels = [27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222, 16, 22, 23, 24, 28, 29, 30, 35, 36, 40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                                   207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]  # Aggiungi i canali di interesse
#selected_channels=None
# Genera il grafico solo per i canali selezionati
for feature_name in feature_to_analyze:
    #analyze_and_plot(data=data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=selected_channels)
    plot_selected_channels_mean_per_group(data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'], selected_channels=selected_channels)
    plot_early_and_late_per_channel(data.copy(), feature_name=feature_name, group_order=['CTL', 'DNV', 'ADV', 'DYS' ], selected_channels=selected_channels)