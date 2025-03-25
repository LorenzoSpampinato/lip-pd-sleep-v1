import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import normaltest, kruskal, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway, wilcoxon, kruskal, normaltest, levene, bartlett, fligner, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from scipy.stats import shapiro

def analyze_feature(df, feature):
    results = {}

    for channel in df['Channel'].unique():
        print(f"\n //////////////////Processing Channel: {channel}//////////////////")
        channel_data = df[df['Channel'] == channel]

        for phase in ['Early', 'Late']:
            print(f"Processing Phase: {phase}")
            phase_data = channel_data[channel_data['Phase_Assigned'] == phase]
            print(f"Number of rows in phase data: {len(phase_data)}")

            groups = phase_data['Group'].unique()
            print(f"Groups: {groups}")
            data = [phase_data[phase_data['Group'] == group].groupby('Subject')[feature].mean().dropna() for group in groups]

            # Statistiche descrittive
            descriptive_stats = phase_data.groupby('Group')[feature].describe()
            print(f"Descriptive Stats for {feature}:\n", descriptive_stats)

            # Test di normalità
            normality = {group: shapiro(group_data)[1] if len(group_data) >= 3 else None for group, group_data in zip(groups, data)}
            print(f"Normality Test Results: {normality}")

            if all(p > 0.05 for p in normality.values() if p is not None):
                levene_p = levene(*data)[1]
                bartlett_p = bartlett(*data)[1]
                var_test = "Levene" if levene_p < 0.05 else "Bartlett"
                var_p = levene_p if levene_p < 0.05 else bartlett_p
            else:
                fligner_p = fligner(*data)[1]
                var_test = "Fligner-Killeen"
                var_p = fligner_p

            print(f"Test di varianza: {var_test} (p={var_p:.2e})")

            if all(p > 0.05 for p in normality.values() if p is not None):
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

            # Test post-hoc
            post_hoc = None
            if p < 0.05:
                combined_data = phase_data[[feature, 'Group']].dropna()
                if test_type == "ANOVA":
                    post_hoc = pairwise_tukeyhsd(combined_data[feature], combined_data['Group'])
                    print(f"Tukey PostHoc Results:\n", post_hoc)
                else:
                    post_hoc = sp.posthoc_dunn(combined_data, val_col=feature, group_col='Group', p_adjust='bonferroni')
                    print(f"Dunn PostHoc Results:\n", post_hoc)

            results[(channel, phase)] = {
                'Descriptive': descriptive_stats,
                'Normality': normality,
                'Test': (test_type, stat, p),
                'PostHoc': post_hoc
            }

        # Confronto tra Early vs. Late per ogni gruppo
        for group in ['CTL', 'DNV', 'ADV', 'DYS']:
            early_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Early')].groupby('Subject')[feature].mean().dropna()
            late_group = channel_data[(channel_data['Group'] == group) & (channel_data['Phase_Assigned'] == 'Late')].groupby('Subject')[feature].mean().dropna()

            print(f"Comparing Early vs Late for {group}")

            if not early_group.empty and not late_group.empty:
                normality_early = shapiro(early_group)[1] if len(early_group) >= 3 else None
                normality_late = shapiro(late_group)[1] if len(late_group) >= 3 else None

                print(f"Normality Test Early: {normality_early}, Late: {normality_late}")

                if (normality_early is None or normality_early > 0.05) and (normality_late is None or normality_late > 0.05):
                    stat, p = f_oneway(early_group, late_group)
                    test_type = "ANOVA"
                    print(f"ANOVA Early vs Late: stat={stat}, p={p}")
                else:
                    stat, p = kruskal(early_group, late_group)
                    test_type = "Kruskal-Wallis"
                    print(f"Kruskal-Wallis Early vs Late: stat={stat}, p={p}")

                results[(channel, f'{group} Early vs Late')] = {
                    'Normality': {f'{group} Early': normality_early, f'{group} Late': normality_late},
                    'Test': (test_type, stat, p)
                }

    return results



def print_results(results):
    """
    Stampa i risultati dell'analisi in modo organizzato.
    """
    for key, value in results.items():
        channel, phase = key

        if phase in ['Early', 'Late']:
            print(f"\n===== Analisi di {channel} - {phase} =====")
            print("Statistiche descrittive:\n", value['Descriptive'])
            print("\nTest di normalità:", value['Normality'])
            print("\nTest statistico:", value['Test'])

            # Verifica il tipo di test post-hoc e stampa i risultati correttamente
            post_hoc = value['PostHoc']
            if post_hoc is not None:
                if hasattr(post_hoc, 'summary'):  # Se è un risultato Tukey HSD
                    print("\nTest post-hoc (Tukey HSD):\n", post_hoc.summary())
                else:
                    print("\nTest post-hoc (Dunn con correzione Bonferroni):\n", post_hoc)

        else:  # Confronto Early vs. Late per ogni gruppo
            print(f"\n===== Confronto {phase} per {channel} =====")
            print("\nTest di normalità:", value['Normality'])
            print("\nTest statistico:", value['Test'])


# Creare una mappa di colori globale per i soggetti
import seaborn as sns


def create_subject_color_map(data):
    unique_groups = data['Group'].unique()  # Prendi i gruppi (es. 'CTL', 'DNV', etc.)
    subject_color_map = {}

    for group in unique_groups:
        subjects_in_group = sorted(data[data['Group'] == group]['Subject'].unique())
        num_subjects = len(subjects_in_group)

        # Genera colori distinti per ogni gruppo
        group_colors = sns.color_palette("tab20", n_colors=num_subjects)

        # Mappa soggetti a colori
        for subject, color in zip(subjects_in_group, group_colors):
            subject_color_map[subject] = color

    return subject_color_map


def plot_feature_per_patient_violin_and_sd_subplot(data, feature_name, channel_name, group_order=None):
    """
    Crea un grafico con subplot organizzati per gruppo.
    Ogni sottotrama mostra:
    - Violin plot della distribuzione per Early e Late, separati.
    - Media e deviazione standard.
    I pazienti dello stesso gruppo sono disposti sulla stessa riga.
    Il titolo di ogni sottotrama include il paziente e il gruppo di appartenenza.
    Include una legenda per distinguere Early e Late.

    Parameters:
        data (pd.DataFrame): Contiene le colonne 'Group', 'Subject', 'Channel', 'Phase_Assigned', e la caratteristica specificata.
        feature_name (str): Il nome della caratteristica da plottare.
        channel_name (str): Nome del canale da analizzare.
        group_order (list): Ordine esplicito dei gruppi (opzionale).
    """
    if group_order:
        data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filtra i dati per il canale specificato
    channel_data = data[data['Channel'] == channel_name]

    # Ottieni i gruppi unici e i pazienti in ogni gruppo
    groups = channel_data['Group'].unique()
    group_to_patients = {group: channel_data[channel_data['Group'] == group]['Subject'].unique() for group in groups}

    # Numero massimo di pazienti in un gruppo
    max_patients_per_group = max(len(patients) for patients in group_to_patients.values())

    # Crea una griglia: righe = gruppi, colonne = massimo numero di pazienti in un gruppo
    n_rows = len(groups)
    n_cols = max_patients_per_group

    # Imposta la figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows), sharey=True)

    # Se c'è solo un gruppo o un paziente, axes potrebbe non essere una matrice
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    # Colori per le fasi
    phase_colors = {'Early': 'skyblue', 'Late': 'salmon'}

    # Loop sui gruppi e pazienti
    for row, group in enumerate(groups):
        patients = group_to_patients[group]

        for col in range(n_cols):
            ax = axes[row][col]

            # Se il numero di pazienti è minore del massimo, lascia i subplot vuoti
            if col >= len(patients):
                ax.axis('off')
                continue

            patient = patients[col]
            patient_data = channel_data[channel_data['Subject'] == patient]

            # Separare i dati per fase
            early_data = patient_data[patient_data['Phase_Assigned'] == 'Early']
            late_data = patient_data[patient_data['Phase_Assigned'] == 'Late']

            # Plot dei violin plot separati
            if not early_data.empty:
                sns.violinplot(
                    x=[0] * len(early_data),  # Posizione sfalsata per Early
                    y=early_data[feature_name],
                    inner='box',
                    color=phase_colors['Early'],
                    width=0.6,
                    ax=ax,
                    alpha=0.6
                )
            if not late_data.empty:
                sns.violinplot(
                    x=[1] * len(late_data),  # Posizione sfalsata per Late
                    y=late_data[feature_name],
                    inner='box',
                    color=phase_colors['Late'],
                    width=0.6,
                    ax=ax,
                    alpha=0.6
                )

            # Aggiungi media e deviazione standard per Early e Late
            for phase, phase_data, offset in zip(
                ['Early', 'Late'], [early_data, late_data], [0, 1]
            ):
                if not phase_data.empty:
                    mean_val = phase_data[feature_name].mean()
                    std_val = phase_data[feature_name].std()
                    ax.errorbar(
                        x=offset,
                        y=mean_val,
                        yerr=std_val,
                        fmt='o',
                        color='black',
                        alpha=0.8
                    )

            # Impostazioni asse
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Early', 'Late'])
            ax.set_title(f"Patient {patient} ({group})", fontsize=10)

        # Etichetta per i gruppi
        axes[row][0].set_ylabel(group, fontsize=14, labelpad=20)

    # Legenda per i colori delle fasi
    handles = [
        plt.Line2D([0], [0], color=phase_colors['Early'], lw=4, label='Early'),
        plt.Line2D([0], [0], color=phase_colors['Late'], lw=4, label='Late')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=12, title="Phase")

    # Etichette comuni
    fig.supxlabel("Phase", fontsize=14)
    fig.supylabel(feature_name, fontsize=14)
    fig.suptitle(f"Distribuzione, Media e SD per paziente nel canale: {channel_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Lascia spazio per il titolo principale
    plt.show()



def plot_early_and_late_barplots_minmaxscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply MinMaxScaler to the entire dataset**
    scaler = MinMaxScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: MinMaxScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_early_and_late_barplots_standardscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply StandardScaler to the entire dataset**
    scaler = StandardScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: StandardScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_early_and_late_barplots_robustscaler(data, feature_name, channel_name, group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Creates and displays barplots for the "Early" and "Late" phases, grouping patients by group.
    The barplots represent the group means, and colored dots represent the individual patient means.

    Parameters:
    - data: DataFrame with the data.
    - feature_name: Name of the feature to plot (y column).
    - channel_name: Channel to plot.
    - group_order: Optional list to order the groups on the x-axis.
    """
    # **Apply RobustScaler to the entire dataset**
    scaler = RobustScaler()
    data[feature_name] = scaler.fit_transform(data[[feature_name]])

    # Ensure that the 'Group' column is categorical with the desired order
    data['Group'] = pd.Categorical(data['Group'], categories=group_order, ordered=True)

    # Filter data by the specified channel
    channel_data = data[data['Channel'] == channel_name]
    #channel_data = data[data['Brain region'] == channel_name]

    # Filter data by "Early" and "Late" phases
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Plot creation
    plt.figure(figsize=(12, 6))
    x_offset = 0.3  # Distance between Early and Late bars
    tick_positions = []
    tick_labels = []

    color_map = subject_color_map

    # Variable to handle patient legends (to avoid duplicates)
    patient_legend_handles = []

    # Plot data for each group
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()

        # **Barplot for group means**
        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.5, label="Early" if group_idx == 0 else "",
            zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.5, label="Late" if group_idx == 0 else "",
            zorder=3
        )

        # Add dots for each patient with a unique color
        for idx, subject in enumerate(early_means.index):
            plt.scatter(
                [group_idx * 2 - x_offset], [early_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        for idx, subject in enumerate(late_means.index):
            plt.scatter(
                [group_idx * 2 + x_offset], [late_means.loc[subject]], color=color_map[subject], zorder=4, alpha=1,
                edgecolor='black', linewidth=1.5, s=70
            )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    # Customizing the X-axis
    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Groups", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Barplot Early and Late ({feature_name}) - Channel: {channel_name} - Scaler: RobustScaler", fontsize=16)

    # Create a unique legend for the patients (to avoid duplicates)
    for subject, color in color_map.items():
        patient_legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=subject))

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_early_and_late_phases_grouped_ordered(data, feature_name, channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=None):
    """
    Crea e mostra i boxplot per le fasi "Early" e "Late", raggruppando i pazienti per gruppo, per un singolo canale.
    Ogni grafico rappresenta un canale specifico, con boxplot per le fasi "Early" e "Late", e barplot per la media dei gruppi
    con intervallo SEM. Ora include anche la media di ciascun paziente.

    Parametri:
    - data: DataFrame con i dati.
    - feature_name: Nome della feature da plottare (colonna y).
    - channel_name: Canale da plottare.
    - group_order: Lista opzionale per ordinare i gruppi sull'asse x.
    """

    # Assicurati che la colonna 'Group' sia categorica con l'ordine desiderato
    data['Group'] = pd.Categorical(data['Group'], categories=group_order)
    print(data.iloc[:6, 7])
    print(data.shape)  # Verifica se ha righe e colonne
    print(data.head())  # Controlla i primi valori

    # Filtra i dati per il canale specifico
    channel_data = data[data['Channel'] == channel_name]

    #channel_data = data[data['Brain region'] == channel_name]

    # Filtra i dati per le fasi "Early" e "Late"
    combined_data = channel_data[channel_data['Phase_Assigned'].isin(["Early", "Late"])]

    # Conta il numero di epoche Early e Late per ogni soggetto
    epoch_counts = combined_data.groupby(['Subject', 'Phase_Assigned']).size().unstack(fill_value=0)

    # Stampa il numero di epoche per ogni soggetto
    print("\nNumero di epoche per ciascun soggetto:")
    print(epoch_counts)

    # Trova gli indici delle epoche Early e Late per ogni soggetto
    early_indices = combined_data[combined_data['Phase_Assigned'] == "Early"].groupby('Subject').apply(
        lambda x: list(x.index))
    late_indices = combined_data[combined_data['Phase_Assigned'] == "Late"].groupby('Subject').apply(
        lambda x: list(x.index))

    # Stampa gli indici delle epoche
    print("\nIndici delle epoche Early per ciascun soggetto:")
    print(early_indices)
    print("\nIndici delle epoche Late per ciascun soggetto:")
    print(late_indices)

    color_map = subject_color_map

    # Creazione del grafico
    plt.figure(figsize=(16, 8))
    x_offset = 0.4
    tick_positions = []
    tick_labels = []

    # Traccia i dati per ogni gruppo
    for group_idx, group in enumerate(combined_data['Group'].cat.categories):
        group_data = combined_data[combined_data['Group'] == group]
        group_data = group_data.sort_values(by=["Subject"])

        early_data = group_data[group_data['Phase_Assigned'] == "Early"]
        late_data = group_data[group_data['Phase_Assigned'] == "Late"]

        early_means = early_data.groupby('Subject')[feature_name].mean()
        late_means = late_data.groupby('Subject')[feature_name].mean()

        early_group_mean = early_means.mean()
        late_group_mean = late_means.mean()
        early_sem = early_means.sem()
        late_sem = late_means.sem()

        sns.boxplot(
            x=np.full(len(early_data), group_idx * 2 - x_offset),
            y=early_data[feature_name],
            hue=early_data['Subject'],
            data=early_data,
            palette=color_map,
            dodge=True,
            width=0.6,
            boxprops=dict(alpha=0.8),
            showfliers=False
        )

        sns.boxplot(
            x=np.full(len(late_data), group_idx * 2 + x_offset),
            y=late_data[feature_name],
            hue=late_data['Subject'],
            data=late_data,
            palette=color_map,
            dodge=True,
            width=0.6,
            boxprops=dict(alpha=0.8),
            showfliers=False
        )

        plt.bar(
            group_idx * 2 - x_offset, early_group_mean, color='red', width=0.2, zorder=3
        )
        plt.bar(
            group_idx * 2 + x_offset, late_group_mean, color='blue', width=0.2, zorder=3
        )

        plt.errorbar(
            group_idx * 2 - x_offset, early_group_mean, yerr=early_sem, fmt='none', color='black', capsize=5, zorder=4
        )
        plt.errorbar(
            group_idx * 2 + x_offset, late_group_mean, yerr=late_sem, fmt='none', color='black', capsize=5, zorder=4
        )

        plt.scatter(
            np.full(len(early_means), group_idx * 2 - x_offset),
            early_means.values,
            color=[color_map[subj] for subj in early_means.index],
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
            s=50
        )

        plt.scatter(
            np.full(len(late_means), group_idx * 2 + x_offset),
            late_means.values,
            color=[color_map[subj] for subj in late_means.index],
            edgecolor='black',
            linewidth=0.5,
            zorder=5,
            s=50
        )

        tick_positions.append(group_idx * 2)
        tick_labels.append(group)

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=12)
    plt.xlabel("Gruppi", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.title(f"Boxplot Early e Late per gruppo ({feature_name}) - Canale: {channel_name}", fontsize=16)

    # Creiamo un dizionario per raggruppare i pazienti per gruppo
    grouped_legends = {}
    for subject, color in subject_color_map.items():
        patient_group = data[data["Subject"] == subject]["Group"].iloc[0]  # Trova il gruppo del paziente
        if patient_group not in grouped_legends:
            grouped_legends[patient_group] = []
        grouped_legends[patient_group].append((subject, color))

    # Creiamo la legenda ordinata per gruppi
    legend_handles = []
    for group in group_order:  # Seguiamo l'ordine definito nei gruppi
        if group in grouped_legends:
            legend_handles.append(
                plt.Line2D([0], [0], color='black', lw=0, label=f"Patients {group}:"))  # Titolo del gruppo
            for subject, color in grouped_legends[group]:
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=subject))

    # Aggiungiamo la legenda Early/Late
    legend_handles.append(plt.Line2D([0], [0], color='red', lw=4, label='Early'))
    legend_handles.append(plt.Line2D([0], [0], color='blue', lw=4, label='Late'))

    # Impostiamo la legenda finale
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title="Patients")

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()



# Carica i dati
file_path = r"D:\TESI\prova statistica\N2N3spectral150_0.5Hz\_no_mean_N2N3spectral_0.5Hz_specific_channels_150_aggregated_with_phases.csv"
data = pd.read_csv(file_path)
#data = data[data['Stage'] == 3]


feature_to_analyze = ['0. Absolute Low Delta Power']
#results = analyze_feature(data, '0. Absolute Low Delta Power')
#print_results(results)



'''
# Ciclo per analizzare ogni caratteristica
for feature in feature_to_analyze:
    print(f"\n### Analisi per caratteristica: {feature} ###")
    results = analyze_feature(data, feature)
'''
# Specifica i canali selezionati per i grafici (uno o più)
#selected_channels = ['F']  # Aggiungi tutti i canali di interesse
selected_channels = [27]
subject_color_map = create_subject_color_map(data)

# Genera i grafici per ogni canale e caratteristica
for channel_name in selected_channels:
    for feature_name in feature_to_analyze:
        print(f"\n### Grafico per caratteristica: {feature_name}, Canale: {channel_name} ###")
        # Grafico 2: Subplot organizzati per pazienti e gruppo
        #plot_feature_per_patient_violin_and_sd_subplot(data,feature_name=feature_name,channel_name=channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'])
        plot_early_and_late_phases_grouped_ordered(data.copy(),feature_name=feature_name,channel_name=channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_early_and_late_barplots_minmaxscaler(data.copy(), feature_name=feature_name, channel_name=channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        plot_early_and_late_barplots_standardscaler(data.copy(), feature_name=feature_name, channel_name=channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)
        #plot_early_and_late_barplots_robustscaler(data.copy(), feature_name=feature_name, channel_name=channel_name,group_order=['CTL', 'DNV', 'ADV', 'DYS'],subject_color_map=subject_color_map)

