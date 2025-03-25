import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests


# Funzione per il test di permutazione tra Early e Late
def permutation_test_statistic(x, y, n_permutations=5000):
    res = permutation_test((x, y), statistic=lambda x, y: np.mean(x) - np.mean(y),
                           n_resamples=n_permutations, alternative="two-sided")
    return res.pvalue


def permutation_test_statistic(x, y, n_permutations=5000):
    res = permutation_test((x, y), statistic=lambda x, y: np.mean(x) - np.mean(y),
                           n_resamples=n_permutations, alternative="two-sided")
    return res.pvalue

import numpy as np
import pandas as pd
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests

def permutation_test_statistic(x, y, n_permutations=5000):
    res = permutation_test((x, y), statistic=lambda x, y: np.mean(x) - np.mean(y),
                           n_resamples=n_permutations, alternative="two-sided")
    return res.pvalue

def permutation_test_per_channel(data, feature_name, selected_channels, n_permutations=5000):
    results = []

    for channel in selected_channels:
        for group in data['Group'].unique():
            early = data[(data['Channel'] == channel) & (data['Group'] == group) & (data['Phase_Assigned'] == 'Early')][feature_name].dropna()
            late = data[(data['Channel'] == channel) & (data['Group'] == group) & (data['Phase_Assigned'] == 'Late')][feature_name].dropna()

            if len(early) > 0 and len(late) > 0:
                p_value = permutation_test_statistic(early, late, n_permutations)
                results.append({'Channel': channel, 'Group': group, 'p_value': p_value})

    results_df = pd.DataFrame(results)

    # Correzione per test multipli con Benjamini-Hochberg (FDR)
    if not results_df.empty:
        corrected_p = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df['p_value_corrected'] = corrected_p
        results_df['Significant'] = results_df['p_value_corrected'] < 0.05

    return results_df


# Funzione per plottare i canali selezionati
def plot_selected_channels_mean_per_group(data, feature_name, group_order, selected_channels):
    plt.figure(figsize=(10, 6))
    color_map = {'Early': 'red', 'Late': 'blue'}

    for phase in ["Early", "Late"]:
        phase_data = data[(data['Phase_Assigned'] == phase) & (data['Channel'].isin(selected_channels))]
        means = phase_data.groupby(['Group', 'Channel'], observed=True)[feature_name].mean().reset_index()

        plt.scatter(
            [group_order.index(g) for g in means['Group']],
            means[feature_name],
            color=color_map[phase],
            label=f"{phase} - Canali Selezionati",
            alpha=0.6,
            edgecolors='black',
            s=100
        )

    plt.xlabel("Gruppi", fontsize=12)
    plt.ylabel(feature_name, fontsize=12)
    plt.title(f"Media per Early e Late - Canali Selezionati ({feature_name})", fontsize=14)
    plt.xticks(ticks=range(len(group_order)), labels=group_order, rotation=45, fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


# Main per eseguire il test e il grafico
def main():
    file_path = r"C:\Users\Lorenzo\Desktop\prova statistica\N2N3spectral150\_no_mean_N2N3meanspectralnorm1_specific_channels_150_aggregated_with_phases.csv"
    data = pd.read_csv(file_path)
    data = data[data['Stage'] == 3]

    feature_to_analyze = ['46. Absolute Power at 2 Hz']
    selected_channels = [27, 33, 34, 38, 39, 47, 48, 26, 20, 19, 12, 11, 3, 2, 222, 16, 22, 23, 24, 28, 29, 30, 35, 36,
                         40, 41, 42, 49, 50, 21, 15, 7, 14, 6,
                         207, 13, 5, 215, 4, 224, 223, 214, 206, 213, 205]

    for feature_name in feature_to_analyze:
        print(f"\n### Analisi per caratteristica: {feature_name} ###")
        results = permutation_test_per_channel(data, feature_name, selected_channels)
        print(results)

        # Plotta i canali selezionati
        plot_selected_channels_mean_per_group(data, feature_name, ['CTL', 'DNV', 'ADV', 'DYS'], selected_channels)
        significant_channels = results[results['Significant'] == True]
        print(significant_channels[['Channel', 'Group', 'p_value_corrected']])
        grouped_significant = significant_channels.groupby('Group')['Channel'].agg(list).reset_index()
        grouped_significant['Num_Significant_Channels'] = grouped_significant['Channel'].apply(len)
        print(grouped_significant)


# Esegui il main
if __name__ == "__main__":
    main()
