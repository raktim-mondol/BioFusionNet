import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def plot_survival_curves(args):
    # Load the data
    data = pd.read_csv(args.data_file)

    # Initialize the Kaplan-Meier estimator
    kmf = KaplanMeierFitter()

    # Define common time points for a 10-year range
    common_time_points = np.linspace(0, 10, 1000)

    plt.figure(figsize=(12, 8))

    # Define color schemes for high risk and low risk
    colors = {
        'High Risk': ['#FFC0CB', '#FF69B4'],
        'Low Risk': ['#90EE90', '#3CB371']
    }

    for risk_group in ['High Risk', 'Low Risk']:
        survival_curves = []

        for fold in data['Fold'].unique():
            fold_data = data[(data['Fold'] == fold) & (data['Risk Group'] == risk_group)]
            kmf.fit(fold_data['Duration'], event_observed=fold_data['Event'])
            
            interpolated_curve = np.interp(common_time_points, kmf.survival_function_.index, kmf.survival_function_.values.flatten(), left=1)
            survival_curves.append(interpolated_curve)

        survival_curves = np.array(survival_curves)
        mean_curve = np.mean(survival_curves, axis=0)
        lower_bound = np.percentile(survival_curves, 2.5, axis=0)
        upper_bound = np.percentile(survival_curves, 97.5, axis=0)

        plt.plot(common_time_points, mean_curve, label=f"{risk_group} (Mean)", color=colors[risk_group][1], linewidth=2)
        plt.fill_between(common_time_points, lower_bound, upper_bound, color=colors[risk_group][0], alpha=0.5, label=f"{risk_group} 95% CI")

    # Perform the log rank test
    T_high_risk = data[data['Risk Group'] == 'High Risk']['Duration']
    E_high_risk = data[data['Risk Group'] == 'High Risk']['Event']
    T_low_risk = data[data['Risk Group'] == 'Low Risk']['Duration']
    E_low_risk = data[data['Risk Group'] == 'Low Risk']['Event']
    results = logrank_test(T_high_risk, T_low_risk, event_observed_A=E_high_risk, event_observed_B=E_low_risk)
    formatted_p_value = "{:.2e}".format(results.p_value)

    # Plot settings
    plt.xlabel("Duration (Years)", fontsize=18)
    plt.ylabel("Survival Probability", fontsize=18)
    plt.title("Mean Survival Curve with 95% Confidence Intervals", fontsize=20)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.figtext(0.35, 0.1, f'p-value: {formatted_p_value} (Log Rank Test)', horizontalalignment='left', fontsize=16, color='black')

    # Save the figure with 300 dpi
    plt.savefig(args.output_file, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Survival Analysis Plots")
    parser.add_argument("--data_file", required=True, help="Path to the CSV file containing the combined extracted data")
    parser.add_argument("--output_file", required=True, help="Path to save the output plot")

    args = parser.parse_args()
    plot_survival_curves(args)
