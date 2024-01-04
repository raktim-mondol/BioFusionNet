import argparse
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def survival_analysis(args):
    np.random.seed(42)
    data = pd.read_csv(args.data_file, index_col='SAMPLE_ID')
    combined_data = pd.DataFrame()

    for fold in range(1, args.num_folds + 1):
        train_patient = pd.read_csv(args.train_id_template.format(fold), index_col='patient_id')
        test_patient = pd.read_csv(args.test_id_template.format(fold), index_col='patient_id')

        x_train = data.loc[train_patient.index, :]
        x_test = data.loc[test_patient.index, :]

        train_predicted_risks = pd.read_csv(args.training_data_predictions_template.format(fold), index_col='Sample')
        test_predicted_risks = pd.read_csv(args.test_data_predictions_template.format(fold), index_col='Sample')

        times = x_test['TIME']
        events = x_test['EVENT']

        median_risk_threshold = np.median(train_predicted_risks)
        high_risk_indices = np.where(test_predicted_risks > median_risk_threshold)[0]
        low_risk_indices = np.where(test_predicted_risks <= median_risk_threshold)[0]

        # Kaplan-Meier Curves
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(10, 7))
        kmf.fit(times[high_risk_indices], event_observed=events[high_risk_indices], label="High Risk")
        kmf.plot(ax=plt.gca(), ci_show=False, color="red")
        kmf.fit(times[low_risk_indices], event_observed=events[low_risk_indices], label="Low Risk")
        kmf.plot(ax=plt.gca(), ci_show=False, color="blue")
        plt.title(f"Kaplan-Meier Survival Curves for High and Low Risk Groups - Fold {fold}")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.xlim(0, 10)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f'{args.output_folder}/survival_curves_fold_{fold}.png')
        plt.close()

        # Log-rank Test
        results = logrank_test(times[high_risk_indices], times[low_risk_indices],
                               event_observed_A=events[high_risk_indices],
                               event_observed_B=events[low_risk_indices])
        print(f"Log-rank Test p-value for Fold {fold}: {results.p_value}")

        # Extracting Data
        df_extracted = pd.DataFrame(index=test_predicted_risks.index)
        df_extracted['Duration'] = times
        df_extracted['Event'] = events
        df_extracted['Risk Group'] = np.where(test_predicted_risks.values.flatten() > median_risk_threshold, "High Risk", "Low Risk")
        df_extracted['Fold'] = f'Fold {fold}'
        df_extracted.set_index(['Fold', df_extracted.index], inplace=True)
        df_extracted.index.names = ['Fold', 'Sample']

        combined_data = combined_data.append(df_extracted)

    combined_csv_path = f"{args.output_folder}/combined_extracted_data.csv"
    combined_data.to_csv(combined_csv_path, index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Survival Analysis with Command-Line Arguments")
    parser.add_argument("--data_file", required=True, help="Path to the clinical survival CSV file")
    parser.add_argument("--train_id_template", required=True, help="Template for train ID file paths")
    parser.add_argument("--test_id_template", required=True, help="Template for test ID file paths")
    parser.add_argument("--training_data_predictions_template", required=True, help="Training data predictions file paths")
    parser.add_argument("--test_data_predictions_template", required=True, help="Test data predictions file paths")
    parser.add_argument("--output_folder", required=True, help="Folder to save output files")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds to process")

    args = parser.parse_args()
    survival_analysis(args)
