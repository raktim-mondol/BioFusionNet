# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:58:07 2022

@author: Raktim
"""

import argparse
import pandas as pd
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt

def run_cox_proportional_hazards_model(args):
    file_data = pd.read_csv(args.data_file)

    columns_to_include = args.columns.split(',')
    subset_data = file_data[columns_to_include]

    cph = CoxPHFitter()
    cph.fit(subset_data, 'TIME', 'EVENT')
    cph.print_summary()
    cph.plot()

    if args.output_file:
        plt.savefig(args.output_file)
        print(f"Plot saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cox Proportional Hazards Model")
    parser.add_argument("--data_file", required=True, help="Path to the CSV file containing the data")
    parser.add_argument("--columns", required=True, help="Comma-separated list of columns to include in the model (e.g., 'TIME,EVENT,Grade,Tumor_Size')")
    parser.add_argument("--output_file", help="Path to save the output plot (optional)")

    args = parser.parse_args()
    run_cox_proportional_hazards_model(args)
