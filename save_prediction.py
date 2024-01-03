import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from patient_dataset import PatientDataset, collate_fn
from multimodal_model import MultimodalModel
from make_predictions import make_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Model Prediction')
    parser.add_argument('--train_data_txt', required=True, help='Path to text file with training patient IDs')
    parser.add_argument('--test_data_txt', required=True, help='Path to text file with testing patient IDs')
    parser.add_argument('--survival_data', required=True, help='Path to survival data CSV file')
    parser.add_argument('--gene_expression_data', required=True, help='Path to gene expression data CSV file')
    parser.add_argument('--clinical_data', required=True, help='Path to clinical data CSV file')
    parser.add_argument('--features_dir', required=True, help='Path to directory containing VAE extracted features')
    parser.add_argument('--model_path', required=True, help='Path to the trained model file')
    parser.add_argument('--feat_out', type=int, default=128, help='Output size of the feature layer in the model')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load patient IDs
    train_patient_ids = pd.read_csv(args.train_data_txt, index_col='patient_id').index.astype(str).tolist()
    test_patient_ids = pd.read_csv(args.test_data_txt, index_col='patient_id').index.astype(str).tolist()

    # Create datasets
    train_dataset = PatientDataset(patient_ids=train_patient_ids, survival_file=args.survival_data, 
                                   gene_expression_file=args.gene_expression_data, clinical_file=args.clinical_data,
                                   features_dir=args.features_dir)
    test_dataset = PatientDataset(patient_ids=test_patient_ids, survival_file=args.survival_data, 
                                  gene_expression_file=args.gene_expression_data, clinical_file=args.clinical_data,
                                  features_dir=args.features_dir)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=12, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=12, collate_fn=collate_fn)

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel(feat_out=args.feat_out, output_dim=128)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # Make predictions for training and test data
    make_predictions(model, train_dataloader, device, "training_data_predictions.txt")
    make_predictions(model, test_dataloader, device, "test_data_predictions.txt")

if __name__ == "__main__":
    main()
