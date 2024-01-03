import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from patient_dataset import PatientDataset, collate_fn
from multimodal_model import MultimodalModel
from train_loop import train_model
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Model Training for Patient Survival Prediction')
    parser.add_argument('--train_data_txt', required=True, help='Path to text file with training patient IDs')
    parser.add_argument('--test_data_txt', required=True, help='Path to text file with testing patient IDs')
    parser.add_argument('--survival_data', required=True, help='Path to survival data CSV file')
    parser.add_argument('--gene_expression_data', required=True, help='Path to gene expression data CSV file')
    parser.add_argument('--clinical_data', required=True, help='Path to clinical data CSV file')
    parser.add_argument('--features_dir', required=True, help='Path to directory containing VAE extracted features')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load patient IDs from text files
    train_patient_ids = pd.read_csv(args.train_data_txt, index_col='patient_id').index.astype(str).tolist()
    test_patient_ids = pd.read_csv(args.test_data_txt, index_col='patient_id').index.astype(str).tolist()

    # Create datasets
    train_dataset = PatientDataset(patient_ids=train_patient_ids, survival_file=args.survival_data, gene_expression_file=args.gene_expression_data, clinical_file=args.clinical_data, features_dir=args.features_dir)
    test_dataset = PatientDataset(patient_ids=test_patient_ids, survival_file=args.survival_data, gene_expression_file=args.gene_expression_data, clinical_file=args.clinical_data, features_dir=args.features_dir)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=12, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=12, collate_fn=collate_fn)

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, features, _, _ = train_dataset[0]
    size_val = features[0].shape[0]
    model = MultimodalModel(feat_out=size_val, output_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Call train_model function
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        num_epochs=1000,
        device=device,
        event_weight=5,
        patience=10,
        model_save_path='cv_1_saved_model.pt',
        log_path='training_log.csv'
    )

if __name__ == "__main__":
    main()
