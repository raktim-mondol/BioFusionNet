# -*- coding: utf-8 -*-
"""
main.py - Main script for training and evaluating VAE model for gene prediction.
"""

# Standard library imports
import argparse
import os
import random
import numpy as np
import pandas as pd
import csv
import math

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# PyCox for survival analysis models
from pycox.models import CoxPH
from pycox.models.loss import CoxPHLoss

# Imports for image processing
from PIL import Image
import torchvision.transforms as transforms

# Imports for machine learning and data processing
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from lifelines.utils import concordance_index

from VAE import VAE

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_2d_tensors(tensors):
    # Find the maximum height (first dimension) among the tensors
    max_height = max(tensor.shape[0] for tensor in tensors)
    
    # Pad each tensor to have the same height as the max_height
    padded_tensors = []
    for tensor in tensors:
        pad_amount = max_height - tensor.shape[0]
        # We're padding only at the bottom (height) here
        padded_tensor = F.pad(tensor, (0, 0, 0, pad_amount))
        padded_tensors.append(padded_tensor)
    
    return torch.stack(padded_tensors)

def custom_normalization(data):
    # Reshape data to 2D - each row is a patch
    num_samples, num_patches, num_features = data.shape
    data_reshaped = data.reshape(num_samples * num_patches, num_features)

    # Calculate mean and std
    mean = data_reshaped.mean(dim=0)
    std = data_reshaped.std(dim=0)

    # Normalize
    normalized_data_reshaped = (data_reshaped - mean) / std

    # Reshape back to original shape
    normalized_data = normalized_data_reshaped.reshape(num_samples, num_patches, num_features)
    return normalized_data

# with clinical data
class PatientDataset(Dataset):
    def __init__(self, patient_ids, survival_file, gene_expression_file, clinical_file):
        # Load the survival data
        self.survival_data = pd.read_csv(survival_file, index_col='PATIENT_ID')
        self.patient_ids = patient_ids
        # Load the gene expression data
        self.gene_expression_data = pd.read_csv(gene_expression_file, index_col='PATIENT_ID')
        # Load and extract the desired clinical columns
        self.clinical_data = pd.read_csv(clinical_file, index_col='PATIENT_ID')
        #self.clinical_data = self.clinical_data[["Subtype", "Grade", "Age", "LN_Status_new", "Tumor_Size_new"]]
        self.clinical_data = self.clinical_data[["Grade", "Age", "LN_Status_new", "Tumor_Size_new"]]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        # Get the patient id for this index
        patient_id = self.patient_ids[index]

        # Load the survival data for this patient
        survival_time = self.survival_data.loc[patient_id, 'OS_MONTHS']
        survival_time = survival_time / 12.0
        event = self.survival_data.loc[patient_id, 'OS_STATUS']
        survival_time = torch.tensor(survival_time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.float32)
        
        # Load clinical features for this patient
        clinical_data = self.clinical_data.loc[patient_id].values
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        
        # Load features for this patient
        features_1 = torch.load(f'/scratch/nk53/rm8989/gene_prediction/code/self_supervised_training/lunit_dino/{patient_id}.pt')
        features_2 = torch.load(f'/scratch/nk53/rm8989/gene_prediction/code/self_supervised_training/mocov3/{patient_id}.pt')
        features_3 = torch.load(f'/scratch/nk53/rm8989/gene_prediction/code/self_supervised_training/brca_dino/{patient_id}.pt')
        
        # Load the gene expression data for this patient
        gene_expression = self.gene_expression_data.loc[patient_id]
        gene_expression = torch.tensor(gene_expression.values, dtype=torch.float32)
        # Commenting out the log transformation; you can uncomment if needed
        #gene_expression = torch.log2(1 + gene_expression)
        
        concatenated_features = torch.cat((features_1, features_2, features_3), dim=1)
        
        
        return patient_id, (survival_time, event), (features_1,features_2,features_3), gene_expression, clinical_data

def collate_fn(batch):
    # similar to train_collate_fn
    patient_ids, survival_data, features, gene_expression, clinical_data = zip(*batch)
    
    #durations = torch.stack([data[0] for data in survival_data])
    #events = torch.stack([data[1] for data in survival_data])
    
    features_1, features_2, features_3 = zip(*features)
    
    features_padded_1 = pad_2d_tensors(features_1)
    
    features_padded_2 = pad_2d_tensors(features_2)
    
    features_padded_3 = pad_2d_tensors(features_3)
    
    concatenated_features = torch.cat([features_padded_1, features_padded_2, features_padded_3], dim=-1)
    
    concatenated_features = custom_normalization(concatenated_features)
    
    return patient_ids, concatenated_features



def load_patient_data(patient_file, batch_size=12):
    data = pd.read_csv(patient_file, index_col='patient_id')
    patient_ids = data.index.astype(str).tolist()
    
    #patient_ids = [pid for pid in patient_ids if len(glob.glob(os.path.join(root_dir, pid, '*.png'))) >=500]

    dataset = PatientDataset(patient_ids=patient_ids, survival_file=SURVIVAL_LOC, 
                             gene_expression_file=GENE_LOC, clinical_file=CLINICAL_LOC)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader
model = VAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            # Save the model when validation loss decreases
            torch.save(model.state_dict(), 'vae_fused_training.pt')
            self.val_loss_min = val_loss

def train(model, train_loader, val_loader, optimizer, epochs, log_interval=10, patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_mse_loss = 0
        total_kld_loss = 0
    
        for pid, data in train_loader:
    
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, MSE, KLD = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_mse_loss += MSE.item()
            total_kld_loss += KLD.item()
              
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_mse = 0
        total_val_kld = 0
        with torch.no_grad():
            for pid, data in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                total_loss, MSE, KLD = loss_function(recon_batch, data, mu, logvar)
    
                total_val_loss += total_loss.item()
                total_val_mse += MSE.item()
                total_val_kld += KLD.item()
    
        val_loss = total_val_loss / len(val_loader.dataset)
        avg_val_mse = total_val_mse / len(val_loader.dataset)
        avg_val_kld = total_val_kld / len(val_loader.dataset)       
        
        print(f'Epoch: {epoch} | Training Loss: {train_loss / len(train_loader.dataset):.4f}, KLD: {total_kld_loss / len(train_loader.dataset):.4f} | '
      f' Validation Loss: {val_loss / len(val_loader.dataset):.4f}, KLD: {avg_val_kld:.4f}')

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def save_fused_features(data_loader, model, device, save_directory):
    model.eval()
    with torch.no_grad():
        for patient_ids, batch_data in data_loader:
            batch_data = batch_data.to(device)
            #print(batch_data.shape)
            # Extract fused features
            mu, logvar = model.encode(batch_data)
            fused_features = model.reparameterize(mu, logvar)
            #print(fused_features.shape)
            # Convert to CPU
            fused_features_cpu = fused_features.cpu()

            # Save features for each patient
            for pid, features in zip(patient_ids, fused_features_cpu):
                #print(pid)
                #print(features.shape)
                patient_save_path = f"{save_directory}/{pid}.pt"
                torch.save(features, patient_save_path)

# Main function to parse arguments and run training
def main(args):
    set_seeds()  # Setting global seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model and optimizer
    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Load training and validation data
    train_loader = load_patient_data(args.train_file, batch_size=args.batch_size)
    val_loader = load_patient_data(args.val_file, batch_size=args.batch_size)

    # Train the model
    train(model, train_loader, val_loader, optimizer, epochs=args.epochs, patience=args.patience)

    # [Optional] Evaluate the model
    # validate(model, val_loader)
    save_fused_features(train_loader, model, device, args.save_dir)
    save_fused_features(val_loader, model, device, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate VAE Model for Gene Prediction")

    # Define command line arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training data file')
    parser.add_argument('--val_file', type=str, required=True, help='Path to the validation data file')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save fused features')

    args = parser.parse_args()
    main(args)
