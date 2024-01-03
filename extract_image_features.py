# -*- coding: utf-8 -*-
"""
Usage:
    python script_name.py --root_dir <path_to_root_dir> --survival_file <path_to_survival_file> --model_name <model_name>

Author: raktim
Date: June 29, 2023
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycox.models import CoxPH
from torchvision import models
from argparse import ArgumentParser

from macenko_color_normalizer import MacenkoColorNormalization

# PatientDataset class
class PatientDataset(Dataset):
    def __init__(self, root_dir, patient_ids, survival_file, transform=None, color_normalization=None):
        self.root_dir = root_dir
        # or index_col = 0
        self.survival_data = pd.read_csv(survival_file, index_col='PATIENT_ID')
        # Convert to numeric

        # list patient folders and sort them
        self.patient_ids = patient_ids
        
        
        self.model_transform = transform 
        
        
        self.color_norm = color_normalization
        
        self.preprocess = transforms.Compose([
            self.color_norm,  # Ensure this is a valid transformation
            # transforms.Resize((224, 224)),  # Uncomment if resizing is needed
            #transforms.ToTensor(), # when model specific transform is used make sure it is commented
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.model_transform  # Include this if it's a valid transformation 
        ])
        


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        # load all images for this patient
        patient_id = self.patient_ids[index]

        # list all image paths for this patient and sort them
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, patient_id, '*.png')))
        
        
        images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            img = self.preprocess(img)
            images.append(img)
        
        # stack all images into a tensor
        images = torch.stack(images)

        # load the survival data for this patient
        survival_time = self.survival_data.loc[patient_id, 'OS_YEARS']
        event = self.survival_data.loc[patient_id, 'OS_STATUS']

        survival_time = torch.tensor(survival_time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.float32)

        return images, (survival_time, event)

# Function to create and parse command-line arguments
def parse_args():
    parser = ArgumentParser(description="Process medical images for survival analysis.")
    parser.add_argument("--root_dir", required=True, help="Path to the root directory of the dataset.")
    parser.add_argument("--survival_file", required=True, help="Path to the survival data CSV file.")
    parser.add_argument("--model_name", required=True, help="Name of the pretrained model to use.")
    return parser.parse_args()

# Main function to execute the script logic
def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize MacenkoColorNormalization
    macenko_color_normalization = MacenkoColorNormalization()

    # Load data
    train_data = pd.read_csv(os.path.join(args.root_dir, 'train_tcga.txt'), index_col='patient_id')
    test_data = pd.read_csv(os.path.join(args.root_dir, 'test_tcga.txt'), index_col='patient_id')
    all_data = pd.read_csv(os.path.join(args.root_dir, 'all_patient.txt'), index_col='patient_id')

    train_patient_ids = train_data.index.astype(str).tolist()
    test_patient_ids = test_data.index.astype(str).tolist()
    all_patient_ids = all_data.index.astype(str).tolist()

    # Load model
    model = timm.create_model(
        model_name=args.model_name,
        pretrained=True,
    ).eval()

    # Model configuration
    data_config = timm.data.resolve_model_data_config(model)
    model_trans = timm.data.create_transform(**data_config, is_training=False)

    # Set up device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = model.to(device)

    # Initialize dataset
    dataset = PatientDataset(root_dir=args.root_dir, patient_ids=all_patient_ids, 
                             survival_file=args.survival_file, transform=model_trans, 
                             color_normalization=macenko_color_normalization)

    patch_group_size = 50
    max_patches_per_patient = 500

    # Process each patient in the dataset
    for index in range(len(dataset)):
        # Get all images for this patient
        all_images, _ = dataset[index]  # We only need the images here, not the survival data
        patient_id = dataset.patient_ids[index]

        all_features = []

        # Process images in groups to manage memory usage
        for i in range(0, min(len(all_images), max_patches_per_patient), patch_group_size):
            # Select a group of images
            image_group = all_images[i:i + patch_group_size].to(device)

            # Extract features without calculating gradients
            with torch.no_grad():
                features = feature_extractor(image_group)

            # Collect the features from the current group
            all_features.append(features.cpu())

        # Concatenate all features for this patient
        all_features = torch.cat(all_features, dim=0)

        # Save the features to disk
        # Ensure the save directory exists
        save_dir = os.path.join(args.root_dir, 'patient_features')
        os.makedirs(save_dir, exist_ok=True)

        # Save features as a tensor
        save_path = os.path.join(save_dir, f'{patient_id}_features.pt')
        torch.save(all_features, save_path)

        print(f"Processed and saved features for patient {patient_id}")

if __name__ == "__main__":
    main()
