import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    def __init__(self, patient_ids, survival_file, gene_expression_file, clinical_file, features_dir):
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
        feature_path = os.path.join(self.features_dir, f'{patient_id}.pt')
        
        #print(features.shape)

        # Load the gene expression data for this patient
        gene_expression = self.gene_expression_data.loc[patient_id]
        gene_expression = torch.tensor(gene_expression.values, dtype=torch.float32)
        # Commenting out the log transformation; you can uncomment if needed
        #gene_expression = torch.log2(1 + gene_expression)

        return patient_id, (survival_time, event), features, gene_expression, clinical_data


def collate_fn(batch):
    # similar to train_collate_fn
    patient_ids, survival_data, features, gene_expression, clinical_data = zip(*batch)
    durations = torch.stack([data[0] for data in survival_data])
    events = torch.stack([data[1] for data in survival_data])

    return patient_ids, features, (durations, events), gene_expression, clinical_data
