from make_predictions import make_predictions
from patient_dataset import PatientDataset, collate_fn
from multimodal_model import MultimodalModel
import torch

# Load trained model
model = MultimodalModel(feat_out=size_val, output_dim=128)
model.load_state_dict(torch.load('cv_1_saved_model.pt'))
model.eval()

# Assuming you have initialized train_dataloader and test_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Make predictions for training data
make_predictions(model, train_dataloader, device, "training_data_predictions.txt")

# Make predictions for test data
make_predictions(model, test_dataloader, device, "test_data_predictions.txt")
