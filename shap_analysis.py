import argparse
import torch
import shap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, image_features, gene_expression, clinical_data):
        return self.model(image_features, gene_expression, clinical_data)

def parse_args():
    parser = argparse.ArgumentParser(description='SHAP Analysis for Multimodal Model')
    parser.add_argument('--analysis_type', choices=['gene', 'clinical', 'image'], required=True, help='Type of SHAP analysis to perform: "gene", "clinical", or "image"')
    parser.add_argument('--list_path', required=True, help='Path to file containing gene names, clinical features, or image feature file')
    return parser.parse_args()

def read_csv_list(file_path):
    return pd.read_csv(file_path, header=None).squeeze().tolist()

def load_image_features(file_path):
    return torch.load(file_path)

def main():
    args = parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel(feat_out=128, output_dim=128)  # Adjust parameters as per your model
    model.load_state_dict(torch.load('saved_model.pt'))
    model.to(device)
    model.eval()

    # Load a batch of data
    data = next(iter(train_dataloader))
    (features_samples, gene_expression_tensor, clinical_data_tensor), _ = data

    # Convert data to the appropriate device
    images = features_samples.to(device)
    genes = gene_expression_tensor.to(device)
    clinical = clinical_data_tensor.to(device)

    # Initialize ModelWrapper
    model_wrapper = ModelWrapper(model)

    # Create DeepExplainer
    explainer = shap.DeepExplainer(model_wrapper, [images, genes, clinical])

    # Compute SHAP values
    shap_values = explainer.shap_values([images, genes, clinical])

    # Visualization based on user's choice
    matplotlib.use('Agg')
    if args.analysis_type in ['gene', 'clinical']:
        features_list = read_csv_list(args.list_path)
        features_array = genes.cpu().numpy() if args.analysis_type == 'gene' else clinical.cpu().numpy()
        shap.summary_plot(shap_values[1 if args.analysis_type == 'gene' else 2], features_array, feature_names=features_list)
        plt.savefig(f'shap_{args.analysis_type}.png', dpi=300)
    elif args.analysis_type == 'image':
        image_features = load_image_features(args.list_path)
        shap.image_plot(shap_values[0], image_features.cpu().numpy())
        plt.savefig('shap_plot_image.png', dpi=300)

if __name__ == "__main__":
    main()
