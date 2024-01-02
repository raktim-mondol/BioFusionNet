# BioFusionNet: Survival Risk Stratification through Multi-Feature and Multi-Modal Data Fusion

![MAGNet Logo](BioFusionNet.png) <!-- If you have a logo or relevant image -->

BioFusionNet is a deep learning framework that integrates multimodal data sources using an co dual attention mechanism, specifically designed for survival analysis tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Multimodal Integration**: Seamlessly combines data from various sources.
- **Attention-Guided Mechanism**: Focuses on the most relevant features for prediction.
- **Designed for Survival Analysis**: Optimized for predicting time-to-event data.

## Installation
First Install Required Libraries
```bash
pip install -r requirements.txt
```
Extracting Image Features
```bash
python extracting_features.py --train_file path/to/training_patient_id.txt \
               --val_file path/to/val_patient_id.txt \
               --batch_size 16 \
               --learning_rate 0.001 \
               --epochs 100 \
               --patience 5 \
               --save_dir path/to/save_fused_features

```


Training VAE
```bash
python training_VAE.py --train_file path/to/training_patient_id.txt \
               --val_file path/to/val_patient_id.txt \
               --batch_size 16 \
               --learning_rate 0.001 \
               --epochs 100 \
               --patience 5 \
               --save_dir path/to/save_fused_features

```
Training Risk Model
```bash
python training_risk_model.py --train_file path/to/training_patient_id.txt \
               --val_file path/to/val_patient_id.txt \
               --batch_size 16 \
               --learning_rate 0.001 \
               --epochs 100 \
               --patience 5 \
               --save_dir path/to/save_fused_features

```


## Proposed Loss Function
```python
def loss_fn(risks, times, events, weights):
    """
    Calculate the Cox proportional hazards loss with weights for imbalance.

    Parameters:
    - risks: Tensor of predicted risk scores (log hazard ratio) from the model.
    - events: Tensor of event indicators (1 if event occurred, 0 for censored).
    - weights: Tensor of weights for each sample.

    Returns:
    - Calculated loss.
    
    """
    

    risks = risks.to(device)
    events = events.to(device)
    weights = weights.to(device)
    
    
    events = events.view(-1)
    risks = risks.view(-1)
    weights = weights.view(-1)
    

    total_weighted_events = torch.sum(weights * events)

    # Sort by risk score
    order = torch.argsort(risks, descending=True)
    risks = risks[order]
    events = events[order]
    weights = weights[order]

    # Calculate the risk set for each time
    hazard_ratio = torch.exp(risks)
    weighted_cumulative_hazard = torch.cumsum(weights * hazard_ratio, dim=0)
    log_risk = torch.log(weighted_cumulative_hazard)
    uncensored_likelihood = weights * (risks - log_risk)

    # Only consider uncensored events
    censored_likelihood = uncensored_likelihood * events
    neg_likelihood = -torch.sum(censored_likelihood) / total_weighted_events

    return neg_likelihood
```

## Dataset
Describe the dataset used and how to access or download it.

## Results

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.


## Contributing
We welcome contributions! Please see our CONTRIBUTING.md for details.


## Acknowledgments

-List of contributors or institutions.
-Any third-party resources or datasets used.
