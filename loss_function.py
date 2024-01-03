import torch

def loss_fn(risks, events, weights, device):
    """
    Calculate the Cox proportional hazards loss with weights for imbalance.

    Parameters:
    - risks: Tensor of predicted risk scores (log hazard ratio) from the model.
    - events: Tensor of event indicators (1 if event occurred, 0 for censored).
    - weights: Tensor of weights for each sample.
    - device: The device (CPU or GPU) on which tensors are allocated.

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
