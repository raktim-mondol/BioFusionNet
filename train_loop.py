# train_model.py
import torch
import csv
import pandas as pd
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from loss_function import loss_fn  

def to_structured_array(events, times):
    dtype = [('event', bool), ('time', float)]
    return np.array(list(zip(events, times)), dtype=dtype)

def train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs, device, event_weight, patience, model_save_path, log_path, survival_data_path):
    # Load survival data
    survival_data = pd.read_csv(survival_data_path, index_col='PATIENT_ID')
    structured_survival_data = to_structured_array(survival_data['OS_STATUS'], survival_data['OS_MONTHS'])

    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Training C-index", "Validation Loss", "Validation C-index", "Validation AUC"])

        best_loss = float('inf')
        epochs_without_improvement = 0
        model.train()

        for epoch in range(num_epochs):
            train_loss, num_batches = 0.0, 0
            all_predictions, all_times, all_events = [], [], []

            # Training Loop
            for pid, features, (time, event), gene, clinical in train_dataloader:
                weights_train = torch.ones(len(event)) * event_weight
                optimizer.zero_grad()
                predictions, times, events = [], [], []

                for pid, feat, tm, et, gn, cl in zip(pid, features, time, event, gene, clinical):
                    feat, gn, cl = feat.to(device), gn.to(device), cl.to(device)
                    prediction = model(feat.unsqueeze(0), gn.unsqueeze(0), cl.unsqueeze(0)).squeeze()
                    predictions.append(prediction)
                    times.append(tm.unsqueeze(0))
                    events.append(et.unsqueeze(0))

                predictions = torch.cat(predictions)
                times = torch.cat(times)
                events = torch.cat(events)
                loss = loss_fn(predictions, times, events, weights_train)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches
            train_c_index = concordance_index_censored(all_events.numpy() == 1, all_times.numpy(), all_predictions.numpy())[0]

            # Validation Loop
            model.eval()
            val_loss, num_val_batches = 0.0, 0
            all_val_predictions, all_val_times, all_val_events = [], [], []

            for pid, features, (time, event), gene, clinical in test_dataloader:
                weights_test = torch.ones(len(event)) * event_weight
                predictions, times, events = [], [], []

                for pid, feat, tm, et, gn, cl in zip(pid, features, time, event, gene, clinical):
                    feat, gn, cl = feat.to(device), gn.to(device), cl.to(device)
                    with torch.no_grad():
                        prediction = model(feat.unsqueeze(0), gn.unsqueeze(0), cl.unsqueeze(0)).squeeze()
                    predictions.append(prediction)
                    times.append(tm.unsqueeze(0))
                    events.append(et.unsqueeze(0))

                predictions = torch.cat(predictions)
                times = torch.cat(times)
                events = torch.cat(events)
                val_loss += loss_fn(predictions, times, events, weights_test).item()
                num_val_batches += 1

            val_loss /= num_val_batches
            val_c_index = concordance_index_censored(all_val_events.numpy() == 1, all_val_times.numpy(), all_val_predictions.numpy())[0]

            # AUC Calculation for Test Data
            time_points = np.array([5, 10])  # Time points for AUC calculation
            val_auc = cumulative_dynamic_auc(structured_survival_data, to_structured_array(all_val_events.numpy(), all_val_times.numpy()), all_val_predictions.numpy(), time_points)

            # Logging
            print(f'Epoch {epoch}: Training Loss: {train_loss:.4f}, C-index: {train_c_index:.4f}, Validation Loss: {val_loss:.4f}, C-index: {val_c_index:.4f}, Validation AUC: {val_auc}')
            writer.writerow([epoch, train_loss, train_c_index, val_loss, val_c_index, val_auc])

            # Save model if improvement
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved at epoch {epoch} with validation loss {val_loss}')
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'No improvement in validation loss for {patience} epochs, stopping training')
                break
