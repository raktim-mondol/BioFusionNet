# train_model.py
import torch
import csv
from sksurv.metrics import concordance_index_censored
from loss_function import loss_fn  # Assuming loss_fn is in a separate file

def to_structured_array(events, times):
    dtype = [('event', bool), ('time', float)]
    return np.array(list(zip(events, times)), dtype=dtype)

def train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs, device, event_weight, patience, model_save_path, log_path):
    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Training C-index", "Validation Loss", "Validation C-index"])

        best_loss = float('inf')
        epochs_without_improvement = 0
        model.train()

        for epoch in range(num_epochs):
            train_loss = 0.0
            num_batches = 0
            all_predictions, all_times, all_events = [], [], []

            for pid, features, (time, event), gene, clinical in train_dataloader:
                weights_train = torch.ones(len(event))
                weights_train[event == 1] = event_weight
                
                optimizer.zero_grad()
                predictions, times, events = [], [], []

                for pid, feat, tm, et, gn, cl in zip(pid, features, time, event, gene, clinical):
                    feat, gn, cl = feat.to(device), gn.to(device), cl.to(device)
                    cl = cl.unsqueeze(0)
                    feat = feat.unsqueeze(0)
                    gn = gn.unsqueeze(0)
                    prediction = model(feat, gn, cl)
                    time = tm.to(device)
                    event = et.to(device)
                    prediction = prediction.squeeze().unsqueeze(0)
                    if torch.isnan(prediction).any():
                        print("NaN values detected in predictions.")
                    
                    predictions.append(prediction)
                    times.append(time.unsqueeze(0))
                    events.append(event.unsqueeze(0))

                predictions = torch.cat(predictions)
                times = torch.cat(times)
                events = torch.cat(events)

                loss = loss_fn(predictions, times, events, weights_train)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                all_predictions.append(predictions.detach().cpu())
                all_times.append(times.detach().cpu())
                all_events.append(events.detach().cpu())

            train_loss /= num_batches
            all_predictions = torch.cat(all_predictions)
            all_times = torch.cat(all_times)
            all_events = torch.cat(all_events)
            bool_events = all_events.numpy() == 1 
            survival_train = to_structured_array(bool_events, all_times.numpy())
            train_c_index, _, _, _, _ = concordance_index_censored(bool_events, all_times.numpy(), all_predictions.numpy())

            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            all_val_predictions, all_val_times, all_val_events = [], [], []
            
            for pid, features, (time, event), gene, clinical in test_dataloader:
                predictions, times, events = [], [], []
                weights_test = torch.ones(len(event))
                weights_test[event == 1] = event_weight

                for pid, feat, tm, et, gn, cl in zip(pid, features, time, event, gene, clinical):
                    feat, gn, cl = feat.to(device), gn.to(device), cl.to(device)
                    with torch.no_grad():
                        prediction = model(feat.unsqueeze(0), gn.unsqueeze(0), cl.unsqueeze(0))
                    predictions.append(prediction.squeeze().unsqueeze(0))
                    times.append(tm.to(device).unsqueeze(0))
                    events.append(et.to(device).unsqueeze(0))

                predictions = torch.cat(predictions)
                times = torch.cat(times)
                events = torch.cat(events)
                val_loss += loss_fn(predictions, times, events, weights_test).item()
                num_val_batches += 1
                all_val_predictions.append(predictions.detach().cpu())
                all_val_times.append(times.detach().cpu())
                all_val_events.append(events.detach().cpu())

            val_loss /= num_val_batches
            all_val_predictions = torch.cat(all_val_predictions)
            all_val_times = torch.cat(all_val_times)
            all_val_events = torch.cat(all_val_events)
            bool_val_events = all_val_events.numpy() == 1
            val_c_index, _, _, _, _ = concordance_index_censored(bool_val_events, all_val_times.numpy(), all_val_predictions.numpy())

            # Logging
            print(f'Epoch {epoch} - Training Loss: {train_loss:.4f}, Training C-index: {train_c_index:.4f}, Validation Loss: {val_loss:.4f}, Validation C-index: {val_c_index:.4f}')
            writer.writerow([epoch, train_loss, train_c_index, val_loss, val_c_index])

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
