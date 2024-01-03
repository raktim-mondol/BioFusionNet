
import os
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader

def to_structured_array(events, times):
    dtype = [('event', bool), ('time', float)]
    return np.array(list(zip(events, times)), dtype=dtype)

def make_predictions(model, dataloader, device, output_file):
    all_predictions, all_times, all_events = [], [], []

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write("Sample, risk_score\n")

        for pid, features, (time, event), gene, clinical in dataloader:
            predictions, times, events = [], [], []

            for pid, feat, tm, et, gn, cl in zip(pid, features, time, event, gene, clinical):
                feat, gn, cl = feat.to(device), gn.to(device), cl.to(device)
                cl = cl.unsqueeze(0)
                feat = feat.unsqueeze(0)
                gn = gn.unsqueeze(0)

                with torch.no_grad():
                    prediction = model(feat, gn, cl)

                f.write(f"{pid}, {prediction.item()}\n")

                predictions.append(prediction)
                times.append(tm.to(device).unsqueeze(0))
                events.append(et.to(device).unsqueeze(0))

            all_predictions.append(torch.cat(predictions).detach().cpu())
            all_times.append(torch.cat(times).detach().cpu())
            all_events.append(torch.cat(events).detach().cpu())

    all_predictions = torch.cat(all_predictions).squeeze()
    all_times = torch.cat(all_times).squeeze()
    all_events = torch.cat(all_events).squeeze()

    bool_events = all_events.numpy() == 1
    final_c_index, _, _, _, _ = concordance_index_censored(bool_events, all_times.numpy(), all_predictions.numpy())
    print(f'C-index: {final_c_index}')




