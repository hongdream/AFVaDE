import os
import scipy.io
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained


def make_dataset(csv_path, n_clients=2, batch_size=128, dataname="iris", seed=42):
    path = os.path.join(".", "npz", f"{dataname}.npz")

    L = []
    loadData = np.load(path, allow_pickle=True)
    for key in loadData:
        client = loadData[key].tolist()
        L.append(client)

    clients_tensor_data = []
    size = []

    for l in L:
        X_client = torch.tensor(l['data'], dtype=torch.float32)
        y_client = torch.tensor(l['label'], dtype=torch.int)
        size.append(l['size'])

        dataset = TensorDataset(X_client, y_client)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        clients_tensor_data.append(dataloader)

    file_path = os.path.join(".", "csv", f"{dataname}.csv")
    data = pd.read_csv(file_path, header=0).values
    data, label = data[:, :-1], data[:, -1].astype(int)

    X_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(label, dtype=torch.long)

    all_clusters = len(np.unique(label))

    return all_clusters, clients_tensor_data, (X_tensor, y_tensor), size