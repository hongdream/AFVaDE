import os
from pathlib import Path

import scipy.io
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler

ASYNFED_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ASYNFED_DIR / "Dataset"

try:


    from k_means_constrained import KMeansConstrained
except ImportError:
    KMeansConstrained = None


def make_noniid_clients_sample(csv_path, n_clients=2, batch_size=128, dataname="iris", seed=42):
    rng = np.random.RandomState(seed)

    data = pd.read_csv(csv_path, header=0).values

    X, y = data[:, :-1], data[:, -1].astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    all_clusters = np.unique(y)
    n_clusters = len(all_clusters)
    clients_tensor_data = []
    clients_numpy_data = []
    size = []

    for client_id in range(n_clients):
        n_clusters_client = rng.randint(n_clusters // 2, n_clusters + 1)
        chosen_clusters = rng.choice(all_clusters, n_clusters_client, replace=False)

        client_idx = []
        for c in chosen_clusters:
            cluster_idx = np.where(y == c)[0]
            rng.shuffle(cluster_idx)

            ratio = rng.uniform(0.25, 0.75)
            split_point = int(len(cluster_idx) * ratio)
            client_idx.extend(cluster_idx[:split_point])

        client_idx = np.array(client_idx)

        X_client = torch.tensor(X[client_idx], dtype=torch.float32)
        y_client = torch.tensor(y[client_idx], dtype=torch.long)
        size.append(len(y_client))

        dataset = TensorDataset(X_client, y_client)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        clients_tensor_data.append(dataloader)


        client = {}
        client["data"] = X[client_idx].astype(np.float32)
        client["label"] = y[client_idx].astype(np.int64)
        client["size"] = len(y[client_idx])
        clients_numpy_data.append(client)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    np.savez(f"./npz_sample/{dataname}.npz", *clients_numpy_data)

    data_to_save = {
        'client_list': clients_numpy_data
    }
    scipy.io.savemat(f"./mat_sample/{dataname}.mat", data_to_save)


    combined_data = np.c_[X, y]
    df = pd.DataFrame(combined_data)
    df.to_csv(f"./csv_sample/{dataname}.csv", index=False)

    return n_clusters, clients_tensor_data, (X_tensor, y_tensor), size

def make_noniid_clients_kmeans(csv_path, n_clients=2, batch_size=128, dataname="iris", seed=42):
    if KMeansConstrained is None:
        raise ImportError(
            "make_noniid_clients_kmeans requires k-means-constrained. "
            "Install it before regenerating client partitions."
        )
    print(f"========== Generate {dataname} with {n_clients} clients ==========")
    rng = np.random.RandomState(seed)

    data = pd.read_csv(csv_path, header=0).values

    X, y = data[:, :-1], data[:, -1].astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    all_clusters = len(np.unique(y))

    n_clusters = n_clients
    kmeans = KMeansConstrained(
        n_clusters=n_clients,
        size_min=all_clusters,
        size_max=None,
        random_state=seed,
        n_init=5
    )


    cluster_labels = kmeans.fit_predict(X)

    clients_tensor_data = []
    clients_numpy_data = []
    size = []

    for i in range(n_clients):
        mask = (cluster_labels == i)

        X_cluster = X[mask]
        Y_cluster = y[mask]

        X_client = torch.tensor(X_cluster, dtype=torch.float32)
        y_client = torch.tensor(Y_cluster, dtype=torch.int)
        size.append(len(Y_cluster))

        dataset = TensorDataset(X_client, y_client)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        clients_tensor_data.append(dataloader)


        client = {}
        client["data"] = X_cluster.astype(np.float32)
        client["label"] = Y_cluster.astype(np.int64)
        client["size"] = len(Y_cluster)
        clients_numpy_data.append(client)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    np.savez(f"./experiment_scalability_wtr_clients/npz/{dataname}_{n_clients}.npz", *clients_numpy_data)

    data_to_save = {
        'client_list': clients_numpy_data
    }
    scipy.io.savemat(f"./experiment_scalability_wtr_clients/mat/{dataname}_{n_clients}.mat", data_to_save)


    combined_data = np.c_[X, y]
    df = pd.DataFrame(combined_data)
    df.to_csv(f"./experiment_scalability_wtr_clients/csv/{dataname}_{n_clients}.csv", index=False)

    return all_clusters, clients_tensor_data, (X_tensor, y_tensor), size

def make_dataset(
        csv_path,
        n_clients=2,
        batch_size=128,
        dataname="iris",
        seed=42,
        data_root=None):

    root = (
        Path(data_root)
        if data_root is not None
        else DEFAULT_DATA_ROOT
    )
    path = root / "npz" / f"{dataname}.npz"

    L = []
    loadData = np.load(
        path,
        allow_pickle=True)
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

    file_path = root / "csv" / f"{dataname}.csv"

    data = pd.read_csv(file_path, header=0).values

    data, label = data[:, :-1], data[:, -1].astype(int)

    X_tensor = torch.tensor(data, dtype=torch.float32)
    y_tensor = torch.tensor(label, dtype=torch.long)

    all_clusters = len(np.unique(label))


    return all_clusters, clients_tensor_data, (X_tensor, y_tensor), size
