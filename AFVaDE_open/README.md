# AFVaDE

AFVaDE combines variational deep embedding with buffered asynchronous federated training. This release includes the model, local training, staleness-aware aggregation, GMM sampling, data loading, evaluation, and command-line entry points.

## Installation

Use Python 3.10 or later and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional `k-means-constrained` package is needed only when calling `make_noniid_clients_kmeans` to regenerate client partitions. Loading existing partitions does not require it.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Encoder, decoder, VaDE losses, and GMM sampling |
| `async_communication.py` | Event scheduling, client training, buffered aggregation, and server training |
| `main_async.py` | Model initialization and single-dataset command-line entry point |
| `dataloader.py` | Federated dataset loading and partition-generation helpers |
| `experiment.py` | Clustering metrics and label matching |
| `run_all_datasets.py` | Batch experiments and console performance summaries |

## Data

This release includes six datasets using the default model and training parameters: `audit`, `banknote_authentication`, `breast_cancer`, `iranian_churn`, `iris`, and `phishing_websites`.

By default, the code reads paired CSV and NPZ files from the `Dataset` directory inside this repository:

```text
AFVaDE_open/
├── main_async.py
├── ...
└── Dataset/
    ├── csv/
    │   ├── audit.csv
    │   ├── iris.csv
    │   └── ...
    └── npz/
        ├── audit.npz
        ├── iris.npz
        └── ...
```

Each CSV has a header, feature columns, and an integer label in its final column. Each NPZ entry contains a dictionary with `data`, `label`, and `size` for one client. CSV features and client features must use consistent preprocessing.

Use `--data_root /path/to/Dataset` to specify another location. The existing partitions determine the actual number of clients.

The six paired datasets occupy approximately 9.53 MiB in total. The batch entry point automatically discovers these six datasets from the included `Dataset` directory.

## Run

Run one dataset; its recorded random seed is selected automatically:

```bash
python main_async.py --dataset iris
```

Run on CPU:

```bash
python main_async.py --dataset iris --no-cuda
```

View all model and communication options:

```bash
python main_async.py --help
```

The default encoder widths are `500 500 2000`, and the decoder mirrors them. Training uses 50 VAE warm-up aggregations followed by 100 full AFVaDE aggregations. Each dataset runs once with its fixed seed:

| Dataset | Fixed seed |
| --- | ---: |
| `audit` | 45 |
| `banknote_authentication` | 44 |
| `breast_cancer` | 46 |
| `iranian_churn` | 43 |
| `iris` | 43 |
| `phishing_websites` | 42 |

The default `--client_gmm_k_strategy local` uses the number of distinct local ground-truth labels to select each client's GMM component count. `global` uses the dataset-wide component count. `nonparametric` is an extension point and currently raises `NotImplementedError`.

The single-dataset entry point prints only the dataset name, Purity, ARI, NMI, and ACC after each run. The fixed seed is used internally and is not printed. It does not save event logs, performance files, model checkpoints, or plots. Progress bars are disabled.

Run each included dataset with the default model and training parameters:

```bash
python main_async.py --dataset audit
python main_async.py --dataset banknote_authentication
python main_async.py --dataset breast_cancer
python main_async.py --dataset iranian_churn
python main_async.py --dataset iris
python main_async.py --dataset phishing_websites
```

Run all six included datasets once each with their fixed seeds:

```bash
python run_all_datasets.py
```

The batch entry point prints one line containing the same four metrics for each dataset, using the fixed seeds above. Failures are reported to standard error and the remaining datasets continue. Every invocation runs the included datasets without saving files or loading earlier results.
