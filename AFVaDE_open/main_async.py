from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import warnings


sys.dont_write_bytecode = True
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from async_communication import (
    AFVaDEAsyncTrainer,
    AsyncAFVaDEConfig,
    AsyncAFVaDEResult,
)
from dataloader import make_dataset
from model import VaDE, collect_gmm_sampling_diagnostics


ASYNFED_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ASYNFED_DIR / "Dataset"

DATASET_SEEDS = {
    "audit": 45,
    "banknote_authentication": 44,
    "breast_cancer": 46,
    "iranian_churn": 43,
    "iris": 43,
    "phishing_websites": 42,
}


def positive_layer_width(value: str) -> int:

    width = int(value)
    if width <= 0:
        raise argparse.ArgumentTypeError("hidden-layer widths must be positive")
    return width


def nonnegative_integer(value: str) -> int:

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def set_random_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_afvade(args: argparse.Namespace, seed: int):

    set_random_seed(seed)

    data_root = Path(args.data_root).resolve()
    csv_path = data_root / "csv" / f"{args.dataset}.csv"
    npz_path = data_root / "npz" / f"{args.dataset}.npz"
    missing_paths = [path for path in (csv_path, npz_path) if not path.is_file()]
    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "找不到数据集文件。请将 Dataset 放在 AFVaDE_open 目录内，或通过 "
            f"--data_root 指定 Dataset 目录：\n{missing_text}"
        )

    args.nClusters, DL, (X, y), size = make_dataset(
        csv_path,
        args.client_num,
        args.batch_size,
        args.dataset,
        seed=seed,
        data_root=data_root,
    )
    args.input_dim = X.shape[1]
    args.client_num = len(DL)
    size = np.asarray(size, dtype=np.float64)
    weight = size / size.sum()

    requested_cuda = bool(args.cuda)
    use_cuda = requested_cuda and torch.cuda.is_available()
    args.cuda = use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")


    server = {
        "vade": VaDE(args).to(device),
        "optimizer": None,
        "lr_s": None,
    }


    initial_server_state = server["vade"].state_dict()
    clients = []
    for client_id, dataloader in enumerate(DL):
        client = {
            "client_id": client_id,
            "dataloader": dataloader,
            "vade": VaDE(args).to(device),
            "optimizer": None,
            "lr_s": None,
            "pre_optimizer": None,
            "pre_lr_s": None,
        }
        client["vade"].load_state_dict(initial_server_state)
        clients.append(client)

    server["optimizer"] = Adam(server["vade"].parameters(), lr=args.lr)
    server["lr_s"] = StepLR(
        server["optimizer"],
        step_size=args.stepsize,
        gamma=args.gamma,
    )
    for client in clients:
        client["optimizer"] = Adam(client["vade"].parameters(), lr=args.lr)
        client["lr_s"] = StepLR(
            client["optimizer"],
            step_size=args.stepsize,
            gamma=args.gamma,
        )
        if args.pre_epoch > 0:


            vae_parameters = list(client["vade"].encoder.parameters()) + list(
                client["vade"].decoder.parameters()
            )
            client["pre_optimizer"] = Adam(vae_parameters, lr=args.lr)
            client["pre_lr_s"] = StepLR(
                client["pre_optimizer"],
                step_size=args.stepsize,
                gamma=args.gamma,
            )

    return server, clients, weight, device


def run_one_async_experiment(args: argparse.Namespace) -> AsyncAFVaDEResult:


    run_args = copy.deepcopy(args)
    seed = DATASET_SEEDS[run_args.dataset]
    run_args.seed = seed
    server, clients, data_weights, device = initialize_afvade(run_args, seed)

    total_aggregations = run_args.pre_epoch + run_args.epoch
    config = AsyncAFVaDEConfig(
        seed=seed,
        client_count=len(clients),
        buffer_size=run_args.buffer_size,
        max_aggregations=total_aggregations,
        pre_aggregations=run_args.pre_epoch,
        slow_client_count=run_args.slow_client_count,
        fast_delay=(run_args.fast_delay_min, run_args.fast_delay_max),
        slow_delay=(run_args.slow_delay_min, run_args.slow_delay_max),
        online_probability=run_args.online_probability,
        availability_interval=run_args.availability_interval,
        server_beta0=run_args.server_beta0,
        staleness_a=run_args.staleness_a,
        staleness_h_min=run_args.staleness_h_min,
        staleness_cap=run_args.staleness_cap,
        gmm_staleness_gamma=run_args.gmm_staleness_gamma,
        buffer_duplicate_policy=run_args.buffer_duplicate_policy,
        client_gmm_k_strategy=run_args.client_gmm_k_strategy,

        eval_interval=total_aggregations + 1,
    )
    trainer = AFVaDEAsyncTrainer(
        run_args,
        config,
        server,
        clients,
        data_weights,
        device,
    )

    with collect_gmm_sampling_diagnostics() as gmm_diagnostics, warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = trainer.run()
    result.summary.update(gmm_diagnostics.csv_summary())


    print(
        f"[dataset={run_args.dataset}] "
        f"Purity={result.summary['final_purity']:.4f} "
        f"ARI={result.summary['final_ari']:.4f} "
        f"NMI={result.summary['final_nmi']:.4f} "
        f"ACC={result.summary['final_acc']:.4f}",
        flush=True,
    )
    return result


def AFVaDE_async(args: argparse.Namespace) -> None:

    run_one_async_experiment(args)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:

    parse = argparse.ArgumentParser(
        description="AFVaDE with buffered heterogeneous asynchronous communication"
    )

    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument(
        "--dataset",
        type=str,
        choices=tuple(DATASET_SEEDS),
        default="audit",
        help="Dataset name; its recorded random seed is selected automatically",
    )
    parse.add_argument("--nClusters", type=int, default=0)
    parse.add_argument("--input_dim", type=int, default=0)
    parse.add_argument("--hid_dim", type=int, default=10)
    parse.add_argument(
        "--network",
        type=positive_layer_width,
        nargs="+",
        default=[500, 500, 2000],
        help="Encoder隐藏层宽度；Decoder自动按逆序构建",
    )
    parse.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA when available; use --no-cuda to force CPU",
    )
    parse.add_argument("--client_num", type=int, default=10)
    parse.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"数据根目录；其中应包含npz和csv两个子文件夹（默认：{DEFAULT_DATA_ROOT}）",
    )
    parse.add_argument("--lr", type=float, default=2e-3)
    parse.add_argument("--stepsize", type=int, default=10)
    parse.add_argument("--gamma", type=float, default=0.85)
    parse.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="VAE预训练结束后的完整AFVaDE服务器聚合次数，默认100",
    )
    parse.add_argument(
        "--pre_epoch",
        type=nonnegative_integer,
        default=50,
        help="正式AFVaDE前仅训练/聚合VAE的服务器聚合次数，默认50",
    )
    parse.add_argument("--iters", type=int, default=5)
    parse.add_argument("--alpha", type=float, default=0.001)
    parse.add_argument("--sample", type=int, default=100)
    parse.add_argument("--xi", type=float, default=1.0)
    parse.add_argument(
        "--client_gmm_k_strategy",
        type=str,
        choices=("global", "local", "nonparametric"),
        default="local",
        help=(
            "客户端GMM簇数策略：global使用全局K；local使用本地标签中的"
            "不同类别数（与原AFVaDE一致）；nonparametric为预留扩展插口"
        ),
    )

    parse.add_argument("--buffer_size", type=int, default=3)
    parse.add_argument("--slow_client_count", type=int, default=3)
    parse.add_argument(
        "--fast_delay_min",
        type=float,
        default=1.0,
        help="快客户端连续均匀延迟区间的下界",
    )
    parse.add_argument(
        "--fast_delay_max",
        type=float,
        default=2.0,
        help="快客户端连续均匀延迟区间的上界",
    )
    parse.add_argument(
        "--slow_delay_min",
        type=float,
        default=5.0,
        help="慢客户端连续均匀延迟区间的下界",
    )
    parse.add_argument(
        "--slow_delay_max",
        type=float,
        default=8.0,
        help="慢客户端连续均匀延迟区间的上界",
    )
    parse.add_argument(
        "--online_probability",
        type=float,
        default=0.8,
        help="上传后客户端在线并接收最新模型的Bernoulli概率",
    )
    parse.add_argument("--availability_interval", type=float, default=1.0)

    parse.add_argument(
        "--server_beta0",
        type=float,
        default=1.0,
        help="V6服务器delta聚合的基础step size，必须大于0",
    )
    parse.add_argument(
        "--staleness_a",
        type=float,
        default=0.5,
        help="Polynomial freshness h(s)=(1+s)^(-a) 的指数，必须非负",
    )
    parse.add_argument(
        "--staleness_h_min",
        type=float,
        default=0.1,
        help="Freshness下限，范围[0,1]",
    )
    parse.add_argument(
        "--staleness_cap",
        type=int,
        default=-1,
        help="Freshness计算使用的最大staleness；<=0表示禁用",
    )
    parse.add_argument(
        "--gmm_staleness_gamma",
        type=float,
        default=0.0,
        help="GMM sampling受staleness影响的强度，范围[0,1]",
    )
    parse.add_argument(
        "--buffer_duplicate_policy",
        type=str,
        choices=("v6_unique", "separate"),
        default="v6_unique",
        help="v6_unique按unique client分配quota；separate把每条update独立处理",
    )

    return parse


def main() -> int:
    args = build_parser().parse_args()
    AFVaDE_async(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
