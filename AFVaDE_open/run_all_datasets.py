from __future__ import annotations

import argparse
import copy
import gc
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch

from main_async import DATASET_SEEDS, build_parser, run_one_async_experiment


METRIC_KEYS = {
    "purity": "final_purity",
    "ari": "final_ari",
    "nmi": "final_nmi",
    "acc": "final_acc",
}


def discover_datasets(data_root: Path) -> list[str]:
    csv_dir = data_root / "csv"
    npz_dir = data_root / "npz"
    if not csv_dir.is_dir() or not npz_dir.is_dir():
        raise FileNotFoundError(
            "Dataset 目录必须同时包含 csv 和 npz 子目录："
            f"\n  - {csv_dir}\n  - {npz_dir}"
        )

    csv_names = {
        path.stem for path in csv_dir.glob("*.csv")
        if path.is_file() and path.stem in DATASET_SEEDS
    }
    npz_names = {
        path.stem for path in npz_dir.glob("*.npz")
        if path.is_file() and path.stem in DATASET_SEEDS
    }

    missing_npz = sorted(csv_names - npz_names)
    missing_csv = sorted(npz_names - csv_names)
    if missing_npz or missing_csv:
        details = []
        if missing_npz:
            details.append("缺少 NPZ：" + ", ".join(missing_npz))
        if missing_csv:
            details.append("缺少 CSV：" + ", ".join(missing_csv))
        raise FileNotFoundError("数据文件没有正确配对：\n  - " + "\n  - ".join(details))

    datasets = sorted(csv_names & npz_names)
    if not datasets:
        raise FileNotFoundError(f"没有在 {data_root} 中找到可运行的数据集")
    return datasets


def cleanup_after_run(dataset: str) -> None:
    try:
        gc.collect()
    except Exception:
        print(
            f"[dataset={dataset}] Memory cleanup failed",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        print(
            f"[dataset={dataset}] CUDA cleanup failed",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()


def build_batch_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Run the included datasets with fixed seeds and print performance"
    return parser


def main() -> int:
    parser = build_batch_parser()
    args = parser.parse_args()
    datasets = discover_datasets(Path(args.data_root).resolve())
    for dataset in datasets:
        dataset_args = copy.deepcopy(args)
        dataset_args.dataset = dataset
        result = None
        try:
            result = run_one_async_experiment(dataset_args)
            metrics = {
                column: float(result.summary[summary_key])
                for column, summary_key in METRIC_KEYS.items()
            }
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("训练结果包含非有限性能指标（NaN/Inf）")
        except Exception as exc:
            print(
                f"[dataset={dataset}] FAILED: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
        finally:
            result = None
            cleanup_after_run(dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
