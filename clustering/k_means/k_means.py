import argparse
import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans


try:
    import torch
except Exception:
    torch = None


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def ensure_out_dir(out_dir: str | Path) -> Path:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    return outp


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    raise TypeError(f"Cannot convert type {type(x)} to numpy array")


def _best_array_from_mapping(obj: Dict[str, Any]) -> np.ndarray:
    best = None
    best_score = -1
    for _, v in obj.items():
        try:
            a = _to_numpy(v)
        except Exception:
            continue
        if a.ndim not in (1, 2, 3):
            continue
        if a.ndim == 1:
            score = a.shape[0]
        elif a.ndim == 2:
            score = a.shape[0] * 100000 + a.shape[1]
        else:
            score = (a.shape[0] * a.shape[1]) * 100000 + a.shape[2]
        if score > best_score:
            best = a
            best_score = score
    if best is None:
        raise KeyError(f"Cannot auto-select feature array from keys: {list(obj.keys())[:50]}")
    return best


def load_feature_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(str(path), allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                arr2 = arr.item()
                if isinstance(arr2, dict):
                    return _best_array_from_mapping(arr2)
            except Exception:
                pass
        return _to_numpy(arr)

    if suffix == ".npz":
        z = np.load(str(path), allow_pickle=True)
        if len(z.files) == 1:
            return _to_numpy(z[z.files[0]])
        md = {k: z[k] for k in z.files}
        return _best_array_from_mapping(md)

    if suffix in (".pt", ".pth"):
        if torch is None:
            raise ImportError("PyTorch is required to load .pt/.pth. Please convert features to .npy/.npz.")
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict):
            return _best_array_from_mapping(obj)
        return _to_numpy(obj)

    if suffix == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return _best_array_from_mapping(obj)
        return _to_numpy(obj)

    if suffix in (".csv", ".txt"):
        arr = np.loadtxt(str(path), delimiter="," if suffix == ".csv" else None)
        return _to_numpy(arr)

    raise ValueError(f"Unsupported feature file type: {path.name}")


def collect_feature_paths(input_path: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    paths: List[Path] = []
    if recursive:
        for e in exts:
            paths.extend(input_path.rglob(f"*{e}"))
    else:
        for e in exts:
            paths.extend(input_path.glob(f"*{e}"))
    return sorted(list({p.resolve() for p in paths}))


@dataclass
class SampleMeta:
    source_file: str
    local_index: int
    global_index: int


def load_all_features(input_path: Path) -> Tuple[np.ndarray, List[SampleMeta]]:
    exts = [".npy", ".npz", ".pt", ".pth", ".pkl", ".csv", ".txt"]
    paths = collect_feature_paths(input_path, exts=exts, recursive=True)
    if len(paths) == 0:
        raise FileNotFoundError(f"No feature files found under: {input_path}")

    all_x: List[np.ndarray] = []
    meta: List[SampleMeta] = []
    g = 0

    for p in paths:
        x = load_feature_file(p)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 2:
            pass
        elif x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        else:
            raise ValueError(f"Unsupported feature shape {x.shape} in file {p}")

        x = x.astype(np.float32, copy=False)
        all_x.append(x)

        for i in range(x.shape[0]):
            meta.append(SampleMeta(source_file=str(p), local_index=i, global_index=g))
            g += 1

    X = np.concatenate(all_x, axis=0)
    return X, meta


def choose_k_deepdpm_style(X: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    n = X.shape[0]
    if n < k_min:
        return max(2, min(k_min, n))
    k_max = min(k_max, n)
    best_k = k_min
    best_score = -1e18
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        score = -inertia - 0.01 * k * math.log(max(n, 2))
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


def run_kmeans(X: np.ndarray, k_mode: str, k_min: int, k_max: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = X.astype(np.float32, copy=False)
    X = l2_normalize_rows(X)

    if k_mode == "fixed":
        raise ValueError("This script only keeps --k_mode deepdpm/auto/eigengap interface; do not use fixed here.")

    K = choose_k_deepdpm_style(X, k_min=k_min, k_max=k_max, seed=seed)

    km = KMeans(n_clusters=K, random_state=seed, n_init=20, max_iter=300)
    labels = km.fit_predict(X).astype(np.int64)

    info = {
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "K_final": int(K),
        "inertia": float(km.inertia_),
    }
    return labels, info


def save_json(out_dir: Path, name: str, obj: Any) -> None:
    with open(out_dir / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_index_map(out_dir: Path, meta: List[SampleMeta]) -> None:
    p = out_dir / "index_map.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["global_index", "source_file", "local_index"])
        for m in meta:
            w.writerow([m.global_index, m.source_file, m.local_index])


def save_labels(out_dir: Path, labels: np.ndarray, meta: List[SampleMeta]) -> None:
    np.save(out_dir / "labels.npy", labels.astype(np.int64))
    p = out_dir / "labels.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["global_index", "label", "source_file", "local_index"])
        for m in meta:
            w.writerow([m.global_index, int(labels[m.global_index]), m.source_file, m.local_index])


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--k_mode", type=str, default="deepdpm")
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=20)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(0)

    input_path = Path(args.input_path)
    out_dir = ensure_out_dir(args.out_dir)
    save_json(out_dir, "config.json", vars(args))

    log("Loading features")
    X, meta = load_all_features(input_path)
    log(f"X={tuple(X.shape)}")

    log("Clustering")
    labels, info = run_kmeans(X, k_mode=str(args.k_mode), k_min=int(args.k_min), k_max=int(args.k_max), seed=0)
    log(f"K_final={info['K_final']}")

    log("Saving")
    save_index_map(out_dir, meta)
    save_labels(out_dir, labels, meta)
    save_json(out_dir, "info.json", info)

    uniq, cnt = np.unique(labels, return_counts=True)
    summary = {int(u): int(c) for u, c in zip(uniq, cnt)}
    save_json(out_dir, "cluster_size_summary.json", summary)

    log(str(out_dir.resolve()))


if __name__ == "__main__":
    main()
