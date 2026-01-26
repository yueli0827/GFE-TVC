import argparse
import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score


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


def choose_k_ch_score(
    X: np.ndarray,
    pre_assign: np.ndarray,
    Z: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    sample_max: int = 5000,
) -> Tuple[int, Dict[str, float]]:
    n = X.shape[0]
    k_min = int(max(2, min(k_min, n)))
    k_max = int(max(k_min, min(k_max, n)))

    if n <= 2:
        return k_min, {}

    rng = np.random.default_rng(seed)
    if n > sample_max:
        sample_idx = rng.choice(n, size=sample_max, replace=False)
    else:
        sample_idx = np.arange(n, dtype=np.int64)

    Xs = X[sample_idx]
    bic_map: Dict[str, float] = {}

    best_k = k_min
    best_score = -1e18

    for k in range(k_min, k_max + 1):
        cent_labels = fcluster(Z, t=int(k), criterion="maxclust").astype(np.int64) - 1
        ys = cent_labels[pre_assign[sample_idx]]
        u = np.unique(ys)
        if u.size < 2 or u.size >= Xs.shape[0]:
            continue
        try:
            score = float(calinski_harabasz_score(Xs, ys))
        except Exception:
            continue
        bic_map[str(k)] = score
        if score > best_score:
            best_score = score
            best_k = int(k)

    if not bic_map:
        return k_min, {}

    return best_k, bic_map


def run_hierarchical(X: np.ndarray, k_mode: str, k_min: int, k_max: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = X.astype(np.float32, copy=False)
    X = l2_normalize_rows(X)

    n, d = X.shape
    if n == 0:
        return np.zeros((0,), dtype=np.int64), {"N": 0, "D": int(d), "K_final": 0}
    if n == 1:
        return np.zeros((1,), dtype=np.int64), {"N": 1, "D": int(d), "K_final": 1}

    k_min = int(max(2, min(k_min, n)))
    k_max = int(max(k_min, min(k_max, n)))

    M = int(min(max(10 * k_max, 50), 2000, n))
    if M < k_max:
        M = k_max

    mbk = MiniBatchKMeans(
        n_clusters=int(M),
        random_state=int(seed),
        n_init=5,
        max_iter=200,
        batch_size=4096,
        reassignment_ratio=0.01,
    )
    pre_assign = mbk.fit_predict(X).astype(np.int64)
    C = mbk.cluster_centers_.astype(np.float32, copy=False)

    Z = linkage(C, method="ward")

    if k_mode == "fixed":
        raise ValueError("This script keeps --k_mode interface but does not support fixed here.")

    K, score_map = choose_k_ch_score(
        X=X,
        pre_assign=pre_assign,
        Z=Z,
        k_min=k_min,
        k_max=min(k_max, M),
        seed=seed,
        sample_max=5000,
    )

    cent_labels = fcluster(Z, t=int(K), criterion="maxclust").astype(np.int64) - 1
    labels = cent_labels[pre_assign].astype(np.int64)

    uniq = np.unique(labels)
    remap = {int(c): i for i, c in enumerate(sorted(uniq.tolist()))}
    labels = np.vectorize(lambda x: remap[int(x)])(labels).astype(np.int64)

    info: Dict[str, Any] = {
        "N": int(n),
        "D": int(d),
        "M_preclusters": int(M),
        "K_final": int(len(np.unique(labels))),
        "k_mode": str(k_mode),
        "k_min": int(k_min),
        "k_max": int(k_max),
    }
    if score_map:
        info["ch_by_k"] = {k: float(v) for k, v in score_map.items()}
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
    labels, info = run_hierarchical(X, k_mode=str(args.k_mode), k_min=int(args.k_min), k_max=int(args.k_max), seed=0)
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
