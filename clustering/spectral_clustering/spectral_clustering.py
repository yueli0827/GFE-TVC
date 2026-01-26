import argparse
import csv
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


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

    if suffix == ".mat":
        md = scipy.io.loadmat(str(path))
        md2 = {k: v for k, v in md.items() if not k.startswith("__")}
        return _best_array_from_mapping(md2)

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

    paths = sorted(list({p.resolve() for p in paths}))
    return paths


@dataclass
class SampleMeta:
    source_file: str
    local_index: int
    global_index: int


def load_all_features(input_path: Path) -> Tuple[np.ndarray, List[SampleMeta]]:
    exts = [".npy", ".npz", ".pt", ".pth", ".pkl", ".mat", ".csv", ".txt"]
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


def compute_ball_stats(X: np.ndarray, idx: np.ndarray, radius_q: float) -> Tuple[np.ndarray, float]:
    pts = X[idx]
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    q = float(np.clip(radius_q, 0.5, 1.0))
    radius = float(np.quantile(dists, q))
    return center, radius


def kmeans_split_indices(X: np.ndarray, idx: np.ndarray, seed: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = X[idx]
    if pts.shape[0] < 2:
        return idx, np.array([], dtype=np.int64)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10, max_iter=max_iter)
    lab = km.fit_predict(pts)
    a = idx[lab == 0]
    b = idx[lab == 1]
    return a, b


@dataclass
class GranularBall:
    ball_id: int
    indices: np.ndarray
    center: np.ndarray
    radius: float
    depth: int


def build_granular_balls(
    X: np.ndarray,
    seed: int,
    gb_max_size: int = 400,
    gb_min_size: int = 30,
    gb_max_depth: int = 10,
    gb_radius_q: float = 0.95,
    gb_split_min_center_dist: float = 0.10,
    gb_kmeans_max_iter: int = 200,
) -> List[GranularBall]:
    n = X.shape[0]
    all_idx = np.arange(n, dtype=np.int64)

    global_center = X.mean(axis=0)
    global_scale = float(np.median(np.linalg.norm(X - global_center[None, :], axis=1)) + 1e-12)
    min_center_dist = gb_split_min_center_dist * global_scale

    _, global_radius = compute_ball_stats(X, all_idx, gb_radius_q)

    balls: List[GranularBall] = []
    queue: List[Tuple[np.ndarray, int]] = [(all_idx, 0)]
    ball_id = 0

    while queue:
        idx, depth = queue.pop(0)
        center, radius = compute_ball_stats(X, idx, gb_radius_q)
        need_split = (idx.shape[0] > gb_max_size) or (radius > global_radius)

        if (not need_split) or (depth >= gb_max_depth) or (idx.shape[0] < 2 * gb_min_size):
            balls.append(GranularBall(ball_id=ball_id, indices=idx, center=center, radius=radius, depth=depth))
            ball_id += 1
            continue

        a, b = kmeans_split_indices(X, idx, seed=seed + depth + int(idx.shape[0]), max_iter=gb_kmeans_max_iter)

        if a.shape[0] < gb_min_size or b.shape[0] < gb_min_size:
            balls.append(GranularBall(ball_id=ball_id, indices=idx, center=center, radius=radius, depth=depth))
            ball_id += 1
            continue

        ca, _ = compute_ball_stats(X, a, gb_radius_q)
        cb, _ = compute_ball_stats(X, b, gb_radius_q)
        cdist = float(np.linalg.norm(ca - cb))
        if cdist < min_center_dist:
            balls.append(GranularBall(ball_id=ball_id, indices=idx, center=center, radius=radius, depth=depth))
            ball_id += 1
            continue

        queue.append((a, depth + 1))
        queue.append((b, depth + 1))

    return balls


def build_knn_affinity(
    C: np.ndarray,
    knn_k: int = 20,
    sigma_mode: str = "self_tuning",
    sigma_value: float = 0.0,
) -> sp.csr_matrix:
    m = C.shape[0]
    knn_k = int(min(max(2, knn_k), max(2, m - 1)))

    nn = NearestNeighbors(n_neighbors=knn_k, algorithm="auto", metric="euclidean")
    nn.fit(C)
    dists, nbrs = nn.kneighbors(C, return_distance=True)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    if sigma_value > 0:
        sigma = float(sigma_value)
        denom_eps = 1e-12
        for i in range(m):
            for jpos in range(knn_k):
                j = int(nbrs[i, jpos])
                if j == i:
                    continue
                d = float(dists[i, jpos])
                w = math.exp(-(d * d) / (2.0 * sigma * sigma + denom_eps))
                rows.append(i); cols.append(j); vals.append(w)
    else:
        if sigma_mode == "median":
            sigma = float(np.median(dists[:, 1:]))
            sigma = max(sigma, 1e-6)
            denom_eps = 1e-12
            for i in range(m):
                for jpos in range(knn_k):
                    j = int(nbrs[i, jpos])
                    if j == i:
                        continue
                    d = float(dists[i, jpos])
                    w = math.exp(-(d * d) / (2.0 * sigma * sigma + denom_eps))
                    rows.append(i); cols.append(j); vals.append(w)
        elif sigma_mode == "self_tuning":
            sigma_i = np.maximum(dists[:, -1].copy(), 1e-6)
            denom_eps = 1e-12
            for i in range(m):
                for jpos in range(knn_k):
                    j = int(nbrs[i, jpos])
                    if j == i:
                        continue
                    d = float(dists[i, jpos])
                    denom = (sigma_i[i] * sigma_i[j] + denom_eps)
                    w = math.exp(-(d * d) / denom)
                    rows.append(i); cols.append(j); vals.append(w)
        else:
            raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    W = sp.coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsr()
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def normalized_laplacian(W: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(W.sum(axis=1)).reshape(-1)
    d = np.maximum(d, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = sp.diags(d_inv_sqrt, offsets=0, format="csr")
    I = sp.identity(W.shape[0], format="csr")
    L = I - (D_inv_sqrt @ W @ D_inv_sqrt)
    return L


def smallest_eigs(L: sp.csr_matrix, k: int) -> Tuple[np.ndarray, np.ndarray]:
    vals, vecs = spla.eigsh(L, k=int(k), which="SM", tol=1e-3, maxiter=5000)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def choose_k_by_eigengap(evals: np.ndarray, k_min: int, k_max: int) -> int:
    k_min = int(max(2, k_min))
    k_max = int(min(k_max, len(evals) - 1))
    if k_max <= k_min:
        return k_min
    gaps = evals[1:k_max + 1] - evals[0:k_max]
    lo = k_min - 1
    hi = k_max - 1
    idx = int(lo + np.argmax(gaps[lo:hi + 1]))
    return int(idx + 1)


def kmeans_on_embedding(U: np.ndarray, k: int, seed: int) -> np.ndarray:
    Y = l2_normalize_rows(U)
    km = KMeans(n_clusters=int(k), random_state=seed, n_init=20, max_iter=300)
    return km.fit_predict(Y).astype(np.int64)


def cluster_stats(emb: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        pts = emb[idx]
        mu = pts.mean(axis=0)
        var = float(np.mean(np.sum((pts - mu[None, :]) ** 2, axis=1)))
        stats[int(c)] = {"size": int(idx.shape[0]), "center": mu, "var": var, "indices": idx}
    return stats


def try_split_cluster(
    emb: np.ndarray,
    labels: np.ndarray,
    target_c: int,
    seed: int,
    min_size: int,
    improve_ratio: float,
) -> Optional[np.ndarray]:
    idx = np.where(labels == target_c)[0]
    if idx.shape[0] < 2 * min_size:
        return None
    pts = emb[idx]
    mu = pts.mean(axis=0)
    sse_before = float(np.sum(np.sum((pts - mu[None, :]) ** 2, axis=1)))
    km = KMeans(n_clusters=2, random_state=seed, n_init=10, max_iter=300)
    sub = km.fit_predict(pts)
    a = pts[sub == 0]
    b = pts[sub == 1]
    if a.shape[0] < min_size or b.shape[0] < min_size:
        return None
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    sse_after = float(
        np.sum(np.sum((a - mu_a[None, :]) ** 2, axis=1)) +
        np.sum(np.sum((b - mu_b[None, :]) ** 2, axis=1))
    )
    if sse_after > sse_before * (1.0 - improve_ratio):
        return None
    new_labels = labels.copy()
    new_c = int(labels.max() + 1)
    if a.shape[0] <= b.shape[0]:
        new_labels[idx[sub == 0]] = new_c
        new_labels[idx[sub == 1]] = target_c
    else:
        new_labels[idx[sub == 1]] = new_c
        new_labels[idx[sub == 0]] = target_c
    return new_labels


def try_merge_closest(emb: np.ndarray, labels: np.ndarray, merge_dist_quantile: float) -> Optional[np.ndarray]:
    stats = cluster_stats(emb, labels)
    cs = sorted(stats.keys())
    if len(cs) <= 2:
        return None
    centers = np.stack([stats[c]["center"] for c in cs], axis=0)
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    finite = dmat[np.isfinite(dmat)]
    if finite.size == 0:
        return None
    thr = float(np.quantile(finite, np.clip(merge_dist_quantile, 0.0, 1.0)))
    i, j = np.unravel_index(int(np.argmin(dmat)), dmat.shape)
    if float(dmat[i, j]) > thr:
        return None
    ci = cs[i]
    cj = cs[j]
    new_labels = labels.copy()
    new_labels[new_labels == cj] = ci
    uniq = sorted(np.unique(new_labels).tolist())
    remap = {c: t for t, c in enumerate(uniq)}
    new_labels = np.vectorize(lambda x: remap[int(x)])(new_labels).astype(np.int64)
    return new_labels


def deepdpm_like_refine_k(
    emb: np.ndarray,
    init_labels: np.ndarray,
    seed: int,
    k_min: int,
    k_max: int,
    max_refine_iter: int = 10,
    split_var_quantile: float = 0.80,
    split_improve_ratio: float = 0.05,
    merge_dist_quantile: float = 0.05,
    split_min_size: int = 20,
) -> np.ndarray:
    labels = init_labels.copy()
    for it in range(max_refine_iter):
        changed = False
        if len(np.unique(labels)) > k_min:
            merged = try_merge_closest(emb, labels, merge_dist_quantile=merge_dist_quantile)
            if merged is not None and len(np.unique(merged)) >= k_min and len(np.unique(merged)) <= k_max:
                labels = merged
                changed = True
        if len(np.unique(labels)) < k_max:
            stats = cluster_stats(emb, labels)
            vars_ = np.array([stats[c]["var"] for c in stats.keys()], dtype=np.float64)
            if vars_.size > 0:
                thr = float(np.quantile(vars_, np.clip(split_var_quantile, 0.0, 1.0)))
                cand = [(c, stats[c]["var"], stats[c]["size"]) for c in stats.keys() if stats[c]["var"] >= thr]
                cand = sorted(cand, key=lambda x: x[1], reverse=True)
                for c, _, size in cand:
                    if size < 2 * split_min_size:
                        continue
                    new_labels = try_split_cluster(
                        emb, labels, target_c=int(c),
                        seed=seed + 97 * it + int(c),
                        min_size=split_min_size,
                        improve_ratio=split_improve_ratio
                    )
                    if new_labels is not None:
                        uniq = sorted(np.unique(new_labels).tolist())
                        remap = {cc: t for t, cc in enumerate(uniq)}
                        labels = np.vectorize(lambda x: remap[int(x)])(new_labels).astype(np.int64)
                        changed = True
                        break
        if not changed:
            break
    K = len(np.unique(labels))
    if K < k_min or K > k_max:
        Kc = int(np.clip(K, k_min, k_max))
        labels = kmeans_on_embedding(emb, Kc, seed=seed + 999)
    return labels


def run_pipeline(X: np.ndarray, k_mode: str, k_min: int, k_max: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    X = X.astype(np.float32, copy=False)
    X = l2_normalize_rows(X)

    balls = build_granular_balls(X, seed=seed)
    m = len(balls)
    C = np.stack([b.center for b in balls], axis=0).astype(np.float32)
    ball_sizes = np.array([b.indices.shape[0] for b in balls], dtype=np.int64)

    point2ball = np.empty((X.shape[0],), dtype=np.int64)
    for bi, b in enumerate(balls):
        point2ball[b.indices] = bi

    W = build_knn_affinity(C, knn_k=20, sigma_mode="self_tuning", sigma_value=0.0)
    L = normalized_laplacian(W)

    k_eigs = int(min(max(4, k_max + 2), m - 1))
    evals, evecs = smallest_eigs(L, k=k_eigs)

    K0 = choose_k_by_eigengap(evals, k_min=k_min, k_max=min(k_max, k_eigs - 1))
    K0 = int(np.clip(K0, k_min, k_max))

    U = evecs[:, :K0].astype(np.float32)
    labels_nodes = kmeans_on_embedding(U, K0, seed=seed + 123)

    if k_mode in ("deepdpm", "auto"):
        split_min_size = max(10, int(np.median(ball_sizes) * 0.25))
        labels_nodes = deepdpm_like_refine_k(
            emb=U,
            init_labels=labels_nodes,
            seed=seed + 777,
            k_min=k_min,
            k_max=k_max,
            max_refine_iter=10,
            split_var_quantile=0.80,
            split_improve_ratio=0.05,
            merge_dist_quantile=0.05,
            split_min_size=split_min_size,
        )

    labels_points = labels_nodes[point2ball].astype(np.int64)

    info = {
        "N": int(X.shape[0]),
        "D": int(X.shape[1]),
        "m_balls": int(m),
        "W_nnz": int(W.nnz),
        "K_initial": int(K0),
        "K_final": int(len(np.unique(labels_nodes))),
    }
    return labels_points, info


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
    labels, info = run_pipeline(X, k_mode=str(args.k_mode), k_min=int(args.k_min), k_max=int(args.k_max), seed=0)
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
