import argparse
import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if isinstance(x, torch.Tensor):
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


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid)
        self.ln1 = nn.LayerNorm(hid)
        self.fc2 = nn.Linear(hid, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_knn_neighbors(X: np.ndarray, knn_k: int = 20, metric: str = "cosine") -> np.ndarray:
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, 0), dtype=np.int64)
    k = int(min(knn_k + 1, n))
    nnm = NearestNeighbors(n_neighbors=k, algorithm="auto", metric=metric)
    nnm.fit(X)
    _, idx = nnm.kneighbors(X, return_distance=True)
    if idx.shape[1] >= 2:
        idx = idx[:, 1:]
    else:
        idx = idx[:, :0]
    return idx.astype(np.int64)


def info_nce_loss(z_a: torch.Tensor, z_p: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=1)
    z_p = F.normalize(z_p, dim=1)
    logits_neg = (z_a @ z_a.t()) / temperature
    b = logits_neg.shape[0]
    logits_neg = logits_neg.masked_fill(torch.eye(b, device=logits_neg.device, dtype=torch.bool), -1e9)
    logits_pos = torch.sum(z_a * z_p, dim=1, keepdim=True) / temperature
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    labels = torch.zeros((b,), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def train_scgc_lite_embeddings(
    X: np.ndarray,
    neighbors: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, d = X.shape

    epochs = 200
    batch_size = 1024
    lr = 1e-3
    weight_decay = 1e-5
    temperature = 0.2
    feat_dropout_p = 0.2

    encoder = MLPEncoder(in_dim=d, hid=256, out_dim=128).to(device)
    proj = ProjectionHead(dim=128).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=lr, weight_decay=weight_decay)

    X_t = torch.from_numpy(X).float()

    gen = torch.Generator()
    gen.manual_seed(seed)

    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        steps = 0

        for s in range(0, n, batch_size):
            idx_a = perm[s:s + batch_size]
            if idx_a.numel() <= 1:
                continue

            idx_a_np = idx_a.cpu().numpy()
            if neighbors.shape[1] == 0:
                idx_p_np = idx_a_np.copy()
            else:
                ridx = np.random.randint(0, neighbors.shape[1], size=idx_a_np.shape[0])
                idx_p_np = neighbors[idx_a_np, ridx]
                bad = idx_p_np < 0
                if np.any(bad):
                    idx_p_np[bad] = idx_a_np[bad]

            xa = X_t[idx_a].to(device, non_blocking=True)
            xp = X_t[torch.from_numpy(idx_p_np)].to(device, non_blocking=True)

            xa = F.dropout(xa, p=feat_dropout_p, training=True)
            xp = F.dropout(xp, p=feat_dropout_p, training=True)

            za = proj(encoder(xa))
            zp = proj(encoder(xp))

            loss = info_nce_loss(za, zp, temperature=temperature)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu().item())
            steps += 1

        if (ep + 1) % 20 == 0:
            log(f"epoch {ep+1}/{epochs} loss {total/max(1,steps):.6f}")

    encoder.eval()
    proj.eval()

    out = np.zeros((n, 128), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, 4096):
            xb = X_t[s:s + 4096].to(device, non_blocking=True)
            zb = proj(encoder(xb))
            zb = F.normalize(zb, dim=1)
            out[s:s + 4096] = zb.detach().cpu().numpy().astype(np.float32)

    return out


def choose_k_by_bic(emb: np.ndarray, k_min: int, k_max: int, seed: int) -> Tuple[int, Dict[str, float]]:
    n = emb.shape[0]
    if n < k_min:
        return max(2, min(k_min, n)), {}
    k_max = min(k_max, n)
    best_k = k_min
    best_bic = float("inf")
    bic_map: Dict[str, float] = {}
    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=int(k),
                covariance_type="diag",
                reg_covar=1e-6,
                max_iter=200,
                n_init=2,
                init_params="kmeans",
                random_state=int(seed),
            )
            gmm.fit(emb)
            bic = float(gmm.bic(emb))
            bic_map[str(k)] = bic
            if bic < best_bic:
                best_bic = bic
                best_k = int(k)
        except Exception:
            continue
    if not bic_map:
        return int(max(2, k_min)), {}
    return best_k, bic_map


def cluster_embeddings(emb: np.ndarray, k_mode: str, k_min: int, k_max: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    K, bic_map = choose_k_by_bic(emb, k_min=k_min, k_max=k_max, seed=seed)
    km = KMeans(n_clusters=int(K), random_state=int(seed), n_init=20, max_iter=300)
    labels = km.fit_predict(emb).astype(np.int64)
    info: Dict[str, Any] = {
        "N": int(emb.shape[0]),
        "D_emb": int(emb.shape[1]),
        "K_final": int(K),
        "k_mode": str(k_mode),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "kmeans_inertia": float(km.inertia_),
    }
    if bic_map:
        info["bic_by_k"] = {k: float(v) for k, v in bic_map.items()}
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
    X = l2_normalize_rows(X.astype(np.float32, copy=False))
    log(f"X={tuple(X.shape)}")

    log("Building kNN graph")
    neighbors = build_knn_neighbors(X, knn_k=20, metric="cosine")
    log(f"neighbors={neighbors.shape}")

    log("Training SCGC-style embeddings")
    emb = train_scgc_lite_embeddings(X, neighbors, seed=0)

    log("Clustering")
    labels, info = cluster_embeddings(emb, k_mode=str(args.k_mode), k_min=int(args.k_min), k_max=int(args.k_max), seed=0)
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
