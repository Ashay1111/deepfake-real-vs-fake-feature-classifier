from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# -- constants -----------------------------------------------------------------
SIGMA = 3.0
MAP_SIZE = 256
GRID = 4
LOW_ENT_THR = 5.8
HIGH_ENT_THR = 6.8
DEFAULT_SPLITS = ("train", "val", "test")


# -- cli -----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted image features.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Directory containing train.csv/val.csv/test.csv. "
        "If omitted, tries ./splits then current directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("features"),
        help="Output directory for feature CSVs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Which splits to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-split row limit for quick tests (0 = no limit).",
    )
    return parser.parse_args()


def resolve_split_dir(split_dir: Path | None) -> Path:
    if split_dir is not None:
        return split_dir
    if Path("splits").is_dir():
        return Path("splits")
    return Path(".")


# -- image loading --------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32)


# -- intensity features ---------------------------------------------------------
def intensity_features(gray: np.ndarray) -> Dict[str, float]:
    return {
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "entropy": _entropy(gray),
    }


def _entropy(gray: np.ndarray) -> float:
    h, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


# -- noise residual features ----------------------------------------------------
def residual_features(gray: np.ndarray) -> Dict[str, float]:
    blurred = gaussian_filter(gray, sigma=SIGMA)
    resid = gray - blurred

    ac = _autocorr_map(resid, MAP_SIZE)
    c = MAP_SIZE // 2

    ac_cl = ac.copy()
    ac_cl[c, c] = 0.0

    yy, xx = np.indices(ac.shape)
    rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    near = (rr >= 1) & (rr <= 3)
    ring = (rr >= 6) & (rr <= 18)

    near_e = float(np.mean(np.abs(ac[near]))) if near.any() else 0.0
    ring_e = float(np.mean(np.abs(ac[ring]))) if ring.any() else 0.0

    return {
        "residual_variance": float(np.var(resid)),
        "autocorr_strength": float(np.mean(np.abs(ac_cl))),
        "offcenter_energy": float(np.mean(ac_cl**2)),
        "near_center_energy": near_e,
        "ring_energy": ring_e,
        "center_ring_gap": near_e - ring_e,
    }


def _autocorr_map(resid: np.ndarray, map_size: int) -> np.ndarray:
    resid = resid - resid.mean()
    f = np.fft.fft2(resid)
    ac = np.fft.ifft2(np.abs(f) ** 2).real
    ac = np.fft.fftshift(ac)

    h, w = ac.shape
    cy, cx = h // 2, w // 2
    half = map_size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    crop = ac[y0:y1, x0:x1]

    if crop.shape != (map_size, map_size):
        py = map_size - crop.shape[0]
        px = map_size - crop.shape[1]
        crop = np.pad(
            crop,
            ((py // 2, py - py // 2), (px // 2, px - px // 2)),
            mode="constant",
        )

    center = crop[map_size // 2, map_size // 2]
    if abs(center) < 1e-6:
        return np.zeros_like(crop, dtype=np.float32)
    return (crop / center).astype(np.float32)


# -- frequency features ---------------------------------------------------------
def frequency_features(gray: np.ndarray) -> Dict[str, float]:
    blurred = gaussian_filter(gray, sigma=SIGMA)
    resid = gray - blurred

    ff = np.fft.fftshift(np.fft.fft2(resid))
    p2 = np.abs(ff) ** 2
    radial = _radial_profile(p2)
    slope, intercept = _loglog_fit(radial)

    low_b = float(np.mean(radial[1:8])) if len(radial) > 8 else 0.0
    high_b = float(np.mean(radial[20:60])) if len(radial) > 60 else 0.0
    ratio = float(low_b / (high_b + 1e-12))

    return {
        "radial_loglog_slope": slope,
        "radial_loglog_intercept": intercept,
        "low_band_power": low_b,
        "high_band_power": high_b,
        "low_high_ratio": ratio,
    }


def _radial_profile(p2: np.ndarray) -> np.ndarray:
    h, w = p2.shape
    y, x = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    max_r = r.max()
    radial = np.bincount(r.ravel(), weights=p2.ravel(), minlength=max_r + 1)
    counts = np.bincount(r.ravel(), minlength=max_r + 1)
    counts[counts == 0] = 1
    return radial / counts


def _loglog_fit(radial: np.ndarray) -> Tuple[float, float]:
    r = np.arange(len(radial), dtype=np.float64)
    p = radial.astype(np.float64)
    mask = (r >= 1.0) & np.isfinite(p) & (p > 0.0)
    if mask.sum() < 2:
        return 0.0, 0.0
    coef = np.polyfit(np.log10(r[mask]), np.log10(p[mask] + 1e-12), deg=1)
    return float(coef[0]), float(coef[1])


# -- patch features -------------------------------------------------------------
def patch_features(gray: np.ndarray) -> Dict[str, float | int]:
    h, w = gray.shape
    ph, pw = h // GRID, w // GRID
    if ph == 0 or pw == 0:
        return {
            "mean_patch_entropy": 0.0,
            "std_patch_entropy": 0.0,
            "low_entropy_patch_ratio": 0.0,
            "high_entropy_patch_ratio": 0.0,
            "largest_cluster_size": 0,
            "num_clusters": 0,
            "cluster_density": 0.0,
        }

    entropies: List[float] = []
    for gy in range(GRID):
        for gx in range(GRID):
            patch = gray[gy * ph : (gy + 1) * ph, gx * pw : (gx + 1) * pw]
            entropies.append(_entropy(patch))

    ents = np.array(entropies)
    low_mask = ents < LOW_ENT_THR
    high_mask = ents > HIGH_ENT_THR

    clusters = _count_clusters(low_mask.reshape(GRID, GRID))
    largest = max(clusters) if clusters else 0

    return {
        "mean_patch_entropy": float(ents.mean()),
        "std_patch_entropy": float(ents.std()),
        "low_entropy_patch_ratio": float(low_mask.mean()),
        "high_entropy_patch_ratio": float(high_mask.mean()),
        "largest_cluster_size": int(largest),
        "num_clusters": int(len(clusters)),
        "cluster_density": float(low_mask.mean()),
    }


def _count_clusters(mask: np.ndarray) -> List[int]:
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    sizes: List[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or vis[y, x]:
                continue
            queue = [(y, x)]
            vis[y, x] = True
            size = 0
            while queue:
                cy, cx = queue.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        queue.append((ny, nx))
            sizes.append(size)
    return sizes


# -- single image -> all features ----------------------------------------------
def extract(path: str) -> Dict[str, float | int]:
    gray = load_gray(path)
    return {
        **intensity_features(gray),
        **residual_features(gray),
        **frequency_features(gray),
        **patch_features(gray),
    }


# -- process one split ---------------------------------------------------------
def process_split(split: str, split_dir: Path, out_dir: Path, limit: int = 0) -> None:
    csv_path = split_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split file: {csv_path}")

    df = pd.read_csv(csv_path)
    if limit > 0:
        df = df.head(limit).copy()

    records = []
    failed = []
    total = len(df)

    print(f"\n[{split}] {total} images")
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            feats = extract(row.path)
            records.append(
                {
                    "path": row.path,
                    "method": row.method,
                    "label": row.label,
                    "augmented": getattr(row, "augmented", 0),
                    **feats,
                }
            )
        except Exception as exc:
            failed.append({"path": row.path, "error": str(exc)})

        if i % 500 == 0 or i == total:
            elapsed = max(time.time() - t0, 1e-9)
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0.0
            print(f"  {i}/{total} - {rate:.1f} img/s - ETA {eta:.0f}s")

    out_df = pd.DataFrame(records)
    out_path = out_dir / f"{split}_features.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path} ({len(records)} rows, {len(failed)} failed)")

    if failed:
        failed_path = out_dir / f"{split}_failed.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"  failed paths -> {failed_path}")


def main() -> None:
    args = parse_args()
    split_dir = resolve_split_dir(args.split_dir).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using split dir: {split_dir}")
    print(f"Output dir: {out_dir}")

    for split in args.splits:
        process_split(split=split, split_dir=split_dir, out_dir=out_dir, limit=args.limit)

    print("\nDone. Output files:")
    for f in sorted(out_dir.glob("*.csv")):
        rows = len(pd.read_csv(f))
        print(f"  {f.name:35s} {rows} rows")


if __name__ == "__main__":
    main()
