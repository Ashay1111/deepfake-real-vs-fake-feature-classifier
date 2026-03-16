#!/usr/bin/env python3
"""Build train/val/test manifests for real-vs-fake classification."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABELS = ("real", "fake")
CSV_COLUMNS = ("path", "label", "method", "augmented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("extracted"),
        help="Root directory containing method folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where train.csv/val.csv/test.csv will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--rotate-degrees",
        type=float,
        default=12.0,
        help="Rotation angle used for StyleCLIP real-image augmentation.",
    )
    parser.add_argument(
        "--translate-x",
        type=int,
        default=12,
        help="Horizontal translation in pixels for StyleCLIP real-image augmentation.",
    )
    parser.add_argument(
        "--translate-y",
        type=int,
        default=8,
        help="Vertical translation in pixels for StyleCLIP real-image augmentation.",
    )
    return parser.parse_args()


def resolve_method_root(method_dir: Path) -> Path:
    nested = method_dir / method_dir.name
    return nested if nested.is_dir() else method_dir


def iter_method_dirs(data_root: Path) -> Iterable[Tuple[str, Path]]:
    for method_dir in sorted(data_root.iterdir()):
        if method_dir.name.startswith(".") or not method_dir.is_dir():
            continue
        method_root = resolve_method_root(method_dir)
        if (method_root / "real").is_dir() and (method_root / "fake").is_dir():
            yield method_dir.name, method_root


def check_image_readable(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_full_manifest(
    data_root: Path,
) -> Tuple[List[Dict[str, object]], List[Tuple[Path, str]]]:
    records: List[Dict[str, object]] = []
    skipped_unreadable: List[Tuple[Path, str]] = []
    for method, method_root in iter_method_dirs(data_root):
        for label in LABELS:
            label_dir = method_root / label
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                readable, reason = check_image_readable(path)
                if not readable:
                    skipped_unreadable.append((path, reason))
                    continue
                records.append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "method": method,
                        "augmented": 0,
                    }
                )
    return records, skipped_unreadable


def stratified_split_within_method(
    records: List[Dict[str, object]], seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["method"]), str(row["label"]))].append(row)

    train_rows: List[Dict[str, object]] = []
    val_rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for key in sorted(grouped):
        rows = grouped[key]
        rng.shuffle(rows)
        n_total = len(rows)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        n_test = n_total - n_train - n_val

        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train : n_train + n_val])
        test_rows.extend(rows[n_train + n_val : n_train + n_val + n_test])

    return train_rows, val_rows, test_rows


def save_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.convert("RGB").save(out_path, quality=95)
    elif ext == ".png":
        image.save(out_path, optimize=True)
    else:
        image.save(out_path)


def augment_styleclip_train_reals(
    train_rows: List[Dict[str, object]],
    output_dir: Path,
    rotate_degrees: float,
    translate_x: int,
    translate_y: int,
) -> List[Dict[str, object]]:
    augmented_rows: List[Dict[str, object]] = []
    aug_root = output_dir / "styleclip_train_real_aug"
    aug_root.mkdir(parents=True, exist_ok=True)

    for row in train_rows:
        method = str(row["method"])
        label = str(row["label"])
        if method.lower() != "styleclip" or label != "real":
            continue

        src_path = Path(str(row["path"]))
        out_ext = src_path.suffix.lower()
        if out_ext not in {".jpg", ".jpeg", ".png"}:
            out_ext = ".jpg"

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                transformed = {
                    "flip": ImageOps.mirror(img),
                    "rot": img.rotate(
                        rotate_degrees, resample=Image.BICUBIC, fillcolor=(0, 0, 0)
                    ),
                    "trans": img.transform(
                        img.size,
                        Image.AFFINE,
                        (1, 0, translate_x, 0, 1, translate_y),
                        resample=Image.BICUBIC,
                        fillcolor=(0, 0, 0),
                    ),
                }

                for aug_name, aug_img in transformed.items():
                    out_path = aug_root / f"{src_path.stem}__{aug_name}{out_ext}"
                    save_image(aug_img, out_path)
                    augmented_rows.append(
                        {
                            "path": str(out_path.resolve()),
                            "label": label,
                            "method": method,
                            "augmented": 1,
                        }
                    )
        except Exception as exc:  # pragma: no cover - only for corrupt files
            print(f"[WARN] failed to augment {src_path}: {exc}")

    return augmented_rows


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    train_rows: List[Dict[str, object]],
    val_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
) -> None:
    split_map = {"train": train_rows, "val": val_rows, "test": test_rows}
    methods = sorted(
        {str(row["method"]) for rows in split_map.values() for row in rows}
    )

    print("\nSummary table")
    print(
        f"{'split':<8}{'method':<14}{'real':>8}{'fake':>8}{'augmented':>12}{'total':>10}"
    )
    print("-" * 60)

    grand_total = 0
    grand_aug = 0
    for split_name, rows in split_map.items():
        split_total = 0
        split_aug = 0
        for method in methods:
            method_rows = [r for r in rows if str(r["method"]) == method]
            if not method_rows:
                continue
            real_count = sum(1 for r in method_rows if str(r["label"]) == "real")
            fake_count = sum(1 for r in method_rows if str(r["label"]) == "fake")
            aug_count = sum(1 for r in method_rows if int(r["augmented"]) == 1)
            total = len(method_rows)
            split_total += total
            split_aug += aug_count
            print(
                f"{split_name:<8}{method:<14}{real_count:>8}{fake_count:>8}"
                f"{aug_count:>12}{total:>10}"
            )
        grand_total += split_total
        grand_aug += split_aug
        print(
            f"{split_name:<8}{'TOTAL':<14}{'':>8}{'':>8}{split_aug:>12}{split_total:>10}"
        )
        print("-" * 60)

    print(f"{'ALL':<8}{'TOTAL':<14}{'':>8}{'':>8}{grand_aug:>12}{grand_total:>10}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    full_manifest, skipped_unreadable = build_full_manifest(data_root)
    if not full_manifest:
        raise RuntimeError(f"No images found under {data_root}")

    if skipped_unreadable:
        print(f"[WARN] skipped unreadable images: {len(skipped_unreadable)}")
        for path, reason in skipped_unreadable[:10]:
            print(f"  - {path}: {reason}")

    train_rows, val_rows, test_rows = stratified_split_within_method(
        full_manifest, seed=args.seed
    )
    train_rows.extend(
        augment_styleclip_train_reals(
            train_rows=train_rows,
            output_dir=output_dir,
            rotate_degrees=args.rotate_degrees,
            translate_x=args.translate_x,
            translate_y=args.translate_y,
        )
    )

    write_csv(train_rows, output_dir / "train.csv")
    write_csv(val_rows, output_dir / "val.csv")
    write_csv(test_rows, output_dir / "test.csv")
    print_summary(train_rows=train_rows, val_rows=val_rows, test_rows=test_rows)


if __name__ == "__main__":
    main()
