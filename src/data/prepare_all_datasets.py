"""
Consolidated dataset preparation pipeline.

This module centralizes the preprocessing steps needed to reproduce the VAD-Net
experiments:
1. CAER-S train/val/test splitting with a fixed seed.
2. Emotic preprocessing into 48x48 grayscale FER-style CSV rows.

The script writes the derived CSVs into the data/ folder and logs the exact
counts and output paths so experiment runs can be reproduced later.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================================
# EMOTIC PREPROCESSING UTILITIES
# ============================================================================

EMOTIC_SPLITS = {
    "train": "annot_arrs_train.csv",
    "val": "annot_arrs_val.csv",
    "test": "annot_arrs_test.csv",
}


def scale_vad_to_minus2_plus2(values: np.ndarray) -> np.ndarray:
    """Scale VAD values from [1,10] to [-2,2] range (Emotic format)."""
    scaled = (values.astype(np.float32) - 5.0) / 2.5
    return np.clip(scaled, -2.0, 2.0)


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 grayscale."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        img = arr
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        img = arr[:, :, :3] if arr.shape[2] >= 3 else arr[:, :, 0]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if img.dtype != np.uint8:
        maxv = float(np.nanmax(img)) if img.size else 0.0
        if maxv <= 1.0:
            img = (np.nan_to_num(img) * 255.0).astype(np.uint8)
        else:
            img = np.nan_to_num(img).astype(np.uint8)
    return img


def image_to_gray48_pixels(image_array: np.ndarray, size: int = 48) -> str:
    """Convert image to 48x48 grayscale and return as space-separated pixel string."""
    img_u8 = to_uint8_image(image_array)
    pil_img = Image.fromarray(img_u8)
    gray = pil_img.convert("L").resize((size, size), Image.BILINEAR)
    gray_arr = np.array(gray, dtype=np.uint8)
    return " ".join(str(int(v)) for v in gray_arr.reshape(-1))


def preprocess_emotic_split(emotic_root: Path, split: str, output_csv: Path) -> int:
    """Preprocess Emotic dataset split into FER-style VAD CSV."""
    ann_path = emotic_root / "annots_arrs" / EMOTIC_SPLITS[split]
    img_root = emotic_root / "img_arrs"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing Emotic annotation file: {ann_path}")

    ann = pd.read_csv(ann_path)
    required = ["Crop_name", "Valence", "Arousal", "Dominance"]
    missing = [c for c in required if c not in ann.columns]
    if missing:
        raise ValueError(f"Missing columns in {ann_path}: {missing}")

    rows = []
    for record in ann.itertuples(index=False):
        crop_name = str(getattr(record, "Crop_name"))
        crop_path = img_root / crop_name
        if not crop_path.exists():
            continue

        try:
            crop = np.load(crop_path, allow_pickle=False)
            pixels = image_to_gray48_pixels(crop, size=48)
        except Exception:
            continue

        vad_raw = np.array([
            float(getattr(record, "Valence")),
            float(getattr(record, "Arousal")),
            float(getattr(record, "Dominance")),
        ], dtype=np.float32)
        v, a, d = scale_vad_to_minus2_plus2(vad_raw)

        rows.append(
            {
                "pixels": pixels,
                "Valence": float(v),
                "Arousal": float(a),
                "Dominance": float(d),
                "source": f"emotic_{split}",
                "emotion": "",
                "path": str(crop_path.as_posix()),
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return len(out_df)


# ============================================================================
# MAIN PREPARATION ORCHESTRATION
# ============================================================================

def log_line(message: str, lines: list[str]) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}"
    print(line)
    lines.append(line)


def prepare_caers_preprocessed_splits(data_dir: Path, seed: int, val_ratio: float, dry_run: bool, logs: list[str]) -> None:
    train_in = data_dir / "train-caers-vad.csv"
    test_in = data_dir / "test-caers-vad.csv"
    train_out = data_dir / "train-caers-vad-preprocessed.csv"
    val_out = data_dir / "val-caers-vad-preprocessed.csv"
    test_out = data_dir / "test-caers-vad-preprocessed.csv"

    if not train_in.exists() or not test_in.exists():
        log_line(
            "[CAER-S] Skipped: expected source CSVs missing (train-caers-vad.csv and/or test-caers-vad.csv).",
            logs,
        )
        return

    train_df = pd.read_csv(train_in)
    test_df = pd.read_csv(test_in)

    if "emotion" in train_df.columns:
        val_parts = []
        keep_parts = []
        for _, group in train_df.groupby("emotion", sort=True):
            shuffled = group.sample(frac=1.0, random_state=seed)
            n_val = max(1, int(round(len(shuffled) * val_ratio)))
            val_parts.append(shuffled.iloc[:n_val])
            keep_parts.append(shuffled.iloc[n_val:])

        val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_out_df = pd.concat(keep_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        shuffled = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_df = shuffled.iloc[:n_val].copy()
        train_out_df = shuffled.iloc[n_val:].copy()

    if dry_run:
        log_line(
            f"[CAER-S] DRY-RUN: would write {len(train_out_df)} train, {len(val_df)} val, {len(test_df)} test rows.",
            logs,
        )
        return

    train_out_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    log_line(f"[CAER-S] Wrote {len(train_out_df)} rows to {train_out}", logs)
    log_line(f"[CAER-S] Wrote {len(val_df)} rows to {val_out}", logs)
    log_line(f"[CAER-S] Wrote {len(test_df)} rows to {test_out}", logs)


def prepare_emotic(data_dir: Path, dry_run: bool, logs: list[str]) -> None:
    emotic_root = data_dir / "emotic_kaggle"

    emotic_outputs = {
        "train": data_dir / "train-emotic-vad-preprocessed.csv",
        "val": data_dir / "val-emotic-vad-preprocessed.csv",
        "test": data_dir / "test-emotic-vad-preprocessed.csv",
    }

    if dry_run:
        log_line("[Emotic] DRY-RUN: would run preprocess_emotic_split for train/val/test.", logs)
        return

    for split, out_path in emotic_outputs.items():
        n = preprocess_emotic_split(emotic_root, split, out_path)
        log_line(f"[Emotic] Wrote {n} rows to {out_path}", logs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full dataset preparation pipeline for reproducibility.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--caers-val-ratio", type=float, default=0.10)
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    logs: list[str] = []

    log_line("Starting full preparation pipeline", logs)
    log_line(f"Data directory: {data_dir.resolve()}", logs)
    log_line(f"Seed: {args.seed}", logs)

    prepare_caers_preprocessed_splits(
        data_dir=data_dir,
        seed=args.seed,
        val_ratio=args.caers_val_ratio,
        dry_run=args.dry_run,
        logs=logs,
    )

    prepare_emotic(
        data_dir=data_dir,
        dry_run=args.dry_run,
        logs=logs,
    )

    log_line("Preparation pipeline completed", logs)

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"prepare_all_datasets_{stamp}.log"

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(logs) + "\n", encoding="utf-8")
        print(f"Saved log to {log_path}")
    except OSError as exc:
        print(f"Warning: could not write preparation log to {log_path}: {exc}")


if __name__ == "__main__":
    main()
