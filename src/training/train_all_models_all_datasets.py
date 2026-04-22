"""Batch launcher for training every supported model on every supported dataset preset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_PRESETS = {
    "fer2013": {
        "display": "FER2013",
        "train_csv": "data/train-fer2013-vad.csv",
        "public_csv": "data/test-fer2013-vad.csv",
        "private_csv": "data/val-fer2013-vad.csv",
    },
    "caers": {
        "display": "Caer-S",
        "train_csv": "data/train-caers-vad.csv",
        "public_csv": "data/test-caers-vad.csv",
        "private_csv": "data/test-caers-vad.csv",
    },
    "caers_preprocessed": {
        "display": "Caer-S-Preprocessed",
        "train_csv": "data/train-caers-vad-preprocessed.csv",
        "public_csv": "data/val-caers-vad-preprocessed.csv",
        "private_csv": "data/test-caers-vad-preprocessed.csv",
    },
    "emotic": {
        "display": "Emotic",
        "train_csv": "data/train-emotic-vad.csv",
        "public_csv": "data/test-emotic-vad.csv",
        "private_csv": "data/test-emotic-vad.csv",
    },
    "emotic_preprocessed": {
        "display": "Emotic-Preprocessed",
        "train_csv": "data/train-emotic-vad-preprocessed.csv",
        "public_csv": "data/val-emotic-vad-preprocessed.csv",
        "private_csv": "data/test-emotic-vad-preprocessed.csv",
    },
}

MODEL_CHOICES = ["resnet18", "resnet50", "efficientnet", "mobilefacenet"]
MODEL_DISPLAY_NAMES = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "efficientnet": "EfficientNet",
    "mobilefacenet": "MobileFaceNet",
}


def parse_csv_list(value: str) -> list[str]:
    """Split a comma-separated CLI argument into a clean list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_required_files(dataset_name: str, preset: dict[str, str]) -> None:
    """Fail fast if the required CSV files for a preset are missing."""
    missing = [key for key in ("train_csv", "public_csv", "private_csv") if not Path(preset[key]).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset preset '{dataset_name}' is missing required files: "
            + ", ".join(f"{key}={preset[key]}" for key in missing)
        )


def run_is_completed(run_dir: Path) -> bool:
    """Return True when a run folder already contains the expected artifacts."""
    log_path = run_dir / "log.csv"
    best_state = run_dir / "best_model_state.pth"
    if not log_path.exists() or not best_state.exists():
        return False
    if log_path.stat().st_size <= 0:
        return False
    return True


def main() -> None:
    """Parse CLI arguments and launch all requested training jobs."""
    parser = argparse.ArgumentParser(description="Train every supported model on every supported dataset preset.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset presets or 'all'.")
    parser.add_argument("--models", default=",".join(MODEL_CHOICES), help="Comma-separated model names.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ccc-weight", type=float, default=0.05)
    parser.add_argument("--lr-backbone-mult", type=float, default=0.2)
    parser.add_argument("--lr-head-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--aug-profile", choices=["none", "light", "medium", "strong"], default="light")
    parser.add_argument("--train-crop-padding", type=int, default=2)
    parser.add_argument("--rotation-deg", type=float, default=3.0)
    parser.add_argument("--sampler-weight", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true", help="Pass --pretrained to compatible models.")
    parser.add_argument("--only-missing", action="store_true", help="Train only combinations without completed artifacts.")
    parser.add_argument("--force", action="store_true", help="Train all selected combinations even if artifacts already exist.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    if args.datasets == "all":
        dataset_names = list(DATASET_PRESETS.keys())
    else:
        dataset_names = parse_csv_list(args.datasets)

    model_names = parse_csv_list(args.models)
    unknown_datasets = [name for name in dataset_names if name not in DATASET_PRESETS]
    unknown_models = [name for name in model_names if name not in MODEL_CHOICES]
    if unknown_datasets:
        raise ValueError("Unknown dataset presets: {}".format(", ".join(unknown_datasets)))
    if unknown_models:
        raise ValueError("Unknown model names: {}".format(", ".join(unknown_models)))

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        preset = DATASET_PRESETS[dataset_name]
        resolved_preset = {
            key: str((repo_root / value).resolve()) if key.endswith("csv") else value
            for key, value in preset.items()
        }
        validate_required_files(dataset_name, resolved_preset)

        for model_name in model_names:
            run_dir_name = f"{preset['display']}_{MODEL_DISPLAY_NAMES[model_name]}"
            run_dir = output_root / run_dir_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if args.only_missing and not args.force and run_is_completed(run_dir):
                print(f"Skipping completed run: {run_dir_name}")
                continue

            cmd = [
                args.python,
                str(Path(__file__).parent / "mainpro_FER.py"),
                "--model",
                model_name,
                "--seed",
                str(args.seed),
                "--output_dir",
                str(run_dir),
                "--optimizer",
                args.optimizer,
                "--lr",
                str(args.lr),
                "--bs",
                str(args.bs),
                "--epochs",
                str(args.epochs),
                "--dropout",
                str(args.dropout),
                "--ccc_weight",
                str(args.ccc_weight),
                "--lr_backbone_mult",
                str(args.lr_backbone_mult),
                "--lr_head_mult",
                str(args.lr_head_mult),
                "--weight_decay",
                str(args.weight_decay),
                "--aug_profile",
                args.aug_profile,
                "--train_crop_padding",
                str(args.train_crop_padding),
                "--rotation_deg",
                str(args.rotation_deg),
                "--sampler_weight",
                str(args.sampler_weight),
                "--mixup_alpha",
                str(args.mixup_alpha),
                "--lr_patience",
                str(args.lr_patience),
                "--early_stop_patience",
                str(args.early_stop_patience),
                "--train_csv",
                str((repo_root / preset["train_csv"]).resolve()),
                "--public_csv",
                str((repo_root / preset["public_csv"]).resolve()),
                "--private_csv",
                str((repo_root / preset["private_csv"]).resolve()),
            ]

            if args.pretrained and model_name == "resnet50":
                cmd.append("--pretrained")

            command_text = " ".join(cmd)
            try:
                (run_dir / "command.txt").write_text(command_text, encoding="utf-8")
            except OSError as exc:
                print(f"Warning: could not write command.txt for {run_dir_name}: {exc}")

            print(command_text)
            if args.dry_run:
                continue

            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()