import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DATASET_PRESETS = {
    "fer2013": {
        "train_csv": "data/train-fer2013-vad.csv",
        "public_csv": "data/test-fer2013-vad.csv",
        "private_csv": "data/val-fer2013-vad.csv",
    },
    "caers": {
        "train_csv": "data/train-caers-vad.csv",
        "public_csv": "data/test-caers-vad.csv",
        "private_csv": "data/test-caers-vad.csv",
    },
    "emotic": {
        "train_csv": "data/train-emotic-vad.csv",
        "public_csv": "data/test-emotic-vad.csv",
        "private_csv": "data/test-emotic-vad.csv",
    },
}


def parse_csv_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def sanitize_name(value):
    out = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def parse_best_metrics(log_path):
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None, None, None

    best_rmse = None
    best_epoch = None
    last_rmse = None

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row["epoch"])
                rmse = float(row["pub_rmse_mean"])
            except Exception:
                continue

            last_rmse = rmse
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch

    return best_rmse, best_epoch, last_rmse


def main():
    parser = argparse.ArgumentParser(description="Grid search across dataset presets with full log archival")
    parser.add_argument("--datasets", default="all", help="Comma list from: {} or 'all'".format(",".join(DATASET_PRESETS.keys())))
    parser.add_argument("--models", default="resnet50")
    parser.add_argument("--lrs", default="8e-5")
    parser.add_argument("--batch-sizes", default="64")
    parser.add_argument("--dropouts", default="0.2")
    parser.add_argument("--ccc-weights", default="0.05")
    parser.add_argument("--lr-backbone-mults", default="0.2")
    parser.add_argument("--lr-head-mults", default="1.0")
    parser.add_argument("--weight-decays", default="1e-5")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--aug-profile", default="light", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--train-crop-padding", type=int, default=2)
    parser.add_argument("--rotation-deg", type=float, default=3.0)
    parser.add_argument("--sampler-weight", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--runs-root", default="gridsearch_runs")
    parser.add_argument("--tag", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.datasets == "all":
        datasets = list(DATASET_PRESETS.keys())
    else:
        datasets = parse_csv_strings(args.datasets)
        unknown = [d for d in datasets if d not in DATASET_PRESETS]
        if unknown:
            raise ValueError("Unknown dataset presets: {}".format(", ".join(unknown)))

    models = parse_csv_strings(args.models)
    lrs = parse_csv_floats(args.lrs)
    batch_sizes = parse_csv_ints(args.batch_sizes)
    dropouts = parse_csv_floats(args.dropouts)
    ccc_weights = parse_csv_floats(args.ccc_weights)
    lr_backbone_mults = parse_csv_floats(args.lr_backbone_mults)
    lr_head_mults = parse_csv_floats(args.lr_head_mults)
    weight_decays = parse_csv_floats(args.weight_decays)

    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_name = "grid-{}".format(stamp) if not args.tag else "grid-{}-{}".format(stamp, sanitize_name(args.tag))
    session_dir = root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    summary_path = session_dir / "summary.csv"
    summary_fields = [
        "run_id",
        "dataset",
        "model",
        "lr",
        "batch_size",
        "dropout",
        "ccc_weight",
        "lr_backbone_mult",
        "lr_head_mult",
        "weight_decay",
        "epochs",
        "status",
        "return_code",
        "best_pub_rmse_mean",
        "best_epoch",
        "last_pub_rmse_mean",
        "duration_sec",
        "run_dir",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

    grid = list(
        itertools.product(
            datasets,
            models,
            lrs,
            batch_sizes,
            dropouts,
            ccc_weights,
            lr_backbone_mults,
            lr_head_mults,
            weight_decays,
        )
    )

    print("Session directory:", session_dir)
    print("Total runs:", len(grid))

    run_counter = 0
    for ds_name, model, lr, bs, dropout, ccc_w, bb_mult, hd_mult, wd in grid:
        run_counter += 1
        preset = DATASET_PRESETS[ds_name]

        run_name = "{:03d}_{}_{}_lr{}_bs{}_do{}_ccc{}_bb{}_hd{}_wd{}".format(
            run_counter,
            ds_name,
            model,
            lr,
            bs,
            dropout,
            ccc_w,
            bb_mult,
            hd_mult,
            wd,
        )
        run_name = sanitize_name(run_name)
        run_dir = session_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Keep each run independent by clearing trainer's default log destination.
        main_log = Path("FER2013_ResNet") / "log.csv"
        if main_log.exists():
            main_log.unlink()

        for artifact_name in ["checkpoint.pth", "best_model.pth", "best_model_state.pth", "last_model.pth"]:
            artifact_path = Path("FER2013_ResNet") / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()

        cmd = [
            args.python,
            str(Path(__file__).parent / "mainpro_FER.py"),
            "--model", model,
            "--pretrained",
            "--optimizer", args.optimizer,
            "--lr", str(lr),
            "--bs", str(bs),
            "--epochs", str(args.epochs),
            "--dropout", str(dropout),
            "--ccc_weight", str(ccc_w),
            "--lr_backbone_mult", str(bb_mult),
            "--lr_head_mult", str(hd_mult),
            "--weight_decay", str(wd),
            "--aug_profile", args.aug_profile,
            "--train_crop_padding", str(args.train_crop_padding),
            "--rotation_deg", str(args.rotation_deg),
            "--sampler_weight", str(args.sampler_weight),
            "--mixup_alpha", str(args.mixup_alpha),
            "--lr_patience", str(args.lr_patience),
            "--early_stop_patience", str(args.early_stop_patience),
            "--train_csv", preset["train_csv"],
            "--public_csv", preset["public_csv"],
            "--private_csv", preset["private_csv"],
        ]

        try:
            (run_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
        except OSError as exc:
            print(f"Warning: could not write command.txt for {run_name}: {exc}")

        print("\n[Run {}/{}] {}".format(run_counter, len(grid), run_name))
        if args.dry_run:
            status = "dry-run"
            rc = 0
            duration = 0.0
            best_rmse = None
            best_epoch = None
            last_rmse = None
            stdout_text = ""
            stderr_text = ""
        else:
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start
            rc = proc.returncode
            status = "ok" if rc == 0 else "failed"
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""

            try:
                (run_dir / "stdout.log").write_text(stdout_text, encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"Warning: could not write stdout.log for {run_name}: {exc}")
            try:
                (run_dir / "stderr.log").write_text(stderr_text, encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"Warning: could not write stderr.log for {run_name}: {exc}")

            if main_log.exists():
                shutil.copy2(main_log, run_dir / "train_log.csv")
                best_rmse, best_epoch, last_rmse = parse_best_metrics(run_dir / "train_log.csv")
            else:
                best_rmse, best_epoch, last_rmse = None, None, None

            for artifact_name in ["checkpoint.pth", "best_model.pth", "best_model_state.pth", "last_model.pth"]:
                artifact_path = Path("FER2013_ResNet") / artifact_name
                if artifact_path.exists():
                    shutil.copy2(artifact_path, run_dir / artifact_name)

            meta = {
                "dataset": ds_name,
                "preset": preset,
                "status": status,
                "return_code": rc,
                "duration_sec": duration,
                "best_pub_rmse_mean": best_rmse,
                "best_epoch": best_epoch,
                "last_pub_rmse_mean": last_rmse,
            }
            try:
                (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except OSError as exc:
                print(f"Warning: could not write meta.json for {run_name}: {exc}")

            if rc != 0:
                print("  -> FAILED (rc={})".format(rc))
            else:
                print("  -> OK (best_pub_rmse_mean={})".format(best_rmse))

        row = {
            "run_id": run_counter,
            "dataset": ds_name,
            "model": model,
            "lr": lr,
            "batch_size": bs,
            "dropout": dropout,
            "ccc_weight": ccc_w,
            "lr_backbone_mult": bb_mult,
            "lr_head_mult": hd_mult,
            "weight_decay": wd,
            "epochs": args.epochs,
            "status": status,
            "return_code": rc,
            "best_pub_rmse_mean": best_rmse,
            "best_epoch": best_epoch,
            "last_pub_rmse_mean": last_rmse,
            "duration_sec": duration,
            "run_dir": str(run_dir),
        }

        with summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writerow(row)

    print("\nDone. Summary:", summary_path)


if __name__ == "__main__":
    main()
