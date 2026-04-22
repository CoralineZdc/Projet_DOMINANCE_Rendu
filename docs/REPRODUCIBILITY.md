# Reproducibility Guide

This project includes data preparation, training, evaluation, and analysis scripts for VAD regression on facial-expression datasets.

## Core Reproducibility Rules

1. Use the same dataset CSV inputs when comparing experiments.
2. Fix the random seed with `--seed` whenever you rerun training or preprocessing.
3. Keep run outputs in separate folders so checkpoints and logs are not overwritten.
4. Record the exact command line used for each experiment.
5. Preserve the dataset and model license notes in [DATASET_LICENSES.md](DATASET_LICENSES.md) and [MODEL_LICENSES.md](MODEL_LICENSES.md).

## Recommended Environment

- Python 3.12
- Pinned package versions from [requirements.txt](../requirements.txt)
- The same CUDA-capable PyTorch wheel family used by the repository environment

If you restore the environment later, install the same package versions before rerunning experiments.

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Deterministic Behavior

The main training script seeds:
- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- cuDNN deterministic settings

That behavior is controlled by `--seed` in the training launcher.

## Canonical Commands

### Prepare auxiliary datasets

```bash
python prepare_data.py
```

This runs the consolidated pipeline that:
- splits CAER-S into train/val/test
- preprocesses Emotic into 48x48 grayscale CSVs

### Train one model

```bash
python train.py --model resnet18 --seed 42 --output_dir runs/MyRun
```

### Train all models and datasets

```bash
python train_all.py --datasets all --models resnet18,resnet50,efficientnet,mobilefacenet --seed 42 --only-missing
```

### Evaluate a trained checkpoint

```bash
python evaluate.py --model runs/MyRun/best_model_state.pth --cuda
```

### Evaluate every saved run

```bash
python evaluate_all.py --runs-root runs --output-root evaluations
```

That command also generates `evaluations/all_loss_curves.png`, which overlays the training and public-loss curves for every run with a `log.csv` file.
Each run folder under `evaluations/` also receives its own `loss_curves.png`.

### Run the smoke test

```bash
python scripts/smoke_test.py
```

### Inspect training logs

```bash
python analyze_training.py --mode display runs/MyRun/train_log.csv
python analyze_training.py --mode plot runs/MyRun/train_log.csv
```

## Expected Output Structure

- `data/`: derived CSV files and dataset artifacts
- `runs/`: one folder per dataset/model combination
- `logs/`: orchestration logs
- `plots/`: generated figures

## Naming Conventions

- Use `Dataset_Model` for run folders.
- Keep one model checkpoint per run folder.
- Keep log files next to the corresponding checkpoint.

## Experiment Checklist

1. Record the dataset CSVs used.
2. Record the model architecture and pretrained backbone choice.
3. Record the seed value.
4. Record the optimizer, learning rate, batch size, and epoch count.
5. Record the output folder name.
6. Keep the exact evaluation checkpoint path.

## Notes

- If you change the preprocessing pipeline, regenerate the derived CSV files before rerunning training.
- If you change the seed, the splits and batch order can change, so do not compare the results directly to a previous run unless that change is intentional.
- For public release, keep the license guidance in the dataset/model docs aligned with the exact upstream terms you verified.
