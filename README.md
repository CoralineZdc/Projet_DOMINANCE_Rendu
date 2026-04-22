# VAD-Net

VAD-Net is a Valence-Arousal-Dominance regression project for facial-expression datasets.
It includes data preparation, model training, evaluation, log analysis, and plotting tools.

## Quick Start

1. Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

2. Prepare the auxiliary derived CSVs and splits:

```bash
python prepare_data.py
```

3. Train a single model:

```bash
python train.py --model resnet18 --seed 42 --output_dir runs/MyRun
```

4. Train all supported model/dataset combinations:

```bash
python train_all.py --datasets all --models resnet18,resnet50,efficientnet,mobilefacenet --seed 42 --only-missing
```

5. Evaluate a checkpoint:

```bash
python evaluate.py --model runs/MyRun/best_model_state.pth --cuda
```

6. Evaluate every saved run under `runs/`:

```bash
python evaluate_all.py --runs-root runs --output-root evaluations
```

The batch evaluator also writes `evaluations/all_loss_curves.png` with the training and public_loss curves for every run that has a `log.csv` file.
Each evaluated run also gets its own `evaluations/<run_name>/loss_curves.png` plot.

7. Inspect or plot training logs:

```bash
python analyze_training.py --mode display runs/MyRun/train_log.csv
python analyze_training.py --mode plot runs/MyRun/train_log.csv
```

## Repository Layout

- `src/data/`: dataset preparation and preprocessing
- `src/training/`: training scripts and batch launchers
- `src/evaluation/`: evaluation, plotting, and training-log analysis
- `src/utils/`: shared dataset and utility modules
- `docs/`: licensing, reproducibility, and evaluation notes

## Reproducibility

The project is designed around fixed seeds, explicit output folders, and stable CSV inputs.
See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the canonical commands, seed handling, and output layout.
For the dataset-level normalization constants used by the loader and inference code, see [docs/NORMALIZATION_STATS.md](docs/NORMALIZATION_STATS.md).

## Licensing Summary

Repository code is MIT-licensed. The datasets and pretrained model weights used by the project have their own terms and must be handled separately.

Verified notes:

- FER2013: DbCL / Open Database Contents style terms
- Balanced Caer-S: ODbL 1.0
- Emotic: open source, but citation is required
- ResNet18 pretrained weights: CC0 Public Domain
- ResNet50 pretrained weights: CC0 Public Domain
- MobileFaceNet: Apache 2.0 upstream license
- EfficientNet: preserve the upstream LICENSE from the source repository

Full details are documented in:

- [docs/DATASET_LICENSES.md](docs/DATASET_LICENSES.md)
- [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)

## Release Policy

1. Publish code, configs, metrics, and documentation.
2. Do not publish raw third-party images, annotations, or merged row-level CSV datasets unless the source terms clearly allow it.
3. Treat checkpoints as constrained by the strictest applicable dataset and upstream model terms.
4. Include required citations, attribution, and notice text where applicable.

## Documentation Index

- [docs/README.md](docs/README.md)
- [docs/NORMALIZATION_STATS.md](docs/NORMALIZATION_STATS.md)
- [docs/EVALUATION_README.md](docs/EVALUATION_README.md)
- [docs/DATASET_LICENSES.md](docs/DATASET_LICENSES.md)
- [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
