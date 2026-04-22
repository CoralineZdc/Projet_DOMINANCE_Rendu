# Documentation Index

This folder collects the project notes that support reproducibility, licensing review, and evaluation.

## Start Here

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): canonical commands, seed handling, and output layout
- [NORMALIZATION_STATS.md](NORMALIZATION_STATS.md): per-dataset image and label means used by the loader
- [EVALUATION_README.md](EVALUATION_README.md): evaluation metrics and usage notes
- [DATASET_LICENSES.md](DATASET_LICENSES.md): dataset rights, attribution, and release constraints
- [MODEL_LICENSES.md](MODEL_LICENSES.md): pretrained weight and checkpoint guidance

## Release Policy Summary

1. Publish code, configs, metrics, and documentation.
2. Do not publish raw third-party images, annotations, or merged row-level CSV datasets unless the source terms clearly allow it.
3. Treat checkpoints as constrained by the strictest applicable dataset and upstream model terms.
4. Keep dataset and model attribution with any public release.

## Notes

- The repository root [README.md](../README.md) provides the main project overview and commands.
- If you change the preprocessing or training pipeline, update [REPRODUCIBILITY.md](REPRODUCIBILITY.md) at the same time.
