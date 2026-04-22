# Normalization Statistics

This page records the dataset-level statistics used by the current FER2013 dataset loader and related inference code.

The values below match the training-split statistics computed from the CSV files in this repository.

## How These Values Are Computed

- `mean_images`: average pixel intensity over the training CSV, scaled to `[0, 1]`, then repeated for three channels.
- `std_images`: standard deviation of pixel intensity over the training CSV, scaled to `[0, 1]`, then repeated for three channels.
- `mean_labels`: per-dimension average of `Valence`, `Arousal`, and `Dominance` over the training CSV.
- `std_labels`: per-dimension standard deviation of `Valence`, `Arousal`, and `Dominance` over the training CSV.

## Per-Dataset Values

### FER2013

- `mean_images = [0.50789523, 0.50789523, 0.50789523]`
- `std_images = [0.25496972, 0.25496972, 0.25496972]`
- `mean_labels = [-0.31888336, 0.46993691, 0.09596027]`
- `std_labels = [1.32287991, 1.28252947, 1.35678995]`

### Caer-S

- `mean_images = [0.23313367, 0.23313367, 0.23313367]`
- `std_images = [0.17864832, 0.17864832, 0.17864832]`
- `mean_labels = [-0.60000002, 0.74285716, 0.08571435]`
- `std_labels = [1.26491106, 0.81915838, 0.63116348]`

### Caer-S-Preprocessed

- `mean_images = [0.23328513, 0.23328513, 0.23328513]`
- `std_images = [0.17864509, 0.17864509, 0.17864509]`
- `mean_labels = [-0.60000002, 0.74285716, 0.08571429]`
- `std_labels = [1.26491106, 0.81915838, 0.63116354]`

### Emotic

- `mean_images = [0.42931858, 0.42931858, 0.42931858]`
- `std_images = [0.26607814, 0.26607814, 0.26607814]`
- `mean_labels = [0.22601773, 0.03438406, 0.43962234]`
- `std_labels = [0.59363019, 0.88679868, 0.78606725]`

### Emotic-Preprocessed

- `mean_images = [0.40602905, 0.40602905, 0.40602905]`
- `std_images = [0.25468904, 0.25468904, 0.25468904]`
- `mean_labels = [0.40394977, 0.21263959, 0.60898191]`
- `std_labels = [0.54638034, 0.82082510, 0.73205173]`

## Notes

- `mean_labels` and `std_labels` are only meaningful when paired with the same dataset split and loader settings used to compute them.
- If you regenerate a dataset CSV, recompute these values from the new file before reusing saved checkpoints.