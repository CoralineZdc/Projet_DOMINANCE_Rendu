# Evaluation Guide

The evaluation entry point is [evaluate.py](../evaluate.py), which loads a trained checkpoint and runs the canonical evaluation pipeline in [src/evaluation/evaluation.py](../src/evaluation/evaluation.py).

This evaluation workflow computes **RMSE**, **CCC**, and **Confusion Matrices** for VAD-Net models.

## Features

### 1. **RMSE (Root Mean Squared Error)**
- Computed separately for Valence, Arousal, and Dominance
- Overall RMSE across all dimensions
- Compares against paper baseline results

### 2. **CCC (Concordance Correlation Coefficient)**
- Measures agreement between predictions and ground truth
- Range: -1 to 1 (1 = perfect agreement)
- Computed for each dimension and overall
- More representative than Pearson correlation for evaluating agreement

### 3. **Confusion Matrices**
- **Emotion Classification**: Valence+Arousal discretized into 4 emotion classes:
  - Negative (low V, high A) = Anger, Fear
  - Negative-Calm (low V, low A) = Sadness, Disgust
  - Positive (high V, high A) = Joy, Surprise
  - Positive-Calm (high V, low A) = Contentment
- **Valence Classification**: Negative vs Positive
- **Arousal Classification**: Low vs High
- Separate visualizations for Public and Private test sets

### 4. **Summary Report**
- Text file with all metrics (RMSE, CCC)
- Saved to `evaluation_results/evaluation_summary.txt`

## Installation Requirements

Install required packages:
```bash
pip install seaborn scikit-learn matplotlib
```

Or if you're using the existing environment, add to requirements:
```bash
pip install seaborn scikit-learn
```

## Usage

### Basic Usage
```bash
python evaluate.py --model runs/MyRun/best_model_state.pth
```

### With Output Directory
```bash
python evaluate.py --model runs/MyRun/best_model_state.pth --output_dir my_evaluation
```

### With CUDA
```bash
python evaluate.py --model runs/MyRun/best_model_state.pth --cuda
```

### Full Options
```bash
python evaluate.py \
  --model runs/MyRun/best_model_state.pth \
  --cuda \
  --cut_size 48 \
  --input_size 48 \
  --output_dir evaluation_results
```

## Reproducibility Notes

- Evaluate the same checkpoint against the same CSV inputs when comparing runs.
- Keep the output directory stable if you want comparable evaluation artifacts.
- If you change preprocessing or seed settings, regenerate the derived CSVs first.

## Output Structure

```
evaluation_results/
├── public_test/
│   └── confusion_matrices.png
├── private_test/
│   └── confusion_matrices.png
└── evaluation_summary.txt
```

## Metrics Explanation

### CCC Formula
```
CCC = (2 * ρ * σ_x * σ_y) / (σ_x² + σ_y² + (μ_x - μ_y)²)

where:
- ρ = Pearson correlation coefficient
- σ_x, σ_y = standard deviations
- μ_x, μ_y = means
```

### RMSE
```
RMSE = √(1/N * Σ(pred_i - target_i)²)
```

## Interpreting Results

**RMSE**: Lower is better (0 = perfect)
- < 0.15 overall: Excellent performance
- 0.15 - 0.25: Good performance
- > 0.25: Needs improvement

**CCC**: Closer to 1 is better
- > 0.8: Excellent agreement
- 0.6 - 0.8: Good agreement
- < 0.6: Fair/poor agreement

**Confusion Matrix**: High diagonal values indicate good classification of discretized emotions
