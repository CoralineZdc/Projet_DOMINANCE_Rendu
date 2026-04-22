"""
Comprehensive evaluation script for VAD-Net models.

Outputs:
1. RMSE per VAD dimension and overall RMSE.
2. CCC per VAD dimension and overall CCC.
3. Confusion matrices derived from discretized VAD values.

Keep the same checkpoint and CSV inputs when comparing runs to preserve reproducibility.
"""

import sys
import os
from pathlib import Path

# Add parent directories to Python path to allow imports from root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Root directory

import torch
import torch.nn as nn
import numpy as np
import transforms as transforms
from src.utils.fer import FER2013
from models.efficientnet_b0 import EfficientNetVAD
from models.mobilefacenet import MobileFaceNetVAD
from models.resnet_reg2 import (
    ResNet18RegressionThreeOutputs,
    ResNet50PretrainedRegressionThreeOutputs,
    ResNet50RegressionThreeOutputs,
)
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


EVAL_INPUT_SIZE = 48


def custom_transform(crops):
    """Convert crops to tensor and normalize."""
    normalize = transforms.Normalize(
        mean=FER2013.image_mean.tolist(),
        std=FER2013.image_std.tolist()
    )
    processed = []
    for crop in crops:
        if crop.size != (EVAL_INPUT_SIZE, EVAL_INPUT_SIZE):
            crop = crop.resize((EVAL_INPUT_SIZE, EVAL_INPUT_SIZE), Image.BILINEAR)
        processed.append(normalize(transforms.ToTensor()(crop)))
    return torch.stack(processed)


def infer_checkpoint_arch(state_dict):
    keys = list(state_dict.keys())

    if any(k.startswith("backbone._conv_stem") for k in keys):
        return "efficientnet"
    if any(k.startswith("backbone.") for k in keys):
        return "resnet50_pretrained"
    if any("layer1.0.conv3.weight" in k for k in keys):
        return "resnet50"
    if any(k.startswith("regression_head") for k in keys):
        return "mobilefacenet"
    return "resnet18"


def build_model_from_state_dict(state_dict):
    arch_hints = []
    inferred_arch = infer_checkpoint_arch(state_dict)
    arch_hints.append(inferred_arch)

    for arch in ["efficientnet", "mobilefacenet", "resnet50_pretrained", "resnet50", "resnet18"]:
        if arch not in arch_hints:
            arch_hints.append(arch)

    builders = {
        "resnet18": lambda: ResNet18RegressionThreeOutputs(),
        "resnet50": lambda: ResNet50RegressionThreeOutputs(),
        "resnet50_pretrained": lambda: ResNet50PretrainedRegressionThreeOutputs(),
        "efficientnet": lambda: EfficientNetVAD(dropout_rate=0.0),
        "mobilefacenet": lambda: MobileFaceNetVAD(dropout_rate=0.0),
    }

    last_error = None
    for arch in arch_hints:
        try:
            model = builders[arch]()
            model.load_state_dict(state_dict)
            return model, arch
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to load checkpoint into any supported architecture: {last_error}")


def compute_rmse(model, dataloader, dimension_names=['Valence', 'Arousal', 'Dominance'], use_cuda=False, max_batches=None):
    """
    Compute RMSE for each dimension separately.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        dimension_names: List of names for each dimension
        use_cuda: Whether to use CUDA
    
    Returns:
        rmse_per_dim: Dictionary with RMSE for each dimension
        overall_rmse: Overall RMSE across all dimensions
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(dataloader):
            # TenCrop produces an extra crop dimension; flatten it before the forward pass.
            if len(inputs.shape) == 5:  # (batch, 10crops, c, h, w)
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)
                
                if use_cuda:
                    inputs = inputs.cuda()
                
                outputs = model(inputs)
                # Average predictions across crops so each image contributes one estimate.
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            else:
                if use_cuda:
                    inputs = inputs.cuda()
                outputs_avg = model(inputs)
            
            all_predictions.append(outputs_avg.cpu().numpy())
            all_targets.append(targets.numpy())

            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 3)
    targets = np.concatenate(all_targets, axis=0)  # (N, 3)

    # Convert normalized targets/predictions back to the original VAD scale before RMSE/CCC.
    label_mean = FER2013.label_mean.numpy()
    label_std = FER2013.label_std.numpy()
    predictions = predictions * label_std + label_mean
    targets = targets * label_std + label_mean
    
    rmse_per_dim = {}
    for i, dim_name in enumerate(dimension_names):
        mse = np.mean((predictions[:, i] - targets[:, i]) ** 2)
        rmse = np.sqrt(mse)
        rmse_per_dim[dim_name] = rmse
    
    # Overall RMSE
    overall_mse = np.mean((predictions - targets) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    
    return rmse_per_dim, overall_rmse, predictions, targets


def compute_ccc(predictions, targets, dimension_names=['Valence', 'Arousal', 'Dominance']):
    """
    Compute Concordance Correlation Coefficient (CCC) for each dimension.
    CCC measures the agreement between two continuous variables.
    
    Args:
        predictions: (N, 3) numpy array of model predictions
        targets: (N, 3) numpy array of ground truth targets
        dimension_names: List of names for each dimension
    
    Returns:
        ccc_per_dim: Dictionary with CCC for each dimension
        overall_ccc: Mean CCC across all dimensions
    """
    ccc_per_dim = {}
    ccc_values = []
    
    for i, dim_name in enumerate(dimension_names):
        pred = predictions[:, i]
        targ = targets[:, i]
        
        # CCC combines correlation, variance, and mean shift into one agreement score.
        mu_x = np.mean(pred)
        mu_y = np.mean(targ)
        
        sigma_x = np.std(pred)
        sigma_y = np.std(targ)
        
        if sigma_x == 0 or sigma_y == 0:
            ccc = 0.0
        else:
            # Pearson correlation coefficient
            rho = np.corrcoef(pred, targ)[0, 1]
            
            # CCC computation
            numerator = 2 * rho * sigma_x * sigma_y
            denominator = sigma_x**2 + sigma_y**2 + (mu_x - mu_y)**2
            
            ccc = numerator / denominator if denominator != 0 else 0.0
        
        ccc_per_dim[dim_name] = ccc
        ccc_values.append(ccc)
    
    overall_ccc = np.mean(ccc_values)
    
    return ccc_per_dim, overall_ccc


def discretize_vad(values, num_classes=2):
    """
    Discretize VAD values into multiple classes based on quantile-based or fixed boundaries.
    
    Args:
        values: (N,) numpy array of values in range [-2, 2]
        num_classes: 2, 3, or 4
    
    Returns:
        classes: (N,) integer array with class labels [0, 1, ..., num_classes-1]
    """
    values = np.asarray(values)
    
    if num_classes == 2:
        # Binary split around zero keeps the interpretation simple and stable.
        return (values >= 0).astype(int)
    
    elif num_classes == 3:
        # Three bins give a coarse low/neutral/high view.
        classes = np.zeros_like(values, dtype=int)
        classes[values >= 0.67] = 2  # High
        classes[(values >= -0.67) & (values < 0.67)] = 1  # Medium
        classes[values < -0.67] = 0  # Low
        return classes
    
    elif num_classes == 4:
        # Four bins are the finest view used here and are useful for sharper confusion plots.
        classes = np.zeros_like(values, dtype=int)
        classes[values >= 1.0] = 3   # Very High
        classes[(values >= 0) & (values < 1.0)] = 2    # High
        classes[(values >= -1.0) & (values < 0)] = 1   # Low
        classes[values < -1.0] = 0   # Very Low
        return classes
    
    else:
        raise ValueError(f"num_classes must be 2, 3, or 4, got {num_classes}")


def get_class_labels(num_classes, dim_name='Valence'):
    """Get class labels for a given number of classes and dimension."""
    if num_classes == 2:
        if dim_name == 'Valence':
            return ['Negative', 'Positive']
        else:
            return ['Low', 'High']
    
    elif num_classes == 3:
        if dim_name == 'Valence':
            return ['Negative', 'Neutral', 'Positive']
        else:
            return ['Low', 'Medium', 'High']
    
    elif num_classes == 4:
        if dim_name == 'Valence':
            return ['Very Negative', 'Negative', 'Positive', 'Very Positive']
        else:
            return ['Very Low', 'Low', 'High', 'Very High']
    
    else:
        raise ValueError(f"num_classes must be 2, 3, or 4, got {num_classes}")


def plot_confusion_matrices(predictions, targets, output_dir='evaluation_results'):
    """
    Plot fine-grained confusion matrices (4-class discretization) for each VAD dimension.
    
    4-class discretization: Very Low / Low / High / Very High
    Boundaries at -1, 0, 1
    
    Args:
        predictions: (N, 3) numpy array of model predictions
        targets: (N, 3) numpy array of ground truth targets
        output_dir: Directory to save plots
    
    Returns:
        None (saves PNG files)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dim_names = ['Valence', 'Arousal', 'Dominance']
    colormaps = ['Blues', 'Oranges', 'Greens']
    num_classes = 4
    
    # One row per dimension keeps the comparison aligned across V, A, and D.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def draw_confusion_matrix(ax, cm, labels, title, cmap):
        """Draw an annotated confusion matrix using matplotlib only."""
        image = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0)
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Count')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        threshold = cm.max() / 2.0 if cm.size else 0.0
        for row_idx in range(cm.shape[0]):
            for col_idx in range(cm.shape[1]):
                value = cm[row_idx, col_idx]
                color = 'white' if value > threshold else 'black'
                ax.text(col_idx, row_idx, format(int(value), 'd'), ha='center', va='center', color=color, fontsize=11)
    
    for dim_idx in range(3):
        dim_name = dim_names[dim_idx]
        cmap = colormaps[dim_idx]
        ax = axes[dim_idx]
        
        # Use the same discretization boundaries for predictions and labels.
        pred_class = discretize_vad(predictions[:, dim_idx], num_classes=num_classes)
        targ_class = discretize_vad(targets[:, dim_idx], num_classes=num_classes)
        
        # Build confusion matrix
        cm = confusion_matrix(targ_class, pred_class, labels=list(range(num_classes)))
        
        # Get class labels
        class_labels = get_class_labels(num_classes, dim_name)

        draw_confusion_matrix(ax, cm, class_labels, f'{dim_name} (4-class)', cmap)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrices saved to {output_path}")
    plt.close()


def evaluate_all_sets(model_path, use_cuda=False, cut_size=48, input_size=48, align_faces=False, output_dir='evaluation_results', max_batches=None, batch_size=128):
    """Evaluate model on both public and private test sets. Compute RMSE, CCC, and confusion matrices."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Support both raw state_dict files and checkpoint dictionaries.
    try:
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            # It's a checkpoint dict
            print(f"Loaded checkpoint format (epoch {state.get('epoch', 'unknown')})")
            state_dict = state['model']
        else:
            # Direct state dict
            print(f"Loaded direct state_dict")
            state_dict = state
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        net, resolved_arch = build_model_from_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model into a supported architecture: {e}")
        return

    # Confirm the resolved architecture matches the stored weight layout.
    print(f"[OK] Model loaded correctly with inferred architecture: {resolved_arch}\n")
    
    if use_cuda:
        net.cuda()

    FER2013._ensure_label_stats()
    FER2013._ensure_image_stats()

    global EVAL_INPUT_SIZE
    EVAL_INPUT_SIZE = input_size
    
    # Keep evaluation deterministic by using the same transform path for both splits.
    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        custom_transform,
    ])
    
    pubset = FER2013(split='PublicTest', transform=transform_test, align_faces=align_faces)
    publoader = torch.utils.data.DataLoader(pubset, batch_size=batch_size)
    
    priset = FER2013(split='PrivateTest', transform=transform_test, align_faces=align_faces)
    priloader = torch.utils.data.DataLoader(priset, batch_size=batch_size)
    
    # Compute the metrics first, then generate the summary tables and plots from the same outputs.
    print("\n" + "="*70)
    print("METRIC COMPUTATION")
    print("="*70)
    
    print("\n[1/2] Computing metrics for Public Test Set...")
    pub_rmse_dict, pub_overall_rmse, pub_predictions, pub_targets = compute_rmse(net, publoader, use_cuda=use_cuda, max_batches=max_batches)
    pub_ccc_dict, pub_overall_ccc = compute_ccc(pub_predictions, pub_targets)
    
    print("[2/2] Computing metrics for Private Test Set...")
    pri_rmse_dict, pri_overall_rmse, pri_predictions, pri_targets = compute_rmse(net, priloader, use_cuda=use_cuda, max_batches=max_batches)
    pri_ccc_dict, pri_overall_ccc = compute_ccc(pri_predictions, pri_targets)
    
    # === Print RMSE Results ===
    print("\n" + "="*70)
    print("RMSE (Root Mean Squared Error) Results")
    print("="*70)
    
    pub_rmse = np.array([pub_rmse_dict['Valence'], pub_rmse_dict['Arousal'], pub_rmse_dict['Dominance']])
    print("\nPublic Test Set:")
    print(f"  Valence    : {pub_rmse[0]:.4f}")
    print(f"  Arousal    : {pub_rmse[1]:.4f}")
    print(f"  Dominance  : {pub_rmse[2]:.4f}")
    print(f"  Overall    : {pub_overall_rmse:.4f}")
    
    pri_rmse = np.array([pri_rmse_dict['Valence'], pri_rmse_dict['Arousal'], pri_rmse_dict['Dominance']])
    print("\nPrivate Test Set:")
    print(f"  Valence    : {pri_rmse[0]:.4f}")
    print(f"  Arousal    : {pri_rmse[1]:.4f}")
    print(f"  Dominance  : {pri_rmse[2]:.4f}")
    print(f"  Overall    : {pri_overall_rmse:.4f}")
    
    # === Print CCC Results ===
    print("\n" + "="*70)
    print("CCC (Concordance Correlation Coefficient) Results")
    print("="*70)
    print("(Range: -1 to 1, where 1 is perfect agreement)")
    
    pub_ccc = np.array([pub_ccc_dict['Valence'], pub_ccc_dict['Arousal'], pub_ccc_dict['Dominance']])
    print("\nPublic Test Set:")
    print(f"  Valence    : {pub_ccc[0]:.4f}")
    print(f"  Arousal    : {pub_ccc[1]:.4f}")
    print(f"  Dominance  : {pub_ccc[2]:.4f}")
    print(f"  Overall    : {pub_overall_ccc:.4f}")
    
    pri_ccc = np.array([pri_ccc_dict['Valence'], pri_ccc_dict['Arousal'], pri_ccc_dict['Dominance']])
    print("\nPrivate Test Set:")
    print(f"  Valence    : {pri_ccc[0]:.4f}")
    print(f"  Arousal    : {pri_ccc[1]:.4f}")
    print(f"  Dominance  : {pri_ccc[2]:.4f}")
    print(f"  Overall    : {pri_overall_ccc:.4f}")
    
    # === Comparison with Paper ===
    print("\n" + "="*70)
    print("PAPER vs YOUR RESULTS (without orthogonal regularization)")
    print("="*70)
    
    paper_results = {
        'Valence': {'public': 0.076, 'private': 0.063},
        'Arousal': {'public': 0.048, 'private': 0.094},
        'Dominance': {'public': 0.078, 'private': 0.069},
    }
    
    dim_names = ['Valence', 'Arousal', 'Dominance']
    
    print("\n{:<12} | {:>8} {:>8} | {:>8} {:>8}".format("Dimension", "Your Pub", "Paper", "Your Pri", "Paper"))
    print("-" * 70)
    
    for i, dim in enumerate(dim_names):
        your_pub = pub_rmse[i]
        your_pri = pri_rmse[i]
        paper_pub = paper_results[dim]['public']
        paper_pri = paper_results[dim]['private']
        
        print(f"{dim:<12} | {your_pub:>8.4f} {paper_pub:>8.4f} | {your_pri:>8.4f} {paper_pri:>8.4f}")
    
    print("-" * 70)
    print(f"{'Overall':<12} | {pub_overall_rmse:>8.4f}          | {pri_overall_rmse:>8.4f}")
    
    print("\n✅ SUCCESS!" if pub_overall_rmse < 0.15 else "\n[Note] Gap still exists")
    
    # === Plot Confusion Matrices ===
    print("\n" + "="*70)
    print("GENERATING CONFUSION MATRICES")
    print("="*70)
    
    print("\nPlotting confusion matrices for Public Test Set...")
    plot_confusion_matrices(pub_predictions, pub_targets, output_dir=os.path.join(output_dir, 'public_test'))
    
    print("Plotting confusion matrices for Private Test Set...")
    plot_confusion_matrices(pri_predictions, pri_targets, output_dir=os.path.join(output_dir, 'private_test'))
    
    # === Save Summary Report ===
    print("\n" + "="*70)
    print("SAVING SUMMARY REPORT")
    print("="*70)
    
    report_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VAD-NET MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"CUDA: {use_cuda}\n")
        f.write(f"Cut Size: {cut_size}, Input Size: {input_size}\n\n")
        
        f.write("="*70 + "\n")
        f.write("RMSE RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"\nPublic Test Set:\n")
        f.write(f"  Valence    : {pub_rmse[0]:.4f}\n")
        f.write(f"  Arousal    : {pub_rmse[1]:.4f}\n")
        f.write(f"  Dominance  : {pub_rmse[2]:.4f}\n")
        f.write(f"  Overall    : {pub_overall_rmse:.4f}\n")
        
        f.write(f"\nPrivate Test Set:\n")
        f.write(f"  Valence    : {pri_rmse[0]:.4f}\n")
        f.write(f"  Arousal    : {pri_rmse[1]:.4f}\n")
        f.write(f"  Dominance  : {pri_rmse[2]:.4f}\n")
        f.write(f"  Overall    : {pri_overall_rmse:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CCC RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"\nPublic Test Set:\n")
        f.write(f"  Valence    : {pub_ccc[0]:.4f}\n")
        f.write(f"  Arousal    : {pub_ccc[1]:.4f}\n")
        f.write(f"  Dominance  : {pub_ccc[2]:.4f}\n")
        f.write(f"  Overall    : {pub_overall_ccc:.4f}\n")
        
        f.write(f"\nPrivate Test Set:\n")
        f.write(f"  Valence    : {pri_ccc[0]:.4f}\n")
        f.write(f"  Arousal    : {pri_ccc[1]:.4f}\n")
        f.write(f"  Dominance  : {pri_ccc[2]:.4f}\n")
        f.write(f"  Overall    : {pri_overall_ccc:.4f}\n")
    
    print(f"✅ Summary report saved to {report_path}")
    print("="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate VAD-Net model (RMSE, CCC, Confusion Matrix)')
    parser.add_argument('--model', type=str, default='FER2013_ResNet/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--cut_size', type=int, default=48, help='Spatial crop size used at evaluation')
    parser.add_argument('--input_size', type=int, default=0, help='Final model input size after crop (0 uses cut_size)')
    parser.add_argument('--align_faces', action='store_true', help='Enable OpenCV Haar-based face alignment before transforms')
    parser.add_argument('--public_csv', type=str, default='', help='Optional override CSV for PublicTest split')
    parser.add_argument('--private_csv', type=str, default='', help='Optional override CSV for PrivateTest split')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results (RMSE, CCC, confusion matrices)')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit the number of batches per split for smoke testing')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used for evaluation')
    args = parser.parse_args()

    FER2013.set_data_protocol('small_split')
    FER2013.set_split_files(
        public_file=args.public_csv or None,
        private_file=args.private_csv or None,
    )
    print('[Small Split] Using current extracted CSVs in ./data')

    use_cuda = args.cuda and torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    eval_input_size = args.input_size if args.input_size > 0 else args.cut_size
    evaluate_all_sets(args.model, use_cuda=use_cuda, cut_size=args.cut_size,
                      input_size=eval_input_size, align_faces=args.align_faces,
                      max_batches=args.max_batches,
                      batch_size=args.batch_size,
                      output_dir=args.output_dir)


if __name__ == '__main__':
    import argparse
    main()
