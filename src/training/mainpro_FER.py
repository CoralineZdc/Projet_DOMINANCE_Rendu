"""Main training script for VAD regression models."""

from __future__ import print_function

import argparse
import csv
import os
import sys
import random
from pathlib import Path

# Add parent directories to Python path to allow imports from root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Root directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))  # src/utils

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from PIL import Image

import transforms as transforms
import training_utils
import orthogonal_regularization
from fer import FER2013
from models.efficientnet_b0 import EfficientNetVAD
from models.mobilefacenet import MobileFaceNetVAD
from models.resnet_reg2 import (
    ResNet18RegressionThreeOutputs,
    ResNet50PretrainedRegressionThreeOutputs,
    ResNet50RegressionThreeOutputs,
)


EVAL_INPUT_SIZE = 48


def set_seed(seed):
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_transform(crops):
    """Convert TenCrop outputs into a normalized mini-batch tensor."""
    normalize = transforms.Normalize(
        mean=FER2013.image_mean.tolist(),
        std=FER2013.image_std.tolist(),
    )
    processed = []
    for crop in crops:
        if crop.size != (EVAL_INPUT_SIZE, EVAL_INPUT_SIZE):
            crop = crop.resize((EVAL_INPUT_SIZE, EVAL_INPUT_SIZE), Image.BILINEAR)
        processed.append(normalize(transforms.ToTensor()(crop)))
    return torch.stack(processed)


def save_checkpoint(state, filename):
    """Save a training checkpoint to disk."""
    torch.save(state, filename)


def mixup_batch(inputs, targets, alpha):
    """Apply mixup augmentation to a batch when alpha is positive."""
    if alpha <= 0.0:
        return inputs, targets

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    mixed_targets = lam * targets + (1.0 - lam) * targets[index]
    return mixed_inputs, mixed_targets


def ccc_loss(outputs, targets, loss_weights, eps=1e-8):
    """Compute the CCC loss term used to complement MSE supervision."""
    mean_out = outputs.mean(dim=0)
    mean_tgt = targets.mean(dim=0)
    var_out = outputs.var(dim=0, unbiased=False)
    var_tgt = targets.var(dim=0, unbiased=False)
    cov = ((outputs - mean_out) * (targets - mean_tgt)).mean(dim=0)

    ccc = (2.0 * cov) / (var_out + var_tgt + (mean_out - mean_tgt) ** 2 + eps)
    ccc_per_dim_loss = 1.0 - ccc
    return (ccc_per_dim_loss * loss_weights).sum()


def _is_head_param(name):
    """Return True when a parameter name belongs to the prediction head."""
    head_tokens = ("head", "heads", "fc", "classifier", "regressor", "output")
    lname = name.lower()
    return any(token in lname for token in head_tokens)


def build_optimizer_param_groups(net, opt):
    """Build parameter groups with separate learning-rate and decay settings."""
    backbone_wd = opt.weight_decay if opt.weight_decay_backbone < 0 else opt.weight_decay_backbone
    head_wd = opt.weight_decay if opt.weight_decay_head < 0 else opt.weight_decay_head

    grouped = {}
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        is_head = _is_head_param(name)
        wd = head_wd if is_head else backbone_wd
        lr_mult = opt.lr_head_mult if is_head else opt.lr_backbone_mult
        if opt.no_wd_norm_bias and (param.ndim == 1 or name.endswith(".bias")):
            wd = 0.0

        grouped.setdefault((float(wd), float(lr_mult)), []).append(param)

    param_groups = []
    summary_items = []
    for (wd, lr_mult), params in grouped.items():
        param_groups.append({"params": params, "weight_decay": wd, "lr": opt.lr * lr_mult})
        summary_items.append("wd={} lr_mult={}: {} tensors".format(wd, lr_mult, len(params)))
    summary = ", ".join(summary_items)
    print("Optimizer param groups -> {}".format(summary))
    return param_groups


def resnet_orthogonal_regularization(net):
    """Compute an orthogonality penalty for ResNet-style skip connections."""
    device = next(net.parameters()).device
    if not all(hasattr(net, name) for name in ("layer1", "layer2", "layer3", "layer4")):
        return torch.zeros((), device=device)

    diff = torch.zeros((), device=device)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer2[0].shortcut[0].weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer3[0].shortcut[0].weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer4[0].shortcut[0].weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer1[0].conv1.weight, stride=1)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer1[1].conv1.weight, stride=1)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer2[0].conv1.weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer2[1].conv1.weight, stride=1)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer3[0].conv1.weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer3[1].conv1.weight, stride=1)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer4[0].conv1.weight, stride=2)
    diff = diff + orthogonal_regularization.deconv_orth_dist(net.layer4[1].conv1.weight, stride=1)
    return diff


def train(epoch, trainloader, net, optimizer, criterion, use_cuda, loss_weights, opt):
    """Run one training epoch and return the average loss."""
    net.train()
    total_loss = 0.0
    total_batches = 0

    print("\nEpoch: {}".format(epoch))
    print("LR: {}".format(optimizer.param_groups[0]["lr"]))

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = mixup_batch(inputs, targets, opt.mixup_alpha)

        optimizer.zero_grad()
        outputs = net(inputs)

        mse_per_dim = criterion(outputs, targets).mean(dim=0)
        loss = (mse_per_dim * loss_weights).sum()

        if opt.ccc_weight > 0.0:
            loss = loss + opt.ccc_weight * ccc_loss(outputs, targets, loss_weights)

        if opt.ortho > 0.0:
            diff = resnet_orthogonal_regularization(net)
            if diff is not None:
                loss = loss + opt.ortho * diff

        loss.backward()
        if opt.grad_clip > 0.0:
            training_utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    print("Train Loss: {:.4f}".format(avg_loss))
    return avg_loss


def evaluate(dataloader, net, criterion, use_cuda, loss_weights, opt):
    """Evaluate a model and return the average loss and per-dimension RMSE."""
    net.eval()
    total_loss = 0.0
    total_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

            mse_per_dim = criterion(outputs_avg, targets).mean(dim=0)
            loss = (mse_per_dim * loss_weights).sum()
            if opt.ccc_weight > 0.0:
                loss = loss + opt.ccc_weight * ccc_loss(outputs_avg, targets, loss_weights)

            total_loss += loss.item()
            total_batches += 1

            all_predictions.append(outputs_avg.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / max(total_batches, 1)

    all_pred = torch.cat(all_predictions, dim=0).numpy()
    all_targ = torch.cat(all_targets, dim=0).numpy()

    label_mean = FER2013.label_mean.numpy()
    label_std = FER2013.label_std.numpy()
    all_pred = all_pred * label_std + label_mean
    all_targ = all_targ * label_std + label_mean

    rmse_per_dim = np.sqrt(np.mean((all_pred - all_targ) ** 2, axis=0))
    return avg_loss, rmse_per_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--lr", default=0.0002, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--ortho", default=0.0, type=float)
    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--weight_decay_backbone", default=-1.0, type=float, help="Override backbone weight decay; -1 uses --weight_decay")
    parser.add_argument("--weight_decay_head", default=-1.0, type=float, help="Override head/classifier weight decay; -1 uses --weight_decay")
    parser.add_argument("--lr_backbone_mult", default=0.2, type=float, help="Backbone LR multiplier for fine-tuning")
    parser.add_argument("--lr_head_mult", default=1.0, type=float, help="Head LR multiplier")
    parser.add_argument("--no_wd_norm_bias", action="store_true", help="Set weight decay to 0 for norm/bias parameters")
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--cut_size", default=48, type=int)
    parser.add_argument("--input_size", default=0, type=int)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--sampler_weight", default=0.0, type=float)
    parser.add_argument("--mixup_alpha", default=0.0, type=float)
    parser.add_argument("--aug_profile", default="light", choices=["none", "light", "medium", "strong"])
    parser.add_argument("--train_crop_padding", default=2, type=int)
    parser.add_argument("--rotation_deg", default=5.0, type=float)
    parser.add_argument("--align_faces", action="store_true")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "efficientnet", "mobilefacenet"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--separate_heads", action="store_true")
    parser.add_argument("--loss_weights", default="1.0,1.0,1.0", type=str)
    parser.add_argument("--ccc_weight", default=0.1, type=float)
    parser.add_argument("--lr_patience", default=8, type=int)
    parser.add_argument("--lr_factor", default=0.5, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--lr_cooldown", default=2, type=int)
    parser.add_argument("--lr_threshold", default=0.002, type=float)
    parser.add_argument("--lr_threshold_mode", default="abs", choices=["rel", "abs"])
    parser.add_argument("--lr_ema_beta", default=0.6, type=float, help="EMA smoothing for monitored RMSE. Set 0 to disable.")
    parser.add_argument("--early_stop_patience", default=40, type=int)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="FER2013_ResNet", type=str)
    parser.add_argument("--train_csv", default="", type=str)
    parser.add_argument("--public_csv", default="", type=str)
    parser.add_argument("--private_csv", default="", type=str)
    parser.add_argument("--cuda", action="store_true", help="Compatibility flag: prefer CUDA if available")
    opt = parser.parse_args()

    set_seed(opt.seed)

    FER2013.set_data_protocol("small_split")
    FER2013.set_split_files(
        train_file=opt.train_csv or None,
        public_file=opt.public_csv or None,
        private_file=opt.private_csv or None,
    )
    print("[Small Split] Using current extracted CSVs in ./data")

    use_cuda = torch.cuda.is_available()
    if opt.cuda and not use_cuda:
        print("[Warning] --cuda was requested but CUDA is not available. Falling back to CPU.")

    parsed_loss_weights = list(map(float, opt.loss_weights.split(",")))
    if len(parsed_loss_weights) != 3:
        raise ValueError("loss_weights must be three comma-separated floats")
    loss_weights = torch.tensor(parsed_loss_weights, dtype=torch.float32)
    if use_cuda:
        loss_weights = loss_weights.cuda()

    total_epoch = opt.epochs
    early_stop_patience = opt.early_stop_patience
    best_score = float("inf")
    early_stop_counter = 0

    path = opt.output_dir
    os.makedirs(path, exist_ok=True)

    FER2013._ensure_label_stats()
    FER2013._ensure_image_stats()

    input_size = opt.input_size
    if input_size <= 0:
        input_size = 224 if opt.pretrained else opt.cut_size

    EVAL_INPUT_SIZE = input_size

    transform_list = []
    if opt.aug_profile in {"light", "medium", "strong"}:
        transform_list.extend(
            [
                transforms.RandomCrop(opt.cut_size, padding=max(0, opt.train_crop_padding)),
                transforms.RandomHorizontalFlip(),
            ]
        )

    if opt.aug_profile in {"medium", "strong"}:
        transform_list.append(transforms.RandomRotation(opt.rotation_deg))

    if opt.aug_profile == "strong":
        transform_list.append(transforms.RandomRotation(max(opt.rotation_deg, 8.0)))

    if hasattr(transforms, "RandomAffine"):
        transform_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))

    if hasattr(transforms, "RandomErasing"):
        transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)))

    transform_list.extend(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=FER2013.image_mean.tolist(),
                std=FER2013.image_std.tolist(),
            ),
        ]
    )

    transform_train = transforms.Compose(transform_list)

    print(
        "Augment: profile={}, pad={}, rot_deg={}".format(
            opt.aug_profile,
            max(0, opt.train_crop_padding),
            opt.rotation_deg,
        )
    )
    if opt.mixup_alpha > 0.0:
        print("Mixup enabled: alpha={}".format(opt.mixup_alpha))
    if opt.sampler_weight > 0.0:
        print("Weighted sampler enabled: weight={}".format(opt.sampler_weight))

    transform_test = transforms.Compose([
        transforms.TenCrop(opt.cut_size),
        custom_transform,
    ])

    trainset = FER2013(split="Training", transform=transform_train, align_faces=opt.align_faces)

    if opt.sampler_weight > 0.0:
        sample_scores = trainset.labels.abs().mean(dim=1)
        sample_weights = 1.0 + opt.sampler_weight * sample_scores
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True)

    pubset = FER2013(split="PublicTest", transform=transform_test, align_faces=opt.align_faces)
    publoader = torch.utils.data.DataLoader(pubset, batch_size=opt.bs)

    if opt.model == "resnet50" and opt.pretrained:
        net = ResNet50PretrainedRegressionThreeOutputs(
            dropout_rate=opt.dropout,
            separate_heads=opt.separate_heads,
            freeze_backbone=opt.freeze_backbone,
        )
    elif opt.model == "resnet50":
        net = ResNet50RegressionThreeOutputs(dropout_rate=opt.dropout, separate_heads=opt.separate_heads)
    elif opt.model == "resnet18":
        net = ResNet18RegressionThreeOutputs(dropout_rate=opt.dropout, separate_heads=opt.separate_heads)
    elif opt.model == "efficientnet":
        net = EfficientNetVAD(dropout_rate=opt.dropout)
    else:
        net = MobileFaceNetVAD(dropout_rate=opt.dropout)

    print(
        "Model: {} (pretrained={}, separate_heads={}, freeze_backbone={})".format(
            opt.model.upper(), opt.pretrained, opt.separate_heads, opt.freeze_backbone
        )
    )
    print(net)
    print("params:", sum(p.numel() for p in net.parameters()))
    print("trainable:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    if use_cuda:
        net.cuda()

    criterion = nn.MSELoss(reduction="none")

    optimizer_param_groups = build_optimizer_param_groups(net, opt)
    if opt.optimizer == "adamw":
        optimizer = optim.AdamW(optimizer_param_groups, lr=opt.lr)
    else:
        optimizer = optim.SGD(optimizer_param_groups, lr=opt.lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=opt.lr_factor,
        patience=opt.lr_patience,
        threshold=opt.lr_threshold,
        threshold_mode=opt.lr_threshold_mode,
        cooldown=opt.lr_cooldown,
        min_lr=opt.min_lr,
    )

    prev_lr = optimizer.param_groups[0]["lr"]
    checkpoint_path = os.path.join(path, "checkpoint.pth")
    start_epoch = 0

    if opt.resume and not opt.no_checkpoint and os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            print("Warning: checkpoint optimizer state is incompatible; using fresh optimizer state")
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception:
            print("Warning: checkpoint scheduler state is incompatible; using fresh scheduler state")
        start_epoch = checkpoint["epoch"] + 1
        best_score = float(checkpoint.get("best_score", float("inf")))
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        print("Resume state: best_score={:.6f}, early_stop_counter={}".format(best_score, early_stop_counter))
    else:
        if opt.no_checkpoint:
            print("Checkpoint disabled; starting from scratch")
        elif os.path.exists(checkpoint_path):
            print("Checkpoint found but resume is disabled; starting from scratch")
        else:
            print("No checkpoint found, starting from scratch")

    log_file = os.path.join(path, "log.csv")
    write_header = True
    if opt.resume and os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        write_header = False

    with open(log_file, "a" if not write_header else "w", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch",
                "train_loss",
                "public_loss",
                "pub_rmse_val",
                "pub_rmse_aro",
                "pub_rmse_dom",
                "pub_rmse_mean",
                "lr",
            ])

    rmse_ema = None

    for epoch in range(start_epoch, total_epoch):
        train_loss = train(epoch, trainloader, net, optimizer, criterion, use_cuda, loss_weights, opt)
        pub_loss, pub_rmse = evaluate(publoader, net, criterion, use_cuda, loss_weights, opt)
        pub_rmse_mean = float(np.mean(pub_rmse))

        print(
            "Public Loss: {:.4f}, RMSE: Val={:.4f} Aro={:.4f} Dom={:.4f}".format(
                pub_loss, pub_rmse[0], pub_rmse[1], pub_rmse[2]
            )
        )
        print("Public Mean RMSE: {:.4f}".format(pub_rmse_mean))

        if opt.lr_ema_beta > 0.0:
            beta = min(max(opt.lr_ema_beta, 0.0), 0.99)
            rmse_ema = pub_rmse_mean if rmse_ema is None else (beta * rmse_ema + (1.0 - beta) * pub_rmse_mean)
            metric_for_scheduler = rmse_ema
            print("Scheduler metric (EMA): {:.4f}".format(metric_for_scheduler))
        else:
            metric_for_scheduler = pub_rmse_mean

        scheduler.step(metric_for_scheduler)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            print("LR reduced: {} -> {}".format(prev_lr, current_lr))
        prev_lr = current_lr

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                pub_loss,
                pub_rmse[0],
                pub_rmse[1],
                pub_rmse[2],
                pub_rmse_mean,
                current_lr,
            ])

        is_best = pub_rmse_mean < best_score
        if is_best:
            best_score = pub_rmse_mean
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if not opt.no_checkpoint:
            checkpoint = {
                "epoch": epoch,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "early_stop_counter": early_stop_counter,
            }
            save_checkpoint(checkpoint, os.path.join(path, "checkpoint.pth"))

            if is_best:
                print("Saving best model...")
                save_checkpoint(checkpoint, os.path.join(path, "best_model.pth"))
                torch.save(net.state_dict(), os.path.join(path, "best_model_state.pth"))

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    torch.save(net.state_dict(), os.path.join(path, "last_model.pth"))

