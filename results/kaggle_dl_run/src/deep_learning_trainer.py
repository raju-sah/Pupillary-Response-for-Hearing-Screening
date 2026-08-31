"""
Deep Learning Trainer & Subject-Independent Cross-Validation Module.

Provides:
1. Leakage-free Stratified Group 5-Fold Cross-Validation for PyTorch models.
2. Channel-wise standardization fitted strictly on training folds.
3. AdamW optimization, CosineAnnealingLR scheduling, and validation ROC-AUC early stopping.
4. Out-of-fold probability accumulation and 500-fold bootstrap 95% CIs.
5. Extraction and aggregation of temporal attention saliency weights.
"""

from typing import Dict, Any, List, Optional, Tuple, Type, Sequence
from dataclasses import dataclass
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.deep_learning_models import (
    PupilTimeSeriesDataset,
    FocalLoss,
    MultiScaleConv1DNet,
    BiLSTMAttentionNet,
    DilatedTCNNet,
    CNNTransformerNet,
)
from src.classical_models import (
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
)


@dataclass
class DeepLearningEvaluationResult:
    """Out-of-fold evaluation summary for a trained Deep Learning architecture."""
    model_name: str
    n_samples: int
    n_subjects: int
    n_positive: int
    n_negative: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1_macro: float
    f1_positive: float
    brier_score: float
    optimal_threshold: float
    fold_roc_aucs: List[float]
    fold_pr_aucs: List[float]
    fold_balanced_accs: List[float]
    ci_95: Dict[str, Tuple[float, float]]
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    y_pred_bin: np.ndarray
    subject_ids: np.ndarray
    attention_weights: Optional[np.ndarray] = None  # (N, T) for BiLSTMAttentionNet


def standardize_channels(
    X_train: np.ndarray,
    X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes multi-channel tensors channel-by-channel:
      X_norm = (X - mean) / (std + 1e-6)
    Statistics are computed exclusively on X_train.
    """
    # X shape: (N, C, T)
    means = np.mean(X_train, axis=(0, 2), keepdims=True)  # (1, C, 1)
    stds = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6  # (1, C, 1)

    X_train_norm = (X_train - means) / stds
    X_val_norm = (X_val - means) / stds

    return X_train_norm, X_val_norm, means, stds


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Runs a single training epoch and returns mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(y_batch)
        total_samples += len(y_batch)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    extract_attention: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluates model on dataloader and returns (y_true, y_pred_proba, attention_weights).
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_attns = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        if extract_attention and hasattr(model, "forward") and isinstance(model, BiLSTMAttentionNet):
            logits, attn = model(x_batch, return_attention=True)
            all_attns.append(attn.cpu().numpy())
        else:
            logits = model(x_batch)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.append(probs)
        all_targets.append(y_batch.numpy().flatten())

    y_pred_proba = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    attns = np.concatenate(all_attns, axis=0) if all_attns else None

    return y_true, y_pred_proba, attns


def evaluate_dl_model_stratified_group_cv(
    model_name: str,
    model_cls: Type[nn.Module],
    model_kwargs: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    max_epochs: int = 35,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_type: str = "focal",
    device: Optional[torch.device] = None,
    seed: int = 42,
    n_bootstraps: int = 500
) -> DeepLearningEvaluationResult:
    """
    Executes Leakage-Free Stratified Group 5-Fold Cross-Validation for a PyTorch Deep Learning Model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_pred_proba = np.zeros(len(y), dtype=float)
    oof_attention: Optional[np.ndarray] = np.zeros((len(y), X.shape[-1]), dtype=float) if model_cls == BiLSTMAttentionNet else None

    fold_roc_aucs = []
    fold_pr_aucs = []
    fold_balanced_accs = []

    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    pos_weight_val = float(neg_count / max(pos_count, 1))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        # Zero subject leakage check
        train_subjs = set(groups[train_idx])
        val_subjs = set(groups[val_idx])
        assert len(train_subjs.intersection(val_subjs)) == 0, f"Subject leakage in fold {fold_idx}!"

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Channel Standardization fit on X_train only
        X_train_norm, X_val_norm, _, _ = standardize_channels(X_train, X_val)

        train_ds = PupilTimeSeriesDataset(X_train_norm, y_train)
        val_ds = PupilTimeSeriesDataset(X_val_norm, y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Initialize model
        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        if loss_type == "focal":
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

        best_val_auc = -1.0
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = train_single_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            y_val_t, y_val_p, _ = evaluate_dataloader(model, val_loader, device, extract_attention=False)
            val_metrics = compute_binary_metrics(y_val_t, y_val_p)
            val_auc = val_metrics["roc_auc"]

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        # Load best model checkpoint for evaluation
        model.load_state_dict(best_state)
        is_bilstm = (model_cls == BiLSTMAttentionNet)
        _, final_val_probs, final_attns = evaluate_dataloader(model, val_loader, device, extract_attention=is_bilstm)

        oof_pred_proba[val_idx] = final_val_probs
        if is_bilstm and final_attns is not None and oof_attention is not None:
            oof_attention[val_idx] = final_attns

        fold_res = compute_binary_metrics(y_val, final_val_probs)
        fold_roc_aucs.append(fold_res["roc_auc"])
        fold_pr_aucs.append(fold_res["pr_auc"])
        fold_balanced_accs.append(fold_res["balanced_accuracy"])

    # Out-Of-Fold pooled performance
    pooled_metrics = compute_binary_metrics(y, oof_pred_proba)
    ci_95 = compute_bootstrap_confidence_intervals(y, oof_pred_proba, n_bootstraps=n_bootstraps, seed=seed)

    return DeepLearningEvaluationResult(
        model_name=model_name,
        n_samples=len(y),
        n_subjects=len(np.unique(groups)),
        n_positive=int(pos_count),
        n_negative=int(neg_count),
        prevalence=float(pos_count / max(len(y), 1)),
        roc_auc=pooled_metrics["roc_auc"],
        pr_auc=pooled_metrics["pr_auc"],
        balanced_accuracy=pooled_metrics["balanced_accuracy"],
        accuracy=pooled_metrics["accuracy"],
        sensitivity=pooled_metrics["sensitivity"],
        specificity=pooled_metrics["specificity"],
        precision=pooled_metrics["precision"],
        npv=pooled_metrics["npv"],
        f1_macro=pooled_metrics["f1_macro"],
        f1_positive=pooled_metrics["f1_positive"],
        brier_score=pooled_metrics["brier_score"],
        optimal_threshold=pooled_metrics["optimal_threshold"],
        fold_roc_aucs=fold_roc_aucs,
        fold_pr_aucs=fold_pr_aucs,
        fold_balanced_accs=fold_balanced_accs,
        ci_95=ci_95,
        y_true=y,
        y_pred_proba=oof_pred_proba,
        y_pred_bin=pooled_metrics["y_pred_bin"],
        subject_ids=groups,
        attention_weights=oof_attention
    )
