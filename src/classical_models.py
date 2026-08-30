"""
Classical Machine Learning Baselines & Subject-Independent Cross-Validation Module.

Provides:
1. Leakage-free Stratified Group K-Fold Cross-Validation (StratifiedGroupKFold grouped by subject_id).
2. Model suite:
   - Dummy Baselines (Stratified, Prior)
   - Single-Feature Physiological Heuristic (Threshold on peak dilation)
   - Regularized Logistic Regression (L2, ElasticNet)
   - Support Vector Classifier (Linear, RBF with Platt calibration)
   - Ensemble Trees: Random Forest, HistGradientBoosting, XGBoost
3. Comprehensive metric computation: ROC-AUC, PR-AUC (Average Precision),
   Balanced Accuracy, Sensitivity/Specificity (at default and Youden-optimal thresholds),
   PPV, NPV, Macro F1, Brier Score.
4. Bootstrap 95% Confidence Intervals (1,000 iterations).
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
from dataclasses import dataclass, asdict
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class ModelEvaluationResult:
    """Out-of-fold evaluation summary and metrics for a trained classifier."""
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
    sensitivity: float  # Recall at optimal threshold
    specificity: float  # TNR at optimal threshold
    precision: float    # PPV at optimal threshold
    npv: float          # NPV at optimal threshold
    f1_macro: float
    f1_positive: float
    brier_score: float
    optimal_threshold: float
    fold_roc_aucs: List[float]
    fold_pr_aucs: List[float]
    fold_balanced_accs: List[float]
    ci_95: Dict[str, Tuple[float, float]]  # Metric -> (lower_ci, upper_ci)
    y_true: np.ndarray
    y_pred_proba: np.ndarray
    y_pred_bin: np.ndarray
    subject_ids: np.ndarray


class SingleFeatureHeuristicClassifier(BaseEstimator, ClassifierMixin):
    """
    Physiological benchmark heuristic: applies a threshold on a single feature column
    (e.g., peak dilation amplitude).
    Fits the optimal threshold using Youden's J statistic on the training set,
    and fits a logistic sigmoid for calibrated probabilistic predictions.
    """
    def __init__(self, feature_idx: int = 2):
        self.feature_idx = feature_idx
        self.classes_ = np.array([0, 1])
        self.threshold_ = 0.0
        self.scale_ = 1.0
        self.intercept_ = 0.0

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        X_arr = np.asarray(X)
        feat_vals = X_arr[:, self.feature_idx]
        y_arr = np.asarray(y, dtype=int)

        # Handle any non-finite entries
        valid = np.isfinite(feat_vals) & np.isfinite(y_arr)
        if np.sum(valid) < 5:
            self.threshold_ = 0.0
            return self

        f_clean = feat_vals[valid]
        y_clean = y_arr[valid]

        if len(np.unique(y_clean)) < 2:
            self.threshold_ = float(np.median(f_clean))
            return self

        # Scan quantiles to find optimal Youden threshold on training data
        candidate_threshs = np.percentile(f_clean, np.linspace(5, 95, 91))
        best_j = -1.0
        best_th = float(np.median(f_clean))

        for th in candidate_threshs:
            pred = (f_clean >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_clean, pred, labels=[0, 1]).ravel()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j = sens + spec - 1.0
            if j > best_j:
                best_j = j
                best_th = float(th)

        self.threshold_ = best_th

        # Fit simple 1D logistic scaling for predict_proba
        lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr.fit(f_clean.reshape(-1, 1), y_clean)
        self.scale_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X)
        feat_vals = X_arr[:, self.feature_idx]
        # Logistic sigmoid probability
        z = self.scale_ * feat_vals + self.intercept_
        z = np.clip(z, -30.0, 30.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X)
        feat_vals = X_arr[:, self.feature_idx]
        return (feat_vals >= self.threshold_).astype(int)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes all standard binary discrimination and screening metrics.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for positive class (class 1).
        threshold: Decision threshold for binarization. If None, optimal Youden threshold is computed.

    Returns:
        Dictionary of metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    # Edge cases
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": float(np.mean(y_true)),
            "balanced_accuracy": 0.5,
            "accuracy": 0.5,
            "sensitivity": 0.0,
            "specificity": 1.0,
            "precision": 0.0,
            "npv": 1.0,
            "f1_macro": 0.0,
            "f1_positive": 0.0,
            "brier_score": 0.25,
            "optimal_threshold": 0.5,
            "y_pred_bin": (y_pred_proba >= 0.5).astype(int)
        }

    # Discrimination Metrics
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    # ROC curve & optimal threshold (Youden's J = TPR - FPR)
    fpr, tpr, roc_threshs = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    best_idx = int(np.argmax(youden_index))
    opt_thresh = float(roc_threshs[best_idx])
    # Bound threshold sensibly
    if not (0.01 <= opt_thresh <= 0.99):
        opt_thresh = 0.50

    op_threshold = threshold if threshold is not None else opt_thresh
    y_pred_bin = (y_pred_proba >= op_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    prec = float(tp / max(tp + fp, 1))
    npv = float(tn / max(tn + fn, 1))
    acc = float(accuracy_score(y_true, y_pred_bin))
    b_acc = float(balanced_accuracy_score(y_true, y_pred_bin))
    f1_pos = float(f1_score(y_true, y_pred_bin, pos_label=1, zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred_bin, average="macro", zero_division=0))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": b_acc,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1_macro": f1_mac,
        "f1_positive": f1_pos,
        "brier_score": brier,
        "optimal_threshold": op_threshold,
        "y_pred_bin": y_pred_bin
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% percentile bootstrap confidence intervals on out-of-fold predictions.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tracked_metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1_macro", "brier_score"]
    boot_vals = {k: [] for k in tracked_metrics}

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = y_pred_proba[idx]

        if len(np.unique(y_b)) < 2:
            continue

        res = compute_binary_metrics(y_b, p_b)
        for k in tracked_metrics:
            boot_vals[k].append(res[k])

    ci_dict = {}
    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    for k in tracked_metrics:
        arr = np.array(boot_vals[k])
        if len(arr) > 10:
            low = float(np.percentile(arr, lower_p))
            high = float(np.percentile(arr, upper_p))
        else:
            low, high = np.nan, np.nan
        ci_dict[k] = (low, high)

    return ci_dict


def get_classical_model_suite(
    random_state: int = 42,
    pos_weight: float = 7.0
) -> Dict[str, Any]:
    """
    Constructs the standard classical ML baseline model suite.
    """
    models = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=random_state),
        "Dummy (Prior/Majority)": DummyClassifier(strategy="prior"),
        "Single Feature Heuristic (Peak Dilation)": SingleFeatureHeuristicClassifier(feature_idx=2),
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=random_state))
        ]),
        "Logistic Regression (ElasticNet)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=1.0, class_weight="balanced", solver="saga", max_iter=1000, random_state=random_state))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=random_state, dual="auto", max_iter=2000),
                cv=3
            ))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=random_state))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            max_depth=6,
            random_state=random_state
        )
    }

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

    return models


def evaluate_model_stratified_group_cv(
    model_name: str,
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    n_bootstraps: int = 1000
) -> ModelEvaluationResult:
    """
    Executes Leakage-Free Stratified Group 5-Fold Cross-Validation.

    Guarantees:
    1. Zero subject overlap between training and testing folds (train_subjs ∩ test_subjs == ∅).
    2. Scalers/Pipelines are fitted exclusively on the training fold X_train.
    3. Aggregates true out-of-fold probabilities and computes 1,000 bootstrap CIs.
    """
    X_mat = np.asarray(X)
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred_proba = np.zeros(len(y_arr), dtype=float)
    fold_roc_aucs = []
    fold_pr_aucs = []
    fold_balanced_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_mat, y_arr, groups=groups_arr)):
        # Verify zero subject leakage
        train_subjs = set(groups_arr[train_idx])
        test_subjs = set(groups_arr[test_idx])
        assert len(train_subjs.intersection(test_subjs)) == 0, f"Subject leakage detected in fold {fold_idx}!"

        X_train, y_train = X_mat[train_idx], y_arr[train_idx]
        X_test, y_test = X_mat[test_idx], y_arr[test_idx]

        # Fit model on training fold only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Clone / fit
            import copy
            fold_model = copy.deepcopy(model)
            fold_model.fit(X_train, y_train)

            # Predict probabilities
            if hasattr(fold_model, "predict_proba"):
                probs = fold_model.predict_proba(X_test)[:, 1]
            elif hasattr(fold_model, "decision_function"):
                dec = fold_model.decision_function(X_test)
                probs = 1.0 / (1.0 + np.exp(-np.clip(dec, -30, 30)))
            else:
                probs = fold_model.predict(X_test).astype(float)

        oof_pred_proba[test_idx] = probs

        # Per-fold metrics
        if len(np.unique(y_test)) >= 2:
            fold_roc_aucs.append(float(roc_auc_score(y_test, probs)))
            fold_pr_aucs.append(float(average_precision_score(y_test, probs)))
            fold_pred_bin = (probs >= 0.5).astype(int)
            fold_balanced_accs.append(float(balanced_accuracy_score(y_test, fold_pred_bin)))

    # Overall Out-Of-Fold pooled performance
    pooled_metrics = compute_binary_metrics(y_arr, oof_pred_proba)
    ci_95 = compute_bootstrap_confidence_intervals(y_arr, oof_pred_proba, n_bootstraps=n_bootstraps, seed=random_state)

    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    prev = float(n_pos / max(len(y_arr), 1))

    return ModelEvaluationResult(
        model_name=model_name,
        n_samples=len(y_arr),
        n_subjects=len(np.unique(groups_arr)),
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prev,
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
        y_true=y_arr,
        y_pred_proba=oof_pred_proba,
        y_pred_bin=pooled_metrics["y_pred_bin"],
        subject_ids=groups_arr
    )
