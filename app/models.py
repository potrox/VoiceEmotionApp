from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .constants import EMOTIONS
from .storage import timestamp

logger = logging.getLogger(__name__)


ENSEMBLE_DEFINITIONS: Dict[str, Tuple[str, ...]] = {
    "Ансамбль SVM + MLP": ("SVM", "MLP"),
    "Ансамбль SVM + RandomForest": ("SVM", "Random Forest"),
    "Ансамбль MLP + RandomForest": ("MLP", "Random Forest"),
    "Ансамбль SVM + MLP + RandomForest": ("SVM", "MLP", "Random Forest"),
}


def sklearn_n_jobs() -> int:
    """Return a safe sklearn/joblib worker count.

    The default is 1 on purpose. On Windows, joblib multiprocessing can fail
    when TEMP or the user profile path contains Cyrillic characters. For this
    diploma application stability is more important than parallel grid-search
    speed, especially on small user datasets.
    """
    try:
        value = int(os.environ.get("VOICE_EMOTION_SKLEARN_N_JOBS", "1"))
    except ValueError:
        value = 1
    return value if value != 0 else 1

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class ModelEvaluation:
    name: str
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    macro_f1: float
    weighted_f1: float
    classification_report: str
    confusion_matrix: List[List[int]]
    per_class_f1: Dict[str, float]
    train_macro_f1: float
    overfit_gap: float
    cv_mean: float | None
    cv_std: float | None
    avg_inference_time_sec: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    complex_score: float = 0.0
    metric_intervals: Dict[str, List[float]] = field(default_factory=dict)
    test_support: int = 0
    roc_auc_macro: float | None = None
    average_precision_macro: float | None = None
    per_class_roc_auc: Dict[str, float] = field(default_factory=dict)
    per_class_average_precision: Dict[str, float] = field(default_factory=dict)
    raw_test_macro_f1: float | None = None
    threshold_cv_macro_f1: float | None = None
    threshold_cv_gain: float | None = None
    decision_biases: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrainingBundle:
    models: Dict[str, Any]
    best_model_name: str
    scaler: Any
    label_encoder: LabelEncoder
    feature_names: List[str]
    selector: Any
    preprocessing_params: Dict[str, Any]
    metrics: Dict[str, Any]
    dataset_info: Dict[str, Any]
    ensemble_weights: Dict[str, Any]
    created_at: str


class TorchMLP:
    def __init__(self, input_dim: int, num_classes: int, hidden_sizes: Tuple[int, ...] = (128, 64), lr: float = 1e-3, epochs: int = 80, batch_size: int = 32, use_gpu: bool = True):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch не установлен. Установите torch из requirements.txt.")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_sizes = tuple(hidden_sizes)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.use_gpu = bool(use_gpu)
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self._build().to(self.device)

    def _build(self):
        layers: List[Any] = []
        in_dim = self.input_dim
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.num_classes))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLP":
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optim.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def state(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_sizes": self.hidden_sizes,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "state_dict": self.model.state_dict(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TorchMLP":
        obj = cls(
            input_dim=state["input_dim"],
            num_classes=state["num_classes"],
            hidden_sizes=tuple(state["hidden_sizes"]),
            lr=state["lr"],
            epochs=state.get("epochs", 1),
            batch_size=state.get("batch_size", 32),
            use_gpu=state.get("use_gpu", False),
        )
        obj.model.load_state_dict(state["state_dict"])
        obj.model.to(obj.device)
        obj.model.eval()
        return obj


def safe_cv_folds(y: np.ndarray, max_folds: int = 5) -> int:
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        return 0
    min_count = int(np.min(counts))
    if min_count < 2:
        return 0
    return max(2, min(max_folds, min_count))


def safe_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int | None = None):
    labels, counts = np.unique(y, return_counts=True)
    stratify = y if len(labels) > 1 and int(np.min(counts)) >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=True)


def _normalize_speaker_ids(speaker_ids: List[str] | np.ndarray | None, n: int) -> np.ndarray | None:
    if speaker_ids is None:
        return None
    arr = np.asarray(speaker_ids, dtype=object)
    if len(arr) != n:
        return None
    return np.asarray([str(value).strip() or "unknown" for value in arr], dtype=object)


def _distribution(values: np.ndarray, labels: List[str] | None = None) -> Dict[str, int]:
    if values.size == 0:
        return {}
    keys, counts = np.unique(values, return_counts=True)
    result: Dict[str, int] = {}
    for key, count in zip(keys, counts):
        if labels is not None and isinstance(key, (int, np.integer)) and 0 <= int(key) < len(labels):
            result[str(labels[int(key)])] = int(count)
        else:
            result[str(key)] = int(count)
    return result


def _speaker_emotion_split_indices(
    y: np.ndarray,
    speaker_ids: List[str] | np.ndarray | None,
    test_size: float,
    random_state: int | None = None,
    labels: List[str] | None = None,
    split_name: str = "test",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Split every speaker/emotion subgroup into train and holdout parts.

    For the user dataset this is the safest default: each speaker contributes
    records to both parts, and every sufficiently represented emotion of that
    speaker is split according to the requested ratio. This prevents a random
    global split from accidentally putting a whole speaker or a whole emotion of
    one speaker only into train or only into test.
    """
    n = int(len(y))
    warnings: List[str] = []
    speaker_arr = _normalize_speaker_ids(speaker_ids, n)
    if speaker_arr is None:
        warnings.append(
            "В датасете нет корректного столбца speaker_id, поэтому разделение по дикторам невозможно; "
            "использовано обычное стратифицированное разделение по эмоциям."
        )
        all_idx = np.arange(n)
        stratify = y if len(np.unique(y)) > 1 and int(np.min(np.unique(y, return_counts=True)[1])) >= 2 else None
        train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=True)
        return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int), {
            "strategy": "fallback_emotion_stratified_no_speaker_id",
            "warnings": warnings,
        }

    rng = np.random.default_rng(random_state)
    train_indices: List[int] = []
    test_indices: List[int] = []
    group_rows: List[Dict[str, Any]] = []

    for speaker in sorted(set(speaker_arr.tolist())):
        speaker_mask = speaker_arr == speaker
        for class_id in sorted(np.unique(y[speaker_mask]).tolist()):
            idx = np.where(speaker_mask & (y == class_id))[0]
            idx = np.asarray(idx, dtype=int).copy()
            rng.shuffle(idx)
            n_group = int(len(idx))
            label_name = labels[int(class_id)] if labels is not None and 0 <= int(class_id) < len(labels) else str(class_id)
            if n_group < 2:
                train_part = idx
                test_part = np.asarray([], dtype=int)
                warnings.append(
                    f"Для диктора {speaker}, эмоции {label_name} найдена только {n_group} запись; "
                    f"её нельзя разделить {int((1 - test_size) * 100)}/{int(test_size * 100)}, поэтому она помещена в обучение."
                )
            else:
                train_ratio = 1.0 - float(test_size)
                train_count = int(round(n_group * train_ratio))
                train_count = max(1, min(n_group - 1, train_count))
                train_part = idx[:train_count]
                test_part = idx[train_count:]
            train_indices.extend(int(i) for i in train_part)
            test_indices.extend(int(i) for i in test_part)
            group_rows.append({
                "speaker_id": str(speaker),
                "emotion": label_name,
                "total": n_group,
                "train": int(len(train_part)),
                split_name: int(len(test_part)),
            })

    train_idx = np.asarray(train_indices, dtype=int)
    test_idx = np.asarray(test_indices, dtype=int)

    invalid = (
        train_idx.size == 0
        or test_idx.size == 0
        or len(np.unique(y[train_idx])) < 2
        or len(np.unique(y[test_idx])) < 2
    )
    if invalid:
        warnings.append(
            "Строгое разделение по диктору и эмоции невозможно из-за слишком малого или несбалансированного набора; "
            "использовано резервное стратифицированное разделение по эмоциям."
        )
        all_idx = np.arange(n)
        stratify = y if len(np.unique(y)) > 1 and int(np.min(np.unique(y, return_counts=True)[1])) >= 2 else None
        train_idx, test_idx = train_test_split(all_idx, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=True)
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        strategy = "fallback_emotion_stratified_too_small_for_speaker_emotion"
    else:
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        strategy = "speaker_emotion_stratified"

    info: Dict[str, Any] = {
        "strategy": strategy,
        "requested_test_size": float(test_size),
        "train_count": int(train_idx.size),
        f"{split_name}_count": int(test_idx.size),
        "train_class_distribution": _distribution(y[train_idx], labels),
        f"{split_name}_class_distribution": _distribution(y[test_idx], labels),
        "train_speaker_distribution": _distribution(speaker_arr[train_idx]),
        f"{split_name}_speaker_distribution": _distribution(speaker_arr[test_idx]),
        "group_rows": group_rows,
        "warnings": warnings,
    }
    return train_idx, test_idx, info


def evaluate_model(name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, labels: List[str], cv_scores: np.ndarray | None = None, best_params: Dict[str, Any] | None = None) -> ModelEvaluation:
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = max(time.perf_counter() - t0, 0.000001)
    avg_time = elapsed / max(len(X_test), 1)
    y_train_pred = model.predict(X_train)
    per_class = f1_score(y_test, y_pred, average=None, labels=list(range(len(labels))), zero_division=0)
    train_macro = float(f1_score(y_train, y_train_pred, average="macro", zero_division=0))
    macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    gap = max(0.0, train_macro - macro)
    warnings: List[str] = []
    if gap >= 0.15:
        warnings.append(f"Возможное переобучение: разрыв train/test macro F1 составляет {gap:.3f}.")
    cv_mean = float(np.mean(cv_scores)) if cv_scores is not None and len(cv_scores) else None
    cv_std = float(np.std(cv_scores)) if cv_scores is not None and len(cv_scores) else None
    stability = 1.0 - min(cv_std or 0.0, 1.0)
    overfit_component = 1.0 - min(gap, 1.0)
    complex_score = 0.40 * macro + 0.30 * float(balanced_accuracy_score(y_test, y_pred)) + 0.20 * overfit_component + 0.10 * stability
    return ModelEvaluation(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        macro_f1=macro,
        weighted_f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        classification_report=classification_report(y_test, y_pred, target_names=labels, zero_division=0),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=list(range(len(labels)))).tolist(),
        per_class_f1={labels[i]: float(v) for i, v in enumerate(per_class)},
        train_macro_f1=train_macro,
        overfit_gap=float(gap),
        cv_mean=cv_mean,
        cv_std=cv_std,
        avg_inference_time_sec=float(avg_time),
        best_params=best_params or {},
        warnings=warnings,
        complex_score=float(complex_score),
    )


class EnsembleSvmMlp:
    def __init__(self, svm: Any, mlp: TorchMLP, weights: Dict[int, Tuple[float, float]]):
        self.svm = svm
        self.mlp = mlp
        self.weights = weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_svm = self.svm.predict_proba(X)
        p_mlp = self.mlp.predict_proba(X)
        out = np.zeros_like(p_svm, dtype=np.float64)
        for i in range(p_svm.shape[1]):
            w_svm, w_mlp = self.weights.get(i, (0.5, 0.5))
            out[:, i] = w_svm * p_svm[:, i] + w_mlp * p_mlp[:, i]
        denom = out.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return out / denom

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class ProbabilityEnsemble:
    def __init__(self, models: Dict[str, Any], weights: Dict[int, Dict[str, float]]):
        if len(models) < 2:
            raise ValueError("Для объединённой модели нужно минимум две базовые модели.")
        self.models = models
        self.weights = weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = {name: model.predict_proba(X) for name, model in self.models.items()}
        first = next(iter(probas.values()))
        out = np.zeros_like(first, dtype=np.float64)
        model_names = list(probas.keys())
        uniform = 1.0 / max(len(model_names), 1)
        for class_idx in range(first.shape[1]):
            class_weights = self.weights.get(class_idx, {})
            for name in model_names:
                out[:, class_idx] += float(class_weights.get(name, uniform)) * probas[name][:, class_idx]
        denom = out.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return out / denom

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class Trainer:
    def __init__(self, test_size: float = 0.30, use_gpu: bool = True, feature_selection: str = "auto", random_state: int | None = None, hyperparameter_mode: str = "quality", selected_models: List[str] | None = None, parameter_mode: str = "auto", manual_params: Dict[str, Any] | None = None):
        self.test_size = test_size
        self.use_gpu = use_gpu
        self.feature_selection = feature_selection
        self.random_state = random_state
        self.hyperparameter_mode = hyperparameter_mode
        self.selected_models = selected_models or ["SVM", "MLP", "Random Forest", *ENSEMBLE_DEFINITIONS.keys()]
        self.parameter_mode = parameter_mode
        self.manual_params = manual_params or {}

    def _fit_selector(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.feature_selection != "auto" or X_train.shape[1] <= 80 or X_train.shape[0] < 20:
            return None, X_train
        k = min(max(32, X_train.shape[0] // 2), X_train.shape[1], 160)
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        return selector, X_new

    def _transform_selector(self, selector: Any, X: np.ndarray) -> np.ndarray:
        return selector.transform(X) if selector is not None else X

    def _train_svm(self, X_train, y_train) -> Tuple[Any, Dict[str, Any], np.ndarray | None]:
        cv_folds = safe_cv_folds(y_train)
        if self.parameter_mode == "manual":
            params = self.manual_params.get("svm", {})
            model = SVC(
                probability=True,
                class_weight="balanced",
                C=float(params.get("C", 1.0)),
                gamma=params.get("gamma", "scale"),
                kernel=params.get("kernel", "rbf"),
            ).fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1_macro", n_jobs=sklearn_n_jobs()) if cv_folds else None
            return model, {**params, "mode": "manual"}, scores

        base = SVC(probability=True, class_weight="balanced")
        grid = {
            "C": [0.1, 1, 5, 10, 30],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        }
        cv_folds = safe_cv_folds(y_train)
        if cv_folds:
            search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv_folds, n_jobs=sklearn_n_jobs())
            search.fit(X_train, y_train)
            model = search.best_estimator_
            params = search.best_params_
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1_macro", n_jobs=sklearn_n_jobs())
            return model, params, scores
        model = base.fit(X_train, y_train)
        return model, {"note": "CV отключена из-за малого числа записей"}, None

    def _train_rf(self, X_train, y_train) -> Tuple[Any, Dict[str, Any], np.ndarray | None]:
        cv_folds = safe_cv_folds(y_train)
        if self.parameter_mode == "manual":
            params = self.manual_params.get("random_forest", {})
            max_depth = params.get("max_depth", None)
            if max_depth in (0, "0", "", "None"):
                max_depth = None
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=sklearn_n_jobs(),
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=max_depth,
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            ).fit(X_train, y_train)
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1_macro", n_jobs=sklearn_n_jobs()) if cv_folds else None
            return model, {**params, "mode": "manual"}, scores

        base = RandomForestClassifier(class_weight="balanced", random_state=self.random_state, n_jobs=sklearn_n_jobs())
        grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16, 32],
            "min_samples_split": [2, 4, 6],
            "min_samples_leaf": [1, 2, 3],
        }
        cv_folds = safe_cv_folds(y_train)
        if cv_folds:
            search = GridSearchCV(base, grid, scoring="f1_macro", cv=cv_folds, n_jobs=sklearn_n_jobs())
            search.fit(X_train, y_train)
            model = search.best_estimator_
            params = search.best_params_
            scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1_macro", n_jobs=sklearn_n_jobs())
            return model, params, scores
        model = base.fit(X_train, y_train)
        return model, {"note": "CV отключена из-за малого числа записей"}, None

    def _train_mlp(self, X_train, y_train, X_val, y_val, num_classes: int) -> Tuple[TorchMLP, Dict[str, Any]]:
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")
        if self.parameter_mode == "manual":
            params = self.manual_params.get("mlp", {})
            model_params = {
                "hidden_sizes": tuple(int(x) for x in params.get("hidden_sizes", (128, 64))),
                "lr": float(params.get("lr", 1e-3)),
                "epochs": int(params.get("epochs", 80)),
                "batch_size": int(params.get("batch_size", 32)),
            }
            model = TorchMLP(X_train.shape[1], num_classes, use_gpu=self.use_gpu, **model_params)
            model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
            return model, {**model_params, "mode": "manual", "device": str(model.device)}

        if self.hyperparameter_mode == "quality":
            hidden_grid = [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)]
            lr_grid = [0.001, 0.0008, 0.0005]
            epoch_grid = [80, 120, 160]
            batch_grid = [16, 32, 64]
            candidates = [
                {"hidden_sizes": hidden, "lr": lr, "epochs": epochs, "batch_size": batch}
                for hidden in hidden_grid
                for lr in lr_grid
                for epochs in epoch_grid
                for batch in batch_grid
            ]
        else:
            candidates = [
                {"hidden_sizes": (64,), "lr": 1e-3, "epochs": 80, "batch_size": 32},
                {"hidden_sizes": (128,), "lr": 1e-3, "epochs": 80, "batch_size": 32},
                {"hidden_sizes": (128, 64), "lr": 1e-3, "epochs": 80, "batch_size": 32},
                {"hidden_sizes": (256, 128), "lr": 8e-4, "epochs": 120, "batch_size": 32},
                {"hidden_sizes": (256, 128, 64), "lr": 5e-4, "epochs": 120, "batch_size": 64},
            ]
        best_model = None
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        for params in candidates:
            model = TorchMLP(X_train.shape[1], num_classes, use_gpu=self.use_gpu, **params)
            model.fit(X_train, y_train)
            score = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
            if score > best_score:
                best_score = float(score)
                best_model = model
                best_params = {**params, "validation_macro_f1": best_score, "device": str(model.device), "candidate_count": len(candidates), "search_mode": self.hyperparameter_mode}
        assert best_model is not None

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        final_model = TorchMLP(X_all.shape[1], num_classes, use_gpu=self.use_gpu, **{k: best_params[k] for k in ["hidden_sizes", "lr", "epochs", "batch_size"]})
        final_model.fit(X_all, y_all)
        return final_model, best_params

    def _ensemble_weights(self, labels: List[str], base_models: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray) -> Dict[int, Dict[str, float]]:
        predictions = {name: model.predict(X_val) for name, model in base_models.items()}
        weights: Dict[int, Dict[str, float]] = {}
        eps = 1e-6
        for i, _label in enumerate(labels):
            y_bin = (y_val == i).astype(int)
            class_scores: Dict[str, float] = {}
            for name, pred in predictions.items():
                pred_bin = (pred == i).astype(int)
                class_scores[name] = float(f1_score(y_bin, pred_bin, zero_division=0))
            total = sum(class_scores.values()) + eps * max(len(class_scores), 1)
            weights[i] = {name: float((score + eps) / total) for name, score in class_scores.items()}
        return weights

    def _weights_for_report(self, labels: List[str], weights: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {labels[i]: {name: float(value) for name, value in model_weights.items()} for i, model_weights in weights.items()}

    def train_all(self, X: np.ndarray, y_text: np.ndarray, feature_names: List[str], dataset_info: Dict[str, Any], preprocessing_params: Dict[str, Any], speaker_ids: List[str] | np.ndarray | None = None) -> Tuple[TrainingBundle, Dict[str, ModelEvaluation]]:
        if len(np.unique(y_text)) < 2:
            raise ValueError("Для обучения необходимо минимум два класса эмоций.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_text)
        labels = list(label_encoder.classes_)
        speaker_arr = _normalize_speaker_ids(speaker_ids, len(y))
        train_idx, test_idx, split_info = _speaker_emotion_split_indices(
            y,
            speaker_arr,
            test_size=self.test_size,
            random_state=self.random_state,
            labels=labels,
            split_name="test",
        )
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        speaker_train = speaker_arr[train_idx] if speaker_arr is not None else None

        dataset_info["split_strategy"] = split_info.get("strategy")
        dataset_info["split_info"] = split_info
        dataset_info["split_warnings"] = split_info.get("warnings", [])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        selector, X_train = self._fit_selector(X_train_scaled, y_train)
        X_test = self._transform_selector(selector, X_test_scaled)

        fit_idx, val_idx, validation_split_info = _speaker_emotion_split_indices(
            y_train,
            speaker_train,
            test_size=0.20,
            random_state=self.random_state,
            labels=labels,
            split_name="validation",
        )
        X_fit, X_val = X_train[fit_idx], X_train[val_idx]
        y_fit, y_val = y_train[fit_idx], y_train[val_idx]
        dataset_info["validation_split_info"] = validation_split_info
        dataset_info["split_warnings"] = dataset_info.get("split_warnings", []) + validation_split_info.get("warnings", [])

        selected = set(self.selected_models)
        if not selected:
            raise ValueError("Не выбрана ни одна модель для обучения.")
        selected_ensembles = [name for name in ENSEMBLE_DEFINITIONS if name in selected]
        required_for_ensembles = {model_name for ensemble_name in selected_ensembles for model_name in ENSEMBLE_DEFINITIONS[ensemble_name]}
        train_svm = "SVM" in selected or "SVM" in required_for_ensembles
        train_mlp = "MLP" in selected or "MLP" in required_for_ensembles
        train_rf = "Random Forest" in selected or "Random Forest" in required_for_ensembles

        models: Dict[str, Any] = {}
        evaluations: Dict[str, ModelEvaluation] = {}
        ensemble_weights_by_name: Dict[str, Dict[str, Dict[str, float]]] = {}

        if train_svm:
            svm, svm_params, svm_cv = self._train_svm(X_train, y_train)
            models["SVM"] = svm
            if "SVM" in selected:
                evaluations["SVM"] = evaluate_model("SVM", svm, X_train, y_train, X_test, y_test, labels, svm_cv, svm_params)

        if train_rf:
            rf, rf_params, rf_cv = self._train_rf(X_train, y_train)
            models["Random Forest"] = rf
            if "Random Forest" in selected:
                evaluations["Random Forest"] = evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test, labels, rf_cv, rf_params)

        if train_mlp:
            mlp, mlp_params = self._train_mlp(X_fit, y_fit, X_val, y_val, len(labels))
            models["MLP"] = mlp
            if "MLP" in selected:
                evaluations["MLP"] = evaluate_model("MLP", mlp, X_train, y_train, X_test, y_test, labels, None, mlp_params)

        for ensemble_name in selected_ensembles:
            base_names = ENSEMBLE_DEFINITIONS[ensemble_name]
            missing = [name for name in base_names if name not in models]
            if missing:
                raise RuntimeError(f"Для объединённой модели {ensemble_name} не обучены базовые модели: {', '.join(missing)}.")
            base_models = {name: models[name] for name in base_names}
            weights = self._ensemble_weights(labels, base_models, X_val, y_val)
            ensemble = ProbabilityEnsemble(base_models, weights)
            models[ensemble_name] = ensemble
            weights_report = self._weights_for_report(labels, weights)
            ensemble_weights_by_name[ensemble_name] = weights_report
            evaluations[ensemble_name] = evaluate_model(
                ensemble_name,
                ensemble,
                X_train,
                y_train,
                X_test,
                y_test,
                labels,
                None,
                {"base_models": list(base_names), "weights_by_class": weights_report},
            )
            evaluations[ensemble_name].complex_score += 0.02

        if not evaluations:
            raise ValueError("Выбранная комбинация моделей не дала результатов для оценки.")
        best_name = max(evaluations.values(), key=lambda ev: ev.complex_score).name
        bundle = TrainingBundle(
            models=models,
            best_model_name=best_name,
            scaler=scaler,
            label_encoder=label_encoder,
            feature_names=feature_names,
            selector=selector,
            preprocessing_params=preprocessing_params,
            metrics={k: v.to_dict() for k, v in evaluations.items()},
            dataset_info=dataset_info,
            ensemble_weights=ensemble_weights_by_name,
            created_at=timestamp(),
        )
        return bundle, evaluations


def save_bundle(bundle: TrainingBundle, models_dir: str | Path) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"model_{timestamp()}.pkl"
    serializable = bundle.__dict__.copy()

    mlp_model = serializable["models"].get("MLP")
    if mlp_model is not None and hasattr(mlp_model, "state"):
        # Legacy TorchMLP models need an explicit tensor state.  The current
        # sklearn MLP is already fully serializable by joblib.
        serializable["mlp_state"] = mlp_model.state()
    joblib.dump(serializable, path)
    protocol = serializable.get("preprocessing_params", {}).get("evaluation_protocol", "")
    pointer_name = "best_model_honest.txt" if protocol == "speaker_independent_grouped_cv" else "best_model.txt"
    best_file = models_dir / pointer_name
    best_file.write_text(str(path), encoding="utf-8")
    return path


def load_bundle(path: str | Path) -> Dict[str, Any]:
    return joblib.load(path)


def transform_features(bundle: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    preprocessing_params = bundle.get("preprocessing_params") or {}
    if preprocessing_params.get("preprocessing_in_model"):
        return X
    scaler = bundle.get("scaler")
    Xs = scaler.transform(X) if scaler is not None else X
    selector = bundle.get("selector")
    if selector is not None:
        Xs = selector.transform(Xs)
    return Xs
