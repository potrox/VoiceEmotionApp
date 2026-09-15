"""Разделение выборки и расчёт метрик качества моделей."""

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from .types import ModelEvaluation

def safe_cv_folds(target_labels: np.ndarray, max_folds: int = 5) -> int:
    unique_labels, label_counts = np.unique(target_labels, return_counts=True)
    if len(unique_labels) < 2:
        return 0
    min_count = int(np.min(label_counts))
    if min_count < 2:
        return 0
    return max(2, min(max_folds, min_count))


def build_cv_strategy(
    target_labels: np.ndarray,
    speaker_ids: list[str] | np.ndarray | None,
    random_state: int = 42,
    max_folds: int = 5,
) -> tuple[int | StratifiedGroupKFold, np.ndarray | None, str] | tuple[None, None, str]:
    normalized_speaker_ids = _normalize_speaker_ids(speaker_ids, len(target_labels))
    if normalized_speaker_ids is not None:
        unique_speaker_count = len(np.unique(normalized_speaker_ids))
        speaker_counts_by_class = [
            len(np.unique(normalized_speaker_ids[target_labels == class_id]))
            for class_id in np.unique(target_labels)
        ]
        fold_count = min(
            max_folds,
            unique_speaker_count,
            min(speaker_counts_by_class, default=0),
        )
        if fold_count >= 2:
            return (
                StratifiedGroupKFold(
                    n_splits=fold_count,
                    shuffle=True,
                    random_state=random_state,
                ),
                normalized_speaker_ids,
                f"stratified_group_{fold_count}_fold",
            )

    fold_count = safe_cv_folds(target_labels, max_folds=max_folds)
    if fold_count:
        return fold_count, None, f"fallback_stratified_{fold_count}_fold"
    return None, None, "disabled_insufficient_samples"

def _normalize_speaker_ids(
    speaker_ids: list[str] | np.ndarray | None, sample_count: int
) -> np.ndarray | None:
    if speaker_ids is None:
        return None
    normalized_ids = np.asarray(speaker_ids, dtype=object)
    if len(normalized_ids) != sample_count:
        return None
    return np.asarray([str(value).strip() or "unknown" for value in normalized_ids], dtype=object)

def _distribution(
    values: np.ndarray, class_names: list[str] | None = None
) -> dict[str, int]:
    if values.size == 0:
        return {}
    keys, counts = np.unique(values, return_counts=True)
    result: dict[str, int] = {}
    for key, count in zip(keys, counts):
        if class_names is not None and isinstance(key, (int, np.integer)) and 0 <= int(key) < len(class_names):
            result[str(class_names[int(key)])] = int(count)
        else:
            result[str(key)] = int(count)
    return result

def _fallback_stratified_split(
    target_labels: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = np.arange(len(target_labels))
    _, class_counts = np.unique(target_labels, return_counts=True)
    stratification_labels = (
        target_labels
        if len(class_counts) > 1 and int(np.min(class_counts)) >= 2
        else None
    )
    training_indices, holdout_indices = train_test_split(
        sample_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification_labels,
        shuffle=True,
    )
    missing_training_classes = set(np.unique(target_labels)) - set(
        np.unique(target_labels[training_indices])
    )
    for class_id in missing_training_classes:
        class_candidates = holdout_indices[target_labels[holdout_indices] == class_id]
        moved_index = int(class_candidates[0])
        training_indices = np.append(training_indices, moved_index)
        holdout_indices = holdout_indices[holdout_indices != moved_index]
    if len(np.unique(target_labels[holdout_indices])) < 2:
        raise ValueError(
            "Недостаточно записей классов для отдельной выборки минимум с двумя "
            "эмоциями. Добавьте данные или уменьшите долю test."
        )
    return np.asarray(training_indices, dtype=int), np.asarray(holdout_indices, dtype=int)


def _best_group_split(
    target_labels: np.ndarray,
    speaker_ids: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    all_classes = set(np.unique(target_labels).tolist())
    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=test_size,
        random_state=random_state,
    )
    best_split: tuple[np.ndarray, np.ndarray] | None = None
    best_score: tuple[int, float] | None = None
    for training_indices, holdout_indices in splitter.split(
        np.zeros(len(target_labels)), target_labels, groups=speaker_ids
    ):
        training_classes = set(np.unique(target_labels[training_indices]).tolist())
        holdout_classes = set(np.unique(target_labels[holdout_indices]).tolist())
        missing_class_count = len(all_classes - training_classes) + len(
            all_classes - holdout_classes
        )
        candidate_score = (
            missing_class_count,
            abs(len(holdout_indices) / len(target_labels) - test_size),
        )
        if best_score is None or candidate_score < best_score:
            best_split = (
                np.asarray(training_indices, dtype=int),
                np.asarray(holdout_indices, dtype=int),
            )
            best_score = candidate_score
    return best_split


def _is_usable_group_split(
    target_labels: np.ndarray,
    split_indices: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    if split_indices is None:
        return False
    training_indices, holdout_indices = split_indices
    all_classes = set(np.unique(target_labels).tolist())
    return (
        training_indices.size > 0
        and holdout_indices.size > 0
        and set(np.unique(target_labels[training_indices]).tolist()) == all_classes
        and len(np.unique(target_labels[holdout_indices])) >= 2
    )


def _readable_class_names(
    class_ids: set[Any], class_names: list[str] | None
) -> list[str]:
    return [
        class_names[int(class_id)]
        if class_names is not None and int(class_id) < len(class_names)
        else str(class_id)
        for class_id in sorted(class_ids)
    ]


def _split_details(
    target_labels: np.ndarray,
    speaker_ids: np.ndarray | None,
    training_indices: np.ndarray,
    holdout_indices: np.ndarray,
    test_size: float,
    split_name: str,
    strategy: str,
    warnings: list[str],
    class_names: list[str] | None,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "requested_test_size": float(test_size),
        "train_count": int(training_indices.size),
        f"{split_name}_count": int(holdout_indices.size),
        "train_class_distribution": _distribution(
            target_labels[training_indices], class_names
        ),
        f"{split_name}_class_distribution": _distribution(
            target_labels[holdout_indices], class_names
        ),
        "train_speaker_distribution": (
            _distribution(speaker_ids[training_indices])
            if speaker_ids is not None
            else {}
        ),
        f"{split_name}_speaker_distribution": (
            _distribution(speaker_ids[holdout_indices])
            if speaker_ids is not None
            else {}
        ),
        "speaker_overlap_count": (
            len(
                set(speaker_ids[training_indices].tolist())
                & set(speaker_ids[holdout_indices].tolist())
            )
            if speaker_ids is not None
            else None
        ),
        "warnings": warnings,
    }


def _group_aware_split_indices(
    target_labels: np.ndarray,
    speaker_ids: list[str] | np.ndarray | None,
    test_size: float,
    random_state: int | None = None,
    class_names: list[str] | None = None,
    split_name: str = "test",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_count = int(len(target_labels))
    if sample_count < 4:
        raise ValueError("Для разделения выборки требуется минимум четыре записи.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("Доля отложенной выборки должна находиться между 0 и 1.")

    warnings: list[str] = []
    effective_random_state = 42 if random_state is None else int(random_state)
    normalized_speaker_ids = _normalize_speaker_ids(speaker_ids, sample_count)
    has_multiple_speakers = (
        normalized_speaker_ids is not None
        and len(np.unique(normalized_speaker_ids)) >= 2
    )

    if not has_multiple_speakers:
        warnings.append(
            "В датасете нет как минимум двух корректных speaker_id, поэтому "
            "разделение по дикторам невозможно; использовано обычное "
            "стратифицированное разделение по эмоциям."
        )
        training_indices, holdout_indices = _fallback_stratified_split(
            target_labels, test_size, effective_random_state
        )
        strategy = "fallback_emotion_stratified_no_speaker_id"
        normalized_speaker_ids = None
    else:
        group_split = _best_group_split(
            target_labels,
            normalized_speaker_ids,
            test_size,
            effective_random_state,
        )
        if group_split is not None and _is_usable_group_split(target_labels, group_split):
            training_indices, holdout_indices = group_split
            strategy = "speaker_disjoint_group_shuffle"
            missing_holdout_classes = set(np.unique(target_labels).tolist()) - set(
                np.unique(target_labels[holdout_indices]).tolist()
            )
            if missing_holdout_classes:
                readable_classes = _readable_class_names(
                    missing_holdout_classes, class_names
                )
                warnings.append(
                    f"В части {split_name} отсутствуют классы: "
                    f"{', '.join(readable_classes)}. Для более устойчивой оценки "
                    "нужны записи этих эмоций от дополнительных дикторов."
                )
        else:
            training_indices, holdout_indices = _fallback_stratified_split(
                target_labels, test_size, effective_random_state
            )
            strategy = "fallback_emotion_stratified_insufficient_speaker_groups"
            warnings.append(
                "Не удалось получить две части минимум с двумя классами при "
                "непересекающихся дикторах; использовано резервное "
                "стратифицированное разделение по эмоциям, поэтому метрика "
                "может быть оптимистичной."
            )

    details = _split_details(
        target_labels,
        normalized_speaker_ids,
        training_indices,
        holdout_indices,
        test_size,
        split_name,
        strategy,
        warnings,
        class_names,
    )
    return training_indices, holdout_indices, details

def evaluate_model(
    name: str,
    model: Any,
    training_features: np.ndarray,
    training_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    cv_scores: np.ndarray | None = None,
    best_params: dict[str, Any] | None = None,
) -> ModelEvaluation:
    started_at = time.perf_counter()
    predicted_labels = model.predict(test_features)
    elapsed_seconds = max(time.perf_counter() - started_at, 0.000001)
    average_inference_seconds = elapsed_seconds / max(len(test_features), 1)
    training_predictions = model.predict(training_features)
    per_class_scores = f1_score(
        test_labels,
        predicted_labels,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    train_macro = float(f1_score(training_labels, training_predictions, average="macro", zero_division=0))
    macro = float(f1_score(test_labels, predicted_labels, average="macro", zero_division=0))
    balanced_accuracy = float(
        balanced_accuracy_score(test_labels, predicted_labels)
    )
    overfitting_gap = max(0.0, train_macro - macro)
    warnings: list[str] = []
    if overfitting_gap >= 0.15:
        warnings.append(f"Возможное переобучение: разрыв train/test macro F1 составляет {overfitting_gap:.3f}.")
    cv_mean = float(np.mean(cv_scores)) if cv_scores is not None and len(cv_scores) else None
    cv_std = float(np.std(cv_scores)) if cv_scores is not None and len(cv_scores) else None
    return ModelEvaluation(
        name=name,
        accuracy=float(accuracy_score(test_labels, predicted_labels)),
        balanced_accuracy=balanced_accuracy,
        precision_macro=float(precision_score(test_labels, predicted_labels, average="macro", zero_division=0)),
        recall_macro=float(recall_score(test_labels, predicted_labels, average="macro", zero_division=0)),
        macro_f1=macro,
        weighted_f1=float(f1_score(test_labels, predicted_labels, average="weighted", zero_division=0)),
        classification_report=classification_report(
            test_labels,
            predicted_labels,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(
            test_labels, predicted_labels, labels=list(range(len(class_names)))
        ).tolist(),
        per_class_f1={class_names[i]: float(v) for i, v in enumerate(per_class_scores)},
        train_macro_f1=train_macro,
        overfit_gap=float(overfitting_gap),
        cv_mean=cv_mean,
        cv_std=cv_std,
        avg_inference_time_sec=float(average_inference_seconds),
        best_params=best_params or {},
        warnings=warnings,
    )
