from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import label_binarize

from .models import ModelEvaluation, TrainingBundle
from .storage import timestamp


SUPPORTED_MODELS = ("Logistic Regression", "Random Forest", "MLP")
MODEL_COMPLEXITY_ORDER = {name: index for index, name in enumerate(SUPPORTED_MODELS)}
PRACTICAL_F1_MARGIN = 0.01


def _select_deployment_model(
    evaluations: Dict[str, ModelEvaluation],
    margin: float = PRACTICAL_F1_MARGIN,
) -> str:
    """Prefer the cheapest model when its grouped-OOF result is practically tied."""
    best_score = max(float(item.cv_mean or 0.0) for item in evaluations.values())
    eligible = [
        item
        for item in evaluations.values()
        if float(item.cv_mean or 0.0) >= best_score - float(margin)
    ]
    return min(
        eligible,
        key=lambda item: MODEL_COMPLEXITY_ORDER.get(item.name, len(MODEL_COMPLEXITY_ORDER)),
    ).name


def _base_probability_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Модель не предоставляет predict_proba или decision_function.")
    return np.clip(probabilities, 1e-9, 1.0)


class ClassBiasClassifier:
    """Class offsets learned only from grouped out-of-fold train predictions."""

    def __init__(self, base_model: Any, class_biases: np.ndarray):
        self.base_model = base_model
        self.class_biases = np.asarray(class_biases, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        adjusted = np.log(_base_probability_matrix(self.base_model, X))
        adjusted += self.class_biases.reshape(1, -1)
        adjusted -= adjusted.max(axis=1, keepdims=True)
        probabilities = np.exp(adjusted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def _group_oof_probabilities(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
) -> np.ndarray:
    class_count = len(np.unique(y))
    probabilities = np.zeros((len(y), class_count), dtype=float)
    assigned = np.zeros(len(y), dtype=bool)
    for fit_idx, validation_idx in cv.split(X, y, groups):
        fold_model = clone(estimator)
        fold_model.fit(X[fit_idx], y[fit_idx])
        probabilities[validation_idx] = _base_probability_matrix(
            fold_model, X[validation_idx]
        )
        assigned[validation_idx] = True
    if not bool(np.all(assigned)):
        raise RuntimeError("Не для всех train-записей получены out-of-fold предсказания.")
    return probabilities


def _tune_class_biases(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    log_scores = np.log(np.clip(probabilities, 1e-9, 1.0))
    biases = np.zeros(probabilities.shape[1], dtype=float)

    def score(values: np.ndarray) -> float:
        predictions = np.argmax(log_scores + values.reshape(1, -1), axis=1)
        return float(f1_score(y, predictions, average="macro", zero_division=0))

    raw_score = score(biases)
    best_score = raw_score
    offsets = np.linspace(-0.40, 0.40, 17)
    for _ in range(3):
        improved = False
        for class_index in range(probabilities.shape[1]):
            current = biases[class_index]
            candidates: List[Tuple[float, np.ndarray]] = []
            for offset in offsets:
                candidate = biases.copy()
                candidate[class_index] = current + float(offset)
                candidates.append((score(candidate), candidate))
            candidate_score, candidate_biases = max(candidates, key=lambda item: item[0])
            if candidate_score > best_score + 1e-6:
                biases = candidate_biases
                best_score = candidate_score
                improved = True
        biases -= biases.mean()
        if not improved:
            break
    if best_score < raw_score + 0.001:
        return np.zeros_like(biases), raw_score, raw_score
    return biases, raw_score, best_score


def _normalize_speaker_ids(speaker_ids: List[str] | np.ndarray | None, n: int) -> np.ndarray:
    if speaker_ids is None or len(speaker_ids) != n:
        raise ValueError(
            "Для честной speaker-independent оценки CSV должен содержать speaker_id для каждой записи."
        )
    groups = np.asarray([str(value).strip() for value in speaker_ids], dtype=object)
    invalid = np.isin(np.char.lower(groups.astype(str)), ["", "unknown", "none", "nan"])
    if bool(np.any(invalid)):
        raise ValueError(
            "Найдены записи без speaker_id. Сначала заполните идентификаторы дикторов; "
            "случайное деление аудиофайлов завышает качество."
        )
    return groups


def _distribution(values: np.ndarray, labels: List[str] | None = None) -> Dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    result: Dict[str, int] = {}
    for key, count in zip(keys, counts):
        if labels is not None and isinstance(key, (int, np.integer)):
            name = labels[int(key)]
        else:
            name = str(key)
        result[name] = int(count)
    return result


def _minimum_class_group_count(y: np.ndarray, groups: np.ndarray) -> int:
    return min(len(np.unique(groups[y == class_id])) for class_id in np.unique(y))


def make_group_cv(
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    max_splits: int = 5,
) -> StratifiedGroupKFold:
    unique_groups = len(np.unique(groups))
    class_group_count = _minimum_class_group_count(y, groups)
    n_splits = min(int(max_splits), unique_groups, class_group_count)
    if n_splits < 3:
        raise ValueError(
            "Для устойчивой групповой кросс-валидации нужно минимум три разных диктора "
            "в каждом классе эмоций."
        )
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))


def speaker_disjoint_holdout_indices(
    y: np.ndarray,
    speaker_ids: List[str] | np.ndarray | None,
    test_size: float = 0.20,
    random_state: int = 42,
    labels: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build a deterministic holdout with no speaker overlap.

    StratifiedGroupKFold is used as a practical grouped holdout generator.  We
    select the fold closest to the requested size and class distribution, while
    requiring every class to be present on both sides.
    """
    groups = _normalize_speaker_ids(speaker_ids, len(y))
    desired_splits = max(3, int(round(1.0 / max(float(test_size), 0.05))))
    max_allowed = min(len(np.unique(groups)), _minimum_class_group_count(y, groups))
    n_splits = min(desired_splits, max_allowed)
    if n_splits < 3:
        raise ValueError(
            "Невозможно сформировать честный holdout: нужно минимум три диктора "
            "в каждом классе. Добавьте данные или сократите число классов."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    all_classes = set(np.unique(y).tolist())
    overall = np.bincount(y, minlength=len(all_classes)) / len(y)
    candidates: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y, groups):
        if set(np.unique(y[train_idx]).tolist()) != all_classes:
            continue
        if set(np.unique(y[test_idx]).tolist()) != all_classes:
            continue
        ratio_error = abs((len(test_idx) / len(y)) - float(test_size))
        test_distribution = np.bincount(y[test_idx], minlength=len(all_classes)) / len(test_idx)
        distribution_error = float(np.abs(overall - test_distribution).mean())
        candidates.append((ratio_error + distribution_error, train_idx, test_idx))
    if not candidates:
        raise ValueError(
            "Не удалось получить speaker-independent test с присутствием всех классов. "
            "Нужно больше дикторов или меньше классов."
        )

    _, train_idx, test_idx = min(candidates, key=lambda item: item[0])
    train_speakers = set(groups[train_idx].tolist())
    test_speakers = set(groups[test_idx].tolist())
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise RuntimeError("Внутренняя ошибка: обнаружено пересечение дикторов train/test.")

    info = {
        "strategy": "speaker_disjoint_stratified_group_kfold",
        "random_state": int(random_state),
        "requested_test_size": float(test_size),
        "actual_test_size": float(len(test_idx) / len(y)),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "train_speaker_count": int(len(train_speakers)),
        "test_speaker_count": int(len(test_speakers)),
        "speaker_overlap_count": 0,
        "train_speaker_distribution": _distribution(groups[train_idx]),
        "test_speaker_distribution": _distribution(groups[test_idx]),
        "train_class_distribution": _distribution(y[train_idx], labels),
        "test_class_distribution": _distribution(y[test_idx], labels),
        "warnings": [],
    }
    return np.asarray(train_idx), np.asarray(test_idx), info


def predefined_holdout_indices(
    y: np.ndarray,
    speaker_ids: List[str] | np.ndarray | None,
    split_labels: List[str] | np.ndarray,
    labels: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Use a corpus-provided train/test split after validating speaker isolation."""
    groups = _normalize_speaker_ids(speaker_ids, len(y))
    splits = np.asarray([str(value).strip().lower() for value in split_labels], dtype=object)
    if len(splits) != len(y) or not set(np.unique(splits)).issubset({"train", "test"}):
        raise ValueError("dataset_split должен содержать только train/test для каждой записи.")
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Предопределённый протокол требует непустые train и test.")
    all_classes = set(np.unique(y).tolist())
    if set(np.unique(y[train_idx]).tolist()) != all_classes or set(np.unique(y[test_idx]).tolist()) != all_classes:
        raise ValueError("В официальных train и test должны присутствовать все классы эмоций.")
    train_speakers = set(groups[train_idx].tolist())
    test_speakers = set(groups[test_idx].tolist())
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise ValueError(
            "Предопределённый train/test содержит пересечение дикторов: "
            + ", ".join(overlap[:10])
        )
    info = {
        "strategy": "predefined_speaker_disjoint_holdout",
        "random_state": None,
        "requested_test_size": None,
        "actual_test_size": float(len(test_idx) / len(y)),
        "train_count": int(len(train_idx)),
        "test_count": int(len(test_idx)),
        "train_speaker_count": int(len(train_speakers)),
        "test_speaker_count": int(len(test_speakers)),
        "speaker_overlap_count": 0,
        "train_speaker_distribution": _distribution(groups[train_idx]),
        "test_speaker_distribution": _distribution(groups[test_idx]),
        "train_class_distribution": _distribution(y[train_idx], labels),
        "test_class_distribution": _distribution(y[test_idx], labels),
        "warnings": [],
    }
    return train_idx, test_idx, info


def _cluster_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    score_matrix: np.ndarray | None = None,
    samples: int = 500,
) -> Dict[str, List[float]]:
    """95% confidence intervals by resampling whole speakers, not clips."""
    rng = np.random.default_rng(int(random_state))
    unique_groups = np.unique(groups)
    macro_scores: List[float] = []
    balanced_scores: List[float] = []
    roc_scores: List[float] = []
    average_precision_scores: List[float] = []
    class_ids = np.arange(score_matrix.shape[1]) if score_matrix is not None else None
    for _ in range(int(samples)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.where(groups == group)[0] for group in sampled_groups])
        yt = y_true[indices]
        yp = y_pred[indices]
        macro_scores.append(float(f1_score(yt, yp, average="macro", zero_division=0)))
        balanced_scores.append(float(balanced_accuracy_score(yt, yp)))
        if score_matrix is not None and class_ids is not None:
            binary = label_binarize(yt, classes=class_ids)
            if binary.shape[1] == len(class_ids) and np.all(binary.sum(axis=0) > 0):
                sampled_scores = score_matrix[indices]
                roc_scores.append(float(roc_auc_score(binary, sampled_scores, average="macro")))
                average_precision_scores.append(
                    float(average_precision_score(binary, sampled_scores, average="macro"))
                )

    def interval(values: List[float]) -> List[float]:
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        return [float(low), float(high)]

    result = {
        "macro_f1_95ci": interval(macro_scores),
        "balanced_accuracy_95ci": interval(balanced_scores),
    }
    if roc_scores:
        result["roc_auc_macro_95ci"] = interval(roc_scores)
        result["average_precision_macro_95ci"] = interval(average_precision_scores)
    return result


def _pipeline(model: Any, feature_count: int, feature_selection: str) -> Pipeline:
    if feature_selection == "none":
        selector: Any = "passthrough"
    else:
        selector = SelectKBest(score_func=f_classif, k=min(96, int(feature_count)))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", selector),
            ("model", model),
        ]
    )


def _score_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores
    raise ValueError("Модель не предоставляет predict_proba или decision_function для ROC-AUC.")


def _evaluate(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_groups: np.ndarray,
    labels: List[str],
    cv_mean: float,
    cv_std: float,
    best_params: Dict[str, Any],
    random_state: int,
) -> ModelEvaluation:
    started = time.perf_counter()
    y_pred = model.predict(X_test)
    avg_time = (time.perf_counter() - started) / max(len(X_test), 1)
    y_train_pred = model.predict(X_train)
    score_matrix = _score_matrix(model, X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    train_macro_f1 = float(f1_score(y_train, y_train_pred, average="macro", zero_division=0))
    raw_test_macro_f1 = macro_f1
    if isinstance(model, ClassBiasClassifier):
        raw_test_macro_f1 = float(
            f1_score(
                y_test,
                model.base_model.predict(X_test),
                average="macro",
                zero_division=0,
            )
        )
    gap = max(0.0, train_macro_f1 - macro_f1)
    warnings: List[str] = []
    if gap >= 0.15:
        warnings.append(f"Переобучение: разрыв train/test macro F1 = {gap:.3f}.")
    class_ids = np.arange(len(labels))
    binary_test = label_binarize(y_test, classes=class_ids)
    roc_auc_macro = float(roc_auc_score(binary_test, score_matrix, average="macro"))
    average_precision_macro = float(
        average_precision_score(binary_test, score_matrix, average="macro")
    )
    per_class_roc_auc = {
        label: float(roc_auc_score(binary_test[:, index], score_matrix[:, index]))
        for index, label in enumerate(labels)
    }
    per_class_average_precision = {
        label: float(average_precision_score(binary_test[:, index], score_matrix[:, index]))
        for index, label in enumerate(labels)
    }
    intervals = _cluster_bootstrap_intervals(
        y_test, y_pred, test_groups, random_state, score_matrix=score_matrix
    )
    per_class = f1_score(
        y_test,
        y_pred,
        average=None,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    return ModelEvaluation(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        macro_f1=macro_f1,
        weighted_f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        classification_report=classification_report(
            y_test,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(
            y_test, y_pred, labels=list(range(len(labels)))
        ).tolist(),
        per_class_f1={labels[index]: float(value) for index, value in enumerate(per_class)},
        train_macro_f1=train_macro_f1,
        overfit_gap=float(gap),
        cv_mean=float(cv_mean),
        cv_std=float(cv_std),
        avg_inference_time_sec=float(avg_time),
        best_params=best_params,
        warnings=warnings,
        complex_score=float(cv_mean),
        metric_intervals=intervals,
        test_support=int(len(y_test)),
        roc_auc_macro=roc_auc_macro,
        average_precision_macro=average_precision_macro,
        per_class_roc_auc=per_class_roc_auc,
        per_class_average_precision=per_class_average_precision,
        raw_test_macro_f1=raw_test_macro_f1,
        threshold_cv_macro_f1=float(best_params.get("threshold_cv_macro_f1", cv_mean)),
        threshold_cv_gain=float(best_params.get("threshold_cv_gain", 0.0)),
        decision_biases={
            labels[index]: float(value)
            for index, value in enumerate(getattr(model, "class_biases", []))
        },
    )


class Trainer:
    """Cost-conscious, leakage-resistant trainer.

    The test set is speaker-disjoint and is never used for hyperparameter or
    model selection.  Every preprocessing step is fitted inside grouped CV.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        use_gpu: bool = False,
        feature_selection: str = "auto",
        random_state: int | None = 42,
        hyperparameter_mode: str = "balanced",
        selected_models: List[str] | None = None,
        parameter_mode: str = "auto",
        manual_params: Dict[str, Any] | None = None,
    ):
        del use_gpu, hyperparameter_mode
        self.test_size = float(test_size)
        self.feature_selection = feature_selection
        self.random_state = 42 if random_state is None else int(random_state)
        self.selected_models = selected_models or list(SUPPORTED_MODELS)
        self.parameter_mode = parameter_mode
        self.manual_params = manual_params or {}

    def _model_spec(self, name: str, feature_count: int) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        if name == "Logistic Regression":
            estimator = LogisticRegression(
                class_weight="balanced",
                max_iter=1200,
                solver="lbfgs",
                random_state=self.random_state,
            )
            pipeline = _pipeline(estimator, feature_count, self.feature_selection)
            if self.parameter_mode == "manual":
                c_value = float(self.manual_params.get("logistic_regression", {}).get("C", 1.0))
                pipeline.set_params(model__C=c_value)
                return pipeline, {}
            return pipeline, {"model__C": [0.03, 0.1, 0.3, 1.0]}

        if name == "Random Forest":
            estimator = RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=self.random_state,
            )
            pipeline = _pipeline(estimator, feature_count, self.feature_selection)
            if self.parameter_mode == "manual":
                params = self.manual_params.get("random_forest", {})
                pipeline.set_params(
                    model__n_estimators=int(params.get("n_estimators", 300)),
                    model__max_depth=params.get("max_depth", 24),
                    model__min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                )
                return pipeline, {}
            return pipeline, {
                "model__n_estimators": [300],
                "model__max_depth": [12, None],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": ["sqrt"],
            }

        if name == "MLP":
            estimator = MLPClassifier(
                activation="relu",
                solver="adam",
                batch_size=128,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=self.random_state,
            )
            pipeline = _pipeline(estimator, feature_count, self.feature_selection)
            if self.parameter_mode == "manual":
                params = self.manual_params.get("mlp", {})
                hidden = tuple(params.get("hidden_layer_sizes", (64,)))
                pipeline.set_params(
                    model__hidden_layer_sizes=hidden,
                    model__alpha=float(params.get("alpha", 0.001)),
                    model__learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                )
                return pipeline, {}
            return pipeline, {
                "model__hidden_layer_sizes": [(16,), (32,), (64,)],
                "model__alpha": [0.005, 0.01, 0.05, 0.1],
                "model__learning_rate_init": [0.001],
            }
        raise ValueError(f"Неподдерживаемая модель: {name}")

    def _fit_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_groups: np.ndarray,
    ) -> Tuple[Any, float, float, Dict[str, Any]]:
        pipeline, grid = self._model_spec(name, X_train.shape[1])
        cv = make_group_cv(y_train, train_groups, self.random_state)
        if grid:
            refit: bool | Any = True
            return_train_score = name == "MLP"
            if name == "MLP":
                def refit(cv_results: Dict[str, Any]) -> int:
                    """Prefer the least overfit MLP within a practical F1 tie."""
                    validation = np.asarray(cv_results["mean_test_score"], dtype=float)
                    training = np.asarray(cv_results["mean_train_score"], dtype=float)
                    best_validation = float(np.nanmax(validation))
                    eligible = np.flatnonzero(validation >= best_validation - 0.005)
                    gaps = np.maximum(0.0, training[eligible] - validation[eligible])
                    return int(eligible[int(np.argmin(gaps))])

            search = GridSearchCV(
                pipeline,
                grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                refit=refit,
                return_train_score=return_train_score,
            )
            search.fit(X_train, y_train, groups=train_groups)
            index = int(search.best_index_)
            cv_mean = float(search.cv_results_["mean_test_score"][index])
            cv_std = float(search.cv_results_["std_test_score"][index])
            params = {
                key.replace("model__", ""): value
                for key, value in search.best_params_.items()
            }
            oof_probabilities = _group_oof_probabilities(
                search.best_estimator_, X_train, y_train, train_groups, cv
            )
            class_biases, raw_oof_f1, threshold_oof_f1 = _tune_class_biases(
                y_train, oof_probabilities
            )
            model = ClassBiasClassifier(search.best_estimator_, class_biases)
            params["selection_metric"] = "grouped_oof_macro_f1_after_threshold_tuning"
            params["cv_splits"] = int(cv.n_splits)
            params["raw_grouped_cv_macro_f1_mean"] = cv_mean
            if return_train_score:
                train_cv_mean = float(search.cv_results_["mean_train_score"][index])
                params["raw_grouped_cv_train_macro_f1_mean"] = train_cv_mean
                params["raw_grouped_cv_overfit_gap"] = max(0.0, train_cv_mean - cv_mean)
                params["selection_rule"] = (
                    "lowest_grouped_cv_train_validation_gap_within_0.005_of_best_validation_f1"
                )
            params["raw_grouped_oof_macro_f1"] = float(raw_oof_f1)
            params["threshold_cv_macro_f1"] = float(threshold_oof_f1)
            params["threshold_cv_gain"] = float(threshold_oof_f1 - raw_oof_f1)
            params["class_biases"] = [float(value) for value in class_biases]
            return model, float(threshold_oof_f1), cv_std, params

        scores = cross_validate(
            clone(pipeline),
            X_train,
            y_train,
            groups=train_groups,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
            return_train_score=False,
        )["test_score"]
        pipeline.fit(X_train, y_train)
        oof_probabilities = _group_oof_probabilities(
            pipeline, X_train, y_train, train_groups, cv
        )
        class_biases, raw_oof_f1, threshold_oof_f1 = _tune_class_biases(
            y_train, oof_probabilities
        )
        model = ClassBiasClassifier(pipeline, class_biases)
        params = {
            "mode": "manual",
            "selection_metric": "grouped_oof_macro_f1_after_threshold_tuning",
            "cv_splits": int(cv.n_splits),
            "raw_grouped_cv_macro_f1_mean": float(np.mean(scores)),
            "raw_grouped_oof_macro_f1": float(raw_oof_f1),
            "threshold_cv_macro_f1": float(threshold_oof_f1),
            "threshold_cv_gain": float(threshold_oof_f1 - raw_oof_f1),
            "class_biases": [float(value) for value in class_biases],
        }
        return model, float(threshold_oof_f1), float(np.std(scores)), params

    def train_all(
        self,
        X: np.ndarray,
        y_text: np.ndarray,
        feature_names: List[str],
        dataset_info: Dict[str, Any],
        preprocessing_params: Dict[str, Any],
        speaker_ids: List[str] | np.ndarray | None = None,
        split_labels: List[str] | np.ndarray | None = None,
    ) -> Tuple[TrainingBundle, Dict[str, ModelEvaluation]]:
        if len(np.unique(y_text)) < 2:
            raise ValueError("Для обучения необходимо минимум два класса эмоций.")

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_text)
        labels = list(label_encoder.classes_)
        groups = _normalize_speaker_ids(speaker_ids, len(y))
        if split_labels is not None and any(str(value).strip() for value in split_labels):
            train_idx, test_idx, split_info = predefined_holdout_indices(
                y, groups, split_labels, labels=labels
            )
        else:
            train_idx, test_idx, split_info = speaker_disjoint_holdout_indices(
                y,
                groups,
                test_size=self.test_size,
                random_state=self.random_state,
                labels=labels,
            )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_groups, test_groups = groups[train_idx], groups[test_idx]

        requested = list(dict.fromkeys(self.selected_models))
        unsupported = [name for name in requested if name not in SUPPORTED_MODELS]
        if unsupported:
            raise ValueError(
                "В честном режиме поддерживаются только Logistic Regression, Random Forest и MLP. "
                f"Уберите: {', '.join(unsupported)}."
            )

        models: Dict[str, Any] = {}
        evaluations: Dict[str, ModelEvaluation] = {}
        for name in requested:
            model, cv_mean, cv_std, params = self._fit_model(name, X_train, y_train, train_groups)
            models[name] = model
            evaluations[name] = _evaluate(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                test_groups,
                labels,
                cv_mean,
                cv_std,
                params,
                self.random_state,
            )

        if not evaluations:
            raise ValueError("Не выбрана ни одна модель для обучения.")

        # Selection is based only on grouped CV over the training speakers.
        best_name = _select_deployment_model(evaluations)
        dataset_info = dict(dataset_info)
        dataset_info.update(
            {
                "split_strategy": split_info["strategy"],
                "split_info": split_info,
                "split_warnings": split_info.get("warnings", []),
                "model_selection": (
                    "grouped_oof_macro_f1_on_training_speakers_with_0.01_"
                    "practical_tie_margin_and_complexity_preference"
                ),
                "model_selection_practical_f1_margin": PRACTICAL_F1_MARGIN,
                "test_used_for_selection": False,
            }
        )
        preprocessing_params = dict(preprocessing_params)
        preprocessing_params.update(
            {
                "preprocessing_in_model": True,
                "feature_profile": "compact_v1_k96_median",
                "random_state": self.random_state,
            }
        )
        bundle = TrainingBundle(
            models=models,
            best_model_name=best_name,
            scaler=None,
            label_encoder=label_encoder,
            feature_names=feature_names,
            selector=None,
            preprocessing_params=preprocessing_params,
            metrics={name: evaluation.to_dict() for name, evaluation in evaluations.items()},
            dataset_info=dataset_info,
            ensemble_weights={},
            created_at=timestamp(),
        )
        return bundle, evaluations
