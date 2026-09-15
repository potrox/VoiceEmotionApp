"""Оркестрация обучения отдельных моделей и ансамблей."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import torch
except ImportError:
    torch = None

from ..constants import FEATURE_SCHEMA_VERSION
from ..storage import timestamp
from .constants import (
    BASE_MODEL_NAMES,
    ENSEMBLE_DEFINITIONS,
    MLP_CANDIDATES,
    RANDOM_FOREST_PARAMETER_GRIDS,
    SVM_PARAMETER_GRIDS,
)
from .ensembles import ProbabilityEnsemble
from .evaluation import (
    _group_aware_split_indices,
    _normalize_speaker_ids,
    build_cv_strategy,
    evaluate_model,
)
from .neural_network import TorchMLP
from .runtime import sklearn_n_jobs
from .types import ModelEvaluation, TrainingBundle


@dataclass(frozen=True)
class PreparedTrainingData:

    label_encoder: LabelEncoder
    class_names: list[str]
    scaler: StandardScaler
    selector: SelectKBest | None
    raw_fit_features: np.ndarray
    fit_features: np.ndarray
    fit_labels: np.ndarray
    fit_speaker_ids: np.ndarray | None
    validation_features: np.ndarray
    validation_labels: np.ndarray
    training_features: np.ndarray
    training_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_split_info: dict[str, Any]
    validation_split_info: dict[str, Any]


@dataclass(frozen=True)
class TrainingPlan:

    selected_models: frozenset[str]
    selected_ensembles: tuple[str, ...]
    train_svm: bool
    train_mlp: bool
    train_random_forest: bool


@dataclass(frozen=True)
class TrainedCandidate:

    final_model: Any
    validation_model: Any
    parameters: dict[str, Any]
    cv_scores: np.ndarray | None = None


@dataclass(frozen=True)
class EnsembleTrainingResults:

    models: dict[str, ProbabilityEnsemble]
    evaluations: dict[str, ModelEvaluation]
    selection_scores: dict[str, float]
    weights_by_name: dict[str, dict[str, dict[str, float]]]


class Trainer:

    def __init__(
        self,
        test_size: float = 0.30,
        use_gpu: bool = True,
        feature_selection: str = "auto",
        random_state: int | None = None,
        hyperparameter_mode: str = "quality",
        selected_models: list[str] | None = None,
        parameter_mode: str = "auto",
        manual_params: dict[str, Any] | None = None,
    ) -> None:
        if hyperparameter_mode not in MLP_CANDIDATES:
            raise ValueError(
                "Режим подбора параметров должен быть 'quality' или 'fast'."
            )
        if parameter_mode not in {"auto", "manual"}:
            raise ValueError(
                "Режим параметров должен быть 'auto' или 'manual'."
            )
        self.test_size = test_size
        self.use_gpu = use_gpu
        self.feature_selection = feature_selection
        self.random_state = 42 if random_state is None else int(random_state)
        self.hyperparameter_mode = hyperparameter_mode
        self.selected_models = selected_models or [
            "SVM", "MLP", "Random Forest", *ENSEMBLE_DEFINITIONS.keys()
        ]
        self.parameter_mode = parameter_mode
        self.manual_params = manual_params or {}

    def _selected_feature_count(self, feature_matrix: np.ndarray) -> int | None:
        sample_count, feature_count = feature_matrix.shape
        if self.feature_selection != "auto" or feature_count <= 80 or sample_count < 20:
            return None
        return min(max(32, sample_count // 2), feature_count, 160)

    @staticmethod
    def _validate_training_inputs(
        feature_matrix: np.ndarray,
        emotion_labels: np.ndarray,
        feature_names: list[str],
        speaker_ids: list[str] | np.ndarray | None,
    ) -> None:
        if feature_matrix.ndim != 2:
            raise ValueError("Матрица признаков должна быть двумерной.")
        sample_count, feature_count = feature_matrix.shape
        if sample_count != len(emotion_labels):
            raise ValueError(
                "Число строк признаков не совпадает с числом меток эмоций."
            )
        if feature_count != len(feature_names):
            raise ValueError(
                "Число столбцов признаков не совпадает с числом их имён."
            )
        if speaker_ids is not None and len(speaker_ids) != sample_count:
            raise ValueError(
                "Число speaker_id не совпадает с числом строк признаков."
            )
        if not np.isfinite(feature_matrix).all():
            raise ValueError("Матрица признаков содержит NaN или бесконечные значения.")

    def _fit_preprocessor(
        self, training_features: np.ndarray, training_labels: np.ndarray
    ) -> tuple[StandardScaler, SelectKBest | None, np.ndarray]:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(training_features)
        selected_feature_count = self._selected_feature_count(training_features)
        if selected_feature_count is None:
            return scaler, None, scaled_features
        selector = SelectKBest(score_func=f_classif, k=selected_feature_count)
        return scaler, selector, selector.fit_transform(scaled_features, training_labels)

    @staticmethod
    def _transform_features(
        feature_matrix: np.ndarray,
        scaler: StandardScaler,
        selector: SelectKBest | None,
    ) -> np.ndarray:
        scaled_features = scaler.transform(feature_matrix)
        return selector.transform(scaled_features) if selector is not None else scaled_features

    def _build_search_pipeline(self, estimator: Any, feature_matrix: np.ndarray) -> Pipeline:
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
        selected_feature_count = self._selected_feature_count(feature_matrix)
        if selected_feature_count is not None:
            steps.append(("selector", SelectKBest(score_func=f_classif, k=selected_feature_count)))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    @staticmethod
    def _best_search_scores(parameter_search: GridSearchCV) -> np.ndarray:
        score_names = sorted(
            name for name in parameter_search.cv_results_
            if name.startswith("split") and name.endswith("_test_score")
        )
        return np.asarray([
            parameter_search.cv_results_[name][parameter_search.best_index_]
            for name in score_names
        ], dtype=float)

    def _train_svm(
        self,
        raw_search_features: np.ndarray,
        search_features: np.ndarray,
        search_labels: np.ndarray,
        search_speaker_ids: np.ndarray | None,
        final_training_features: np.ndarray,
        final_training_labels: np.ndarray,
    ) -> tuple[Any, Any, dict[str, Any], np.ndarray | None]:
        cv_strategy, cv_groups, cv_strategy_name = build_cv_strategy(
            search_labels, search_speaker_ids, self.random_state
        )
        base_parameters = {
            "probability": True,
            "class_weight": "balanced",
            "random_state": self.random_state,
        }
        if self.parameter_mode == "manual":
            requested = self.manual_params.get("svm", {})
            parameters = {
                "C": float(requested.get("C", 1.0)),
                "gamma": requested.get("gamma", "scale"),
                "kernel": requested.get("kernel", "rbf"),
            }
            cv_scores = None
            if cv_strategy is not None:
                pipeline = self._build_search_pipeline(
                    SVC(**base_parameters, **parameters), raw_search_features
                )
                cv_scores = cross_val_score(
                    pipeline, raw_search_features, search_labels, cv=cv_strategy,
                    groups=cv_groups, scoring="f1_macro", n_jobs=sklearn_n_jobs()
                )
            report_parameters = {
                **parameters, "mode": "manual", "cv_strategy": cv_strategy_name
            }
        else:
            parameter_grid = SVM_PARAMETER_GRIDS[self.hyperparameter_mode]
            if cv_strategy is not None:
                parameter_search = GridSearchCV(
                    self._build_search_pipeline(SVC(**base_parameters), raw_search_features),
                    parameter_grid, scoring="f1_macro", cv=cv_strategy,
                    n_jobs=sklearn_n_jobs()
                )
                parameter_search.fit(
                    raw_search_features, search_labels, groups=cv_groups
                )
                parameters = {
                    name.removeprefix("classifier__"): value
                    for name, value in parameter_search.best_params_.items()
                }
                cv_scores = self._best_search_scores(parameter_search)
                report_parameters = {
                    **parameters, "cv_strategy": cv_strategy_name
                }
            else:
                parameters = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
                cv_scores = None
                report_parameters = {
                    **parameters,
                    "cv_strategy": cv_strategy_name,
                    "note": "CV отключена из-за малого числа записей",
                }

        validation_model = SVC(**base_parameters, **parameters).fit(search_features, search_labels)
        final_model = SVC(**base_parameters, **parameters).fit(
            final_training_features, final_training_labels
        )
        return final_model, validation_model, report_parameters, cv_scores

    def _train_random_forest(
        self,
        raw_search_features: np.ndarray,
        search_features: np.ndarray,
        search_labels: np.ndarray,
        search_speaker_ids: np.ndarray | None,
        final_training_features: np.ndarray,
        final_training_labels: np.ndarray,
    ) -> tuple[Any, Any, dict[str, Any], np.ndarray | None]:
        cv_strategy, cv_groups, cv_strategy_name = build_cv_strategy(
            search_labels, search_speaker_ids, self.random_state
        )
        base_parameters = {
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": sklearn_n_jobs(),
        }
        if self.parameter_mode == "manual":
            requested = self.manual_params.get("random_forest", {})
            requested_depth = requested.get("max_depth", None)
            max_depth = None if requested_depth in (0, "0", "", "None", None) else int(requested_depth)
            parameters = {
                "n_estimators": int(requested.get("n_estimators", 200)),
                "max_depth": max_depth,
                "min_samples_split": int(requested.get("min_samples_split", 2)),
                "min_samples_leaf": int(requested.get("min_samples_leaf", 1)),
            }
            cv_scores = None
            if cv_strategy is not None:
                pipeline = self._build_search_pipeline(
                    RandomForestClassifier(**base_parameters, **parameters), raw_search_features
                )
                cv_scores = cross_val_score(
                    pipeline, raw_search_features, search_labels, cv=cv_strategy,
                    groups=cv_groups, scoring="f1_macro", n_jobs=sklearn_n_jobs()
                )
            report_parameters = {
                **parameters, "mode": "manual", "cv_strategy": cv_strategy_name
            }
        else:
            parameter_grid = RANDOM_FOREST_PARAMETER_GRIDS[
                self.hyperparameter_mode
            ]
            if cv_strategy is not None:
                parameter_search = GridSearchCV(
                    self._build_search_pipeline(
                        RandomForestClassifier(**base_parameters), raw_search_features
                    ),
                    parameter_grid, scoring="f1_macro", cv=cv_strategy,
                    n_jobs=sklearn_n_jobs()
                )
                parameter_search.fit(
                    raw_search_features, search_labels, groups=cv_groups
                )
                parameters = {
                    name.removeprefix("classifier__"): value
                    for name, value in parameter_search.best_params_.items()
                }
                cv_scores = self._best_search_scores(parameter_search)
                report_parameters = {
                    **parameters, "cv_strategy": cv_strategy_name
                }
            else:
                parameters = {
                    "n_estimators": 200, "max_depth": None,
                    "min_samples_split": 2, "min_samples_leaf": 1,
                }
                cv_scores = None
                report_parameters = {
                    **parameters,
                    "cv_strategy": cv_strategy_name,
                    "note": "CV отключена из-за малого числа записей",
                }

        validation_model = RandomForestClassifier(**base_parameters, **parameters).fit(
            search_features, search_labels
        )
        final_model = RandomForestClassifier(**base_parameters, **parameters).fit(
            final_training_features, final_training_labels
        )
        return final_model, validation_model, report_parameters, cv_scores

    @staticmethod
    def _parse_hidden_sizes(value: Any) -> tuple[int, ...]:
        parts = (
            [part.strip() for part in value.split(",") if part.strip()]
            if isinstance(value, str)
            else list(value)
        )
        hidden_sizes = tuple(int(part) for part in parts)
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("Размеры скрытых слоёв MLP должны быть положительными числами.")
        return hidden_sizes

    def _manual_mlp_parameters(self) -> dict[str, Any]:
        requested = self.manual_params.get("mlp", {})
        return {
            "hidden_sizes": self._parse_hidden_sizes(
                requested.get("hidden_sizes", (128, 64))
            ),
            "learning_rate": float(requested.get("lr", 1e-3)),
            "epochs": int(requested.get("epochs", 80)),
            "batch_size": int(requested.get("batch_size", 32)),
        }

    def _select_mlp_parameters(
        self,
        raw_fit_features: np.ndarray,
        fit_labels: np.ndarray,
        fit_speaker_ids: np.ndarray | None,
        class_count: int,
    ) -> tuple[dict[str, Any], float]:
        tuning_train_indices, tuning_indices, _ = _group_aware_split_indices(
            fit_labels,
            fit_speaker_ids,
            test_size=0.20,
            random_state=self.random_state,
            split_name="mlp_tuning",
        )
        tuning_scaler, tuning_selector, tuning_training_features = (
            self._fit_preprocessor(
                raw_fit_features[tuning_train_indices],
                fit_labels[tuning_train_indices],
            )
        )
        tuning_features = self._transform_features(
            raw_fit_features[tuning_indices],
            tuning_scaler,
            tuning_selector,
        )
        best_parameters: dict[str, Any] = {}
        best_tuning_score = -1.0
        for candidate_parameters in MLP_CANDIDATES[self.hyperparameter_mode]:
            candidate_model = TorchMLP(
                tuning_training_features.shape[1],
                class_count,
                use_gpu=self.use_gpu,
                random_state=self.random_state,
                **candidate_parameters,
            )
            candidate_model.fit(
                tuning_training_features, fit_labels[tuning_train_indices]
            )
            candidate_score = float(
                f1_score(
                    fit_labels[tuning_indices],
                    candidate_model.predict(tuning_features),
                    average="macro",
                    zero_division=0,
                )
            )
            if candidate_score > best_tuning_score:
                best_tuning_score = candidate_score
                best_parameters = candidate_parameters
        if not best_parameters:
            raise RuntimeError("Не удалось обучить ни одного кандидата MLP.")
        return best_parameters, best_tuning_score

    def _train_mlp(
        self,
        raw_fit_features: np.ndarray,
        fit_features: np.ndarray,
        fit_labels: np.ndarray,
        fit_speaker_ids: np.ndarray | None,
        final_training_features: np.ndarray,
        final_training_labels: np.ndarray,
        class_count: int,
    ) -> tuple[TorchMLP, TorchMLP, dict[str, Any]]:
        if torch is None:
            raise RuntimeError("PyTorch не установлен.")

        if self.parameter_mode == "manual":
            best_parameters = self._manual_mlp_parameters()
            tuning_score = None
            candidate_count = 1
        else:
            best_parameters, tuning_score = self._select_mlp_parameters(
                raw_fit_features,
                fit_labels,
                fit_speaker_ids,
                class_count,
            )
            candidate_count = len(MLP_CANDIDATES[self.hyperparameter_mode])

        validation_model = TorchMLP(
            fit_features.shape[1],
            class_count,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            **best_parameters,
        )
        validation_model.fit(fit_features, fit_labels)
        final_model = TorchMLP(
            final_training_features.shape[1],
            class_count,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
            **best_parameters,
        )
        final_model.fit(final_training_features, final_training_labels)
        report_parameters = {
            **best_parameters,
            "device": str(validation_model.device),
            "candidate_count": candidate_count,
            "search_mode": (
                "manual"
                if self.parameter_mode == "manual"
                else self.hyperparameter_mode
            ),
        }
        if tuning_score is not None:
            report_parameters["tuning_macro_f1"] = float(tuning_score)
        return final_model, validation_model, report_parameters

    def _ensemble_weights(
        self,
        class_names: list[str],
        validation_models: dict[str, Any],
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> dict[int, dict[str, float]]:
        predictions = {
            name: model.predict(validation_features)
            for name, model in validation_models.items()
        }
        weights: dict[int, dict[str, float]] = {}
        smoothing = 1e-6
        for class_index, _class_name in enumerate(class_names):
            expected_class = (validation_labels == class_index).astype(int)
            class_scores = {
                model_name: float(f1_score(
                    expected_class, (predicted_classes == class_index).astype(int),
                    zero_division=0
                ))
                for model_name, predicted_classes in predictions.items()
            }
            total = sum(class_scores.values()) + smoothing * max(len(class_scores), 1)
            weights[class_index] = {
                model_name: float((score + smoothing) / total)
                for model_name, score in class_scores.items()
            }
        return weights

    @staticmethod
    def _weights_for_report(
        class_names: list[str], weights: dict[int, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        return {
            class_names[class_index]: {
                model_name: float(value)
                for model_name, value in model_weights.items()
            }
            for class_index, model_weights in weights.items()
        }

    def _prepare_training_data(
        self,
        feature_matrix: np.ndarray,
        emotion_labels: np.ndarray,
        speaker_ids: list[str] | np.ndarray | None,
    ) -> PreparedTrainingData:
        if len(np.unique(emotion_labels)) < 2:
            raise ValueError("Для обучения необходимо минимум два класса эмоций.")

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(emotion_labels)
        class_names = list(label_encoder.classes_)
        normalized_speaker_ids = _normalize_speaker_ids(speaker_ids, len(encoded_labels))

        training_indices, test_indices, test_split_info = _group_aware_split_indices(
            encoded_labels,
            normalized_speaker_ids,
            test_size=self.test_size,
            random_state=self.random_state,
            class_names=class_names,
            split_name="test",
        )
        raw_training_features = feature_matrix[training_indices]
        raw_test_features = feature_matrix[test_indices]
        training_labels = encoded_labels[training_indices]
        test_labels = encoded_labels[test_indices]
        training_speaker_ids = (
            normalized_speaker_ids[training_indices]
            if normalized_speaker_ids is not None
            else None
        )

        fit_indices, validation_indices, validation_split_info = _group_aware_split_indices(
            training_labels,
            training_speaker_ids,
            test_size=0.20,
            random_state=self.random_state,
            class_names=class_names,
            split_name="validation",
        )
        raw_fit_features = raw_training_features[fit_indices]
        raw_validation_features = raw_training_features[validation_indices]
        fit_labels = training_labels[fit_indices]
        validation_labels = training_labels[validation_indices]
        fit_speaker_ids = (
            training_speaker_ids[fit_indices]
            if training_speaker_ids is not None
            else None
        )

        scaler, selector, training_features = self._fit_preprocessor(
            raw_training_features, training_labels
        )
        test_features = self._transform_features(raw_test_features, scaler, selector)
        search_scaler, search_selector, fit_features = self._fit_preprocessor(
            raw_fit_features, fit_labels
        )
        validation_features = self._transform_features(
            raw_validation_features, search_scaler, search_selector
        )

        return PreparedTrainingData(
            label_encoder=label_encoder,
            class_names=class_names,
            scaler=scaler,
            selector=selector,
            raw_fit_features=raw_fit_features,
            fit_features=fit_features,
            fit_labels=fit_labels,
            fit_speaker_ids=fit_speaker_ids,
            validation_features=validation_features,
            validation_labels=validation_labels,
            training_features=training_features,
            training_labels=training_labels,
            test_features=test_features,
            test_labels=test_labels,
            test_split_info=test_split_info,
            validation_split_info=validation_split_info,
        )

    def _build_training_plan(self) -> TrainingPlan:
        selected_models = frozenset(self.selected_models)
        if not selected_models:
            raise ValueError("Не выбрана ни одна модель для обучения.")
        supported_models = BASE_MODEL_NAMES | ENSEMBLE_DEFINITIONS.keys()
        unknown_models = sorted(selected_models - supported_models)
        if unknown_models:
            raise ValueError(
                "Неизвестные модели в плане обучения: "
                + ", ".join(unknown_models)
            )
        selected_ensembles = tuple(
            name for name in ENSEMBLE_DEFINITIONS if name in selected_models
        )
        required_base_models = {
            model_name
            for ensemble_name in selected_ensembles
            for model_name in ENSEMBLE_DEFINITIONS[ensemble_name]
        }
        return TrainingPlan(
            selected_models=selected_models,
            selected_ensembles=selected_ensembles,
            train_svm="SVM" in selected_models or "SVM" in required_base_models,
            train_mlp="MLP" in selected_models or "MLP" in required_base_models,
            train_random_forest=(
                "Random Forest" in selected_models
                or "Random Forest" in required_base_models
            ),
        )

    @staticmethod
    def _evaluate_candidate(
        name: str,
        final_model: Any,
        validation_model: Any,
        parameters: dict[str, Any],
        cv_scores: np.ndarray | None,
        data: PreparedTrainingData,
    ) -> tuple[ModelEvaluation, float]:
        selection_score = float(
            f1_score(
                data.validation_labels,
                validation_model.predict(data.validation_features),
                average="macro",
                zero_division=0,
            )
        )
        reported_parameters = {
            **parameters,
            "selection_macro_f1": selection_score,
        }
        evaluation = evaluate_model(
            name,
            final_model,
            data.training_features,
            data.training_labels,
            data.test_features,
            data.test_labels,
            data.class_names,
            cv_scores,
            reported_parameters,
        )
        return evaluation, selection_score

    @staticmethod
    def _add_split_metadata(
        dataset_info: dict[str, Any], data: PreparedTrainingData, random_state: int
    ) -> None:
        dataset_info["split_strategy"] = data.test_split_info.get("strategy")
        dataset_info["split_info"] = data.test_split_info
        dataset_info["validation_split_info"] = data.validation_split_info
        dataset_info["split_warnings"] = (
            data.test_split_info.get("warnings", [])
            + data.validation_split_info.get("warnings", [])
        )
        dataset_info["random_state"] = random_state

    def _train_base_candidates(
        self, plan: TrainingPlan, data: PreparedTrainingData
    ) -> dict[str, TrainedCandidate]:
        candidates: dict[str, TrainedCandidate] = {}
        if plan.train_svm:
            candidates["SVM"] = TrainedCandidate(
                *self._train_svm(
                    data.raw_fit_features,
                    data.fit_features,
                    data.fit_labels,
                    data.fit_speaker_ids,
                    data.training_features,
                    data.training_labels,
                )
            )
        if plan.train_random_forest:
            candidates["Random Forest"] = TrainedCandidate(
                *self._train_random_forest(
                    data.raw_fit_features,
                    data.fit_features,
                    data.fit_labels,
                    data.fit_speaker_ids,
                    data.training_features,
                    data.training_labels,
                )
            )
        if plan.train_mlp:
            final_model, validation_model, parameters = self._train_mlp(
                data.raw_fit_features,
                data.fit_features,
                data.fit_labels,
                data.fit_speaker_ids,
                data.training_features,
                data.training_labels,
                len(data.class_names),
            )
            candidates["MLP"] = TrainedCandidate(
                final_model, validation_model, parameters
            )
        return candidates

    def _train_ensembles(
        self,
        ensemble_names: tuple[str, ...],
        data: PreparedTrainingData,
        final_models: dict[str, Any],
        validation_models: dict[str, Any],
    ) -> EnsembleTrainingResults:
        ensemble_models: dict[str, ProbabilityEnsemble] = {}
        evaluations: dict[str, ModelEvaluation] = {}
        selection_scores: dict[str, float] = {}
        weights_by_name: dict[str, dict[str, dict[str, float]]] = {}
        for ensemble_name in ensemble_names:
            base_model_names = ENSEMBLE_DEFINITIONS[ensemble_name]
            missing_models = [
                name for name in base_model_names if name not in final_models
            ]
            if missing_models:
                raise RuntimeError(
                    f"Для объединённой модели {ensemble_name} не обучены "
                    "базовые модели: " + ", ".join(missing_models)
                )
            weights = self._ensemble_weights(
                data.class_names,
                {name: validation_models[name] for name in base_model_names},
                data.validation_features,
                data.validation_labels,
            )
            final_ensemble = ProbabilityEnsemble(
                {name: final_models[name] for name in base_model_names}, weights
            )
            validation_ensemble = ProbabilityEnsemble(
                {name: validation_models[name] for name in base_model_names},
                weights,
            )
            weights_report = self._weights_for_report(data.class_names, weights)
            evaluation, selection_score = self._evaluate_candidate(
                ensemble_name,
                final_ensemble,
                validation_ensemble,
                {
                    "base_models": list(base_model_names),
                    "weights_by_class": weights_report,
                },
                None,
                data,
            )
            ensemble_models[ensemble_name] = final_ensemble
            evaluations[ensemble_name] = evaluation
            selection_scores[ensemble_name] = selection_score
            weights_by_name[ensemble_name] = weights_report
        return EnsembleTrainingResults(
            ensemble_models, evaluations, selection_scores, weights_by_name
        )

    def train_all(
        self,
        feature_matrix: np.ndarray,
        emotion_labels: np.ndarray,
        feature_names: list[str],
        dataset_info: dict[str, Any],
        preprocessing_params: dict[str, Any],
        speaker_ids: list[str] | np.ndarray | None = None,
    ) -> tuple[TrainingBundle, dict[str, ModelEvaluation]]:
        self._validate_training_inputs(
            feature_matrix, emotion_labels, feature_names, speaker_ids
        )
        data = self._prepare_training_data(feature_matrix, emotion_labels, speaker_ids)
        plan = self._build_training_plan()
        dataset_metadata = dict(dataset_info)
        self._add_split_metadata(dataset_metadata, data, self.random_state)

        base_candidates = self._train_base_candidates(plan, data)
        models = {
            name: candidate.final_model
            for name, candidate in base_candidates.items()
        }
        validation_models = {
            name: candidate.validation_model
            for name, candidate in base_candidates.items()
        }
        evaluations: dict[str, ModelEvaluation] = {}
        selection_scores: dict[str, float] = {}
        for name, candidate in base_candidates.items():
            if name not in plan.selected_models:
                continue
            evaluation, selection_score = self._evaluate_candidate(
                name,
                candidate.final_model,
                candidate.validation_model,
                candidate.parameters,
                candidate.cv_scores,
                data,
            )
            evaluations[name] = evaluation
            selection_scores[name] = selection_score

        ensemble_results = self._train_ensembles(
            plan.selected_ensembles,
            data,
            models,
            validation_models,
        )
        models.update(ensemble_results.models)
        evaluations.update(ensemble_results.evaluations)
        selection_scores.update(ensemble_results.selection_scores)

        if not evaluations:
            raise ValueError("Выбранная комбинация моделей не дала результатов для оценки.")

        best_model_name = max(selection_scores, key=selection_scores.get)
        dataset_metadata["model_selection_metric"] = "validation_macro_f1"
        dataset_metadata["model_selection_scores"] = selection_scores
        bundle = TrainingBundle(
            models=models,
            best_model_name=best_model_name,
            scaler=data.scaler,
            label_encoder=data.label_encoder,
            feature_names=feature_names,
            selector=data.selector,
            preprocessing_params={
                **preprocessing_params,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            },
            metrics={name: result.to_dict() for name, result in evaluations.items()},
            dataset_info=dataset_metadata,
            ensemble_weights=ensemble_results.weights_by_name,
            created_at=timestamp(),
        )
        return bundle, evaluations
