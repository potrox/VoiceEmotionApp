"""Проверки разбиения данных, обучения и сохранения моделей."""

from __future__ import annotations

import numpy as np
import joblib

from app.ml.evaluation import _group_aware_split_indices, build_cv_strategy
from app.ml.persistence import load_bundle, save_bundle, transform_features
from app.ml.training import Trainer


def test_group_split_keeps_speakers_in_one_partition() -> None:
    emotion_labels = np.tile([0, 1], 16)
    speaker_ids = np.repeat([f"speaker_{index}" for index in range(8)], 4)

    training_indices, test_indices, split_info = _group_aware_split_indices(
        emotion_labels,
        speaker_ids,
        test_size=0.25,
        random_state=42,
        class_names=["calm", "joy"],
    )

    training_speakers = set(speaker_ids[training_indices])
    test_speakers = set(speaker_ids[test_indices])
    assert training_speakers.isdisjoint(test_speakers)
    assert split_info["strategy"] == "speaker_disjoint_group_shuffle"
    assert split_info["speaker_overlap_count"] == 0


def test_cross_validation_keeps_speakers_in_one_fold() -> None:
    emotion_labels = np.tile([0, 1], 16)
    speaker_ids = np.repeat([f"speaker_{index}" for index in range(8)], 4)
    cv_strategy, cv_groups, strategy_name = build_cv_strategy(
        emotion_labels, speaker_ids, random_state=42
    )

    assert cv_groups is not None
    assert strategy_name.startswith("stratified_group_")
    for training_indices, validation_indices in cv_strategy.split(
        np.zeros((len(emotion_labels), 1)), emotion_labels, cv_groups
    ):
        assert set(speaker_ids[training_indices]).isdisjoint(
            set(speaker_ids[validation_indices])
        )


def test_manual_svm_pipeline_trains_with_isolated_test_speakers(tmp_path) -> None:
    random_generator = np.random.default_rng(42)
    feature_matrix = random_generator.normal(size=(48, 12))
    emotion_labels = np.tile(np.asarray(["calm", "joy"]), 24)
    feature_matrix[emotion_labels == "joy", :3] += 1.4
    speaker_ids = np.repeat([f"speaker_{index}" for index in range(12)], 4)
    trainer = Trainer(
        test_size=0.25,
        use_gpu=False,
        feature_selection="none",
        selected_models=["SVM"],
        parameter_mode="manual",
        manual_params={"svm": {"C": 1, "kernel": "linear", "gamma": "scale"}},
        random_state=42,
    )

    bundle, evaluations = trainer.train_all(
        feature_matrix,
        emotion_labels,
        [f"feature_{index}" for index in range(feature_matrix.shape[1])],
        {},
        {},
        speaker_ids,
    )

    assert bundle.best_model_name == "SVM"
    assert evaluations["SVM"].macro_f1 >= 0.0
    assert bundle.dataset_info["split_info"]["speaker_overlap_count"] == 0
    assert bundle.dataset_info["validation_split_info"]["speaker_overlap_count"] == 0
    assert bundle.dataset_info["model_selection_metric"] == "validation_macro_f1"
    assert bundle.dataset_info["model_selection_scores"]["SVM"] == (
        evaluations["SVM"].best_params["selection_macro_f1"]
    )
    assert evaluations["SVM"].best_params["cv_strategy"].startswith(
        "stratified_group_"
    )
    model_path = save_bundle(bundle, tmp_path)
    loaded_bundle = load_bundle(model_path)
    predictions = loaded_bundle["models"]["SVM"].predict(
        transform_features(loaded_bundle, feature_matrix[:2])
    )
    assert predictions.shape == (2,)


def test_loading_rejects_an_outdated_feature_schema(tmp_path) -> None:
    model_path = tmp_path / "outdated.pkl"
    joblib.dump({"preprocessing_params": {"feature_schema_version": 2}}, model_path)

    try:
        load_bundle(model_path)
    except ValueError as exc:
        assert "устаревшей схемой признаков" in str(exc)
    else:
        raise AssertionError("Устаревшая модель должна быть отклонена")


def test_ensemble_trains_required_bases_but_reports_only_selected_model() -> None:
    random_generator = np.random.default_rng(42)
    feature_matrix = random_generator.normal(size=(64, 12)).astype(np.float32)
    emotion_labels = np.tile(np.asarray(["calm", "joy"]), 32)
    feature_matrix[emotion_labels == "joy", :3] += 1.4
    speaker_ids = np.repeat([f"speaker_{index}" for index in range(16)], 4)
    ensemble_name = "Ансамбль SVM + MLP"
    trainer = Trainer(
        test_size=0.25,
        use_gpu=False,
        feature_selection="none",
        selected_models=[ensemble_name],
        parameter_mode="manual",
        manual_params={
            "svm": {"C": 1, "kernel": "linear", "gamma": "scale"},
            "mlp": {
                "hidden_sizes": (16,),
                "lr": 1e-3,
                "epochs": 2,
                "batch_size": 16,
            },
        },
        random_state=42,
    )

    bundle, evaluations = trainer.train_all(
        feature_matrix,
        emotion_labels,
        [f"feature_{index}" for index in range(feature_matrix.shape[1])],
        {},
        {},
        speaker_ids,
    )

    assert set(bundle.models) == {"SVM", "MLP", ensemble_name}
    assert set(evaluations) == {ensemble_name}
    assert bundle.best_model_name == ensemble_name
