"""Проверки работы с локальной базой экспериментов."""

from __future__ import annotations

from app.storage import ExperimentDB


def test_database_connections_close_after_each_operation(tmp_path) -> None:
    database_path = tmp_path / "experiments.sqlite"
    database = ExperimentDB(database_path)
    database.add_experiment(
        dataset_path="dataset.csv",
        selected_models=["SVM"],
        parameters={},
        metrics={},
        best_model_name="SVM",
        model_path="model.pkl",
        report_path="report.txt",
        duration_sec=1.0,
        experiment_signature="signature",
    )

    stored_experiment = database.find_by_signature("signature")
    assert stored_experiment["id"] == 1
    assert stored_experiment["best_model_name"] == "SVM"
    database_path.unlink()
    assert database_path.exists() is False
