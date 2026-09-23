from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FeatureExtractor
from app.honest_models import Trainer
from app.models import save_bundle
from app.storage import ExperimentDB, TrainingReport, write_training_report


def _signature(csv_path: Path, settings: dict) -> str:
    digest = hashlib.sha256()
    with csv_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить воспроизводимый эксперимент VoiceEmotionApp на Dusha")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["Logistic Regression", "Random Forest", "MLP"],
        default=["Logistic Regression", "Random Forest", "MLP"],
    )
    parser.add_argument("--tag", default="v5_regularized_mlp")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Использовать только official train и создать внутренний speaker-disjoint holdout",
    )
    args = parser.parse_args()

    app_root = args.app_root.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir
        else app_root / "cache" / "dusha_compact_v1"
    )
    models_dir = app_root / "models"
    reports_dir = app_root / "reports"
    database_dir = app_root / "database"
    for path in [cache_dir, models_dir, reports_dir, database_dir]:
        path.mkdir(parents=True, exist_ok=True)

    settings = {
        "protocol": f"official_dusha_split_grouped_cv_{args.tag}_oof_thresholds",
        "models": list(args.models),
        "feature_profile": "compact_v1_k96_median",
        "sample_rate": 16000,
        "denoise": False,
        "trim_silence": True,
        "normalize_amplitude": True,
        "feature_selection": "auto",
        "selected_feature_count": 96,
        "missing_value_strategy": "median_inside_cv_pipeline",
        "seed": int(args.seed),
        "development_only": bool(args.development_only),
    }
    signature = _signature(args.csv, settings)
    db = ExperimentDB(database_dir / "experiments.sqlite")
    previous = db.find_by_signature(signature)
    if previous and Path(str(previous.get("model_path", ""))).is_file():
        print("Эксперимент уже выполнен.", flush=True)
        print(json.dumps(previous, ensure_ascii=False, indent=2, default=str), flush=True)
        return

    data = pd.read_csv(args.csv)
    required = {"file_path", "emotion", "speaker_id", "dataset_split"}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"В CSV отсутствуют поля: {', '.join(missing_columns)}")
    if args.development_only:
        data = data.loc[
            data["dataset_split"].astype(str).str.lower() == "train"
        ].copy()
        if data.empty:
            raise ValueError("В CSV нет строк official train для development-only запуска.")
    missing_files = [path for path in data["file_path"].astype(str) if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Не найдено WAV-файлов: {len(missing_files)}; пример: {missing_files[:3]}")
    print(
        f"Датасет: {len(data)} записей, {data['speaker_id'].nunique()} дикторов, "
        f"классы={data['emotion'].value_counts().to_dict()}",
        flush=True,
    )

    extractor = FeatureExtractor(
        cache_dir,
        sample_rate=16000,
        denoise=False,
        trim=True,
        normalize=True,
    )
    feature_started = time.perf_counter()

    def progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - feature_started
        rate = done / elapsed if elapsed else 0.0
        remaining = (total - done) / rate if rate else 0.0
        print(
            f"Признаки: {done}/{total}; прошло {elapsed / 60:.1f} мин; "
            f"осталось примерно {remaining / 60:.1f} мин",
            flush=True,
        )

    matrix = extractor.build_matrix(data, progress_callback=progress)
    if len(matrix.errors) / max(len(data), 1) > 0.05:
        raise RuntimeError("Ошибок извлечения признаков больше 5%:\n" + "\n".join(matrix.errors[:30]))
    print(f"Матрица признаков: {matrix.X.shape}; ошибок: {len(matrix.errors)}", flush=True)

    trainer = Trainer(
        test_size=0.20,
        feature_selection="auto",
        random_state=args.seed,
        selected_models=settings["models"],
        parameter_mode="auto",
    )
    training_started = time.perf_counter()
    bundle, evaluations = trainer.train_all(
        matrix.X,
        matrix.y,
        matrix.feature_names,
        {
            "dataset_path": str(args.csv.resolve()),
            "records_count": int(len(matrix.y)),
            "class_distribution": data["emotion"].value_counts().to_dict(),
            "annotation_protocol": "Dawid-Skene confidence >= 0.9",
        },
        {
            "sample_rate": 16000,
            "normalize_amplitude": True,
            "denoise": False,
            "trim_silence": True,
            "evaluation_protocol": "speaker_independent_grouped_cv",
        },
        speaker_ids=matrix.speaker_ids,
        split_labels=None if args.development_only else matrix.split_labels,
    )
    print(f"Обучение завершено за {(time.perf_counter() - training_started) / 60:.1f} мин", flush=True)

    model_path = save_bundle(bundle, models_dir)
    report = TrainingReport(
        dataset_path=str(args.csv.resolve()),
        processed_files=int(len(matrix.y)),
        class_distribution=data["emotion"].value_counts().to_dict(),
        feature_info={
            "feature_count": int(matrix.X.shape[1]),
            "feature_names_sample": matrix.feature_names[:25],
            "cache_dir": str(cache_dir),
            "split_info": bundle.dataset_info.get("split_info", {}),
            "validation_split_info": {},
            "annotation_protocol": "Dawid-Skene confidence >= 0.9",
        },
        model_results={name: value.to_dict() for name, value in evaluations.items()},
        best_model_name=bundle.best_model_name,
        warnings=matrix.errors + bundle.dataset_info.get("split_warnings", []),
        duration_sec=time.perf_counter() - feature_started,
    )
    report_path = write_training_report(report, reports_dir)
    metrics = {name: value.to_dict() for name, value in evaluations.items()}
    db.add_experiment(
        dataset_path=str(args.csv.resolve()),
        selected_models=settings["models"],
        parameters={name: value.best_params for name, value in evaluations.items()},
        metrics=metrics,
        model_path=str(model_path),
        report_path=str(report_path),
        duration_sec=report.duration_sec,
        warnings=report.warnings,
        experiment_signature=signature,
    )
    safe_tag = "".join(char if char.isalnum() or char in "-_" else "_" for char in args.tag)
    summary_path = reports_dir / f"dusha_{safe_tag}_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "settings": settings,
                "best_model": bundle.best_model_name,
                "model_path": str(model_path),
                "report_path": str(report_path),
                "split_info": bundle.dataset_info.get("split_info", {}),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Лучшая модель: {bundle.best_model_name}", flush=True)
    for name, value in evaluations.items():
        print(
            f"{name}: grouped OOF threshold F1={value.cv_mean:.4f}; "
            f"raw CV macro F1={value.best_params.get('raw_grouped_cv_macro_f1_mean', value.cv_mean):.4f}"
            f"±{value.cv_std:.4f}; "
            f"test macro F1={value.macro_f1:.4f}; balanced accuracy={value.balanced_accuracy:.4f}; "
            f"precision={value.precision_macro:.4f}; recall={value.recall_macro:.4f}; "
            f"ROC-AUC={value.roc_auc_macro:.4f}; PR-AUC={value.average_precision_macro:.4f}; "
            f"raw test macro F1={value.raw_test_macro_f1:.4f}; "
            f"OOF threshold gain={value.threshold_cv_gain:+.4f}",
            flush=True,
        )
    print(f"Модель: {model_path}", flush=True)
    print(f"Отчёт: {report_path}", flush=True)
    print(f"JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
