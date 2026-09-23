from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multimodal_components import _cache_path
from scripts.run_emotion2vec_embedding_pilot import _fit_embedding_baseline
from scripts.experiment_common import CLASSES, classification_metrics as _metrics, sample_train as _sample_train


class PairProjectionClassifier(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(192, 96),
        )
        self.classifier = nn.Linear(96, classes)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.projector(values)
        return self.classifier(torch.nn.functional.gelu(projection)), projection


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_embeddings(data: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    paths = [_cache_path(cache_dir, str(value)) for value in data["file_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} embeddings; first: {missing[0]}")
    return np.vstack([np.load(path, allow_pickle=False) for path in paths]).astype(np.float32)


def _contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor,
    speakers: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    values = torch.nn.functional.normalize(projection, dim=1)
    logits = values @ values.T / float(temperature)
    count = len(labels)
    self_mask = torch.eye(count, device=labels.device, dtype=torch.bool)
    same_class = labels[:, None].eq(labels[None, :])
    different_speaker = speakers[:, None].ne(speakers[None, :])
    positive = same_class & different_speaker & ~self_mask

    # Give the known confusing boundaries more influence in the denominator:
    # anger<->joy (high arousal) and calm<->sadness (low arousal).
    a, c, j, s = [CLASSES.index(value) for value in ("anger", "calm", "joy", "sadness")]
    hard_negative = (
        ((labels[:, None] == a) & (labels[None, :] == j))
        | ((labels[:, None] == j) & (labels[None, :] == a))
        | ((labels[:, None] == c) & (labels[None, :] == s))
        | ((labels[:, None] == s) & (labels[None, :] == c))
    )
    logits = logits + hard_negative.float() * np.log(2.0)
    logits = logits.masked_fill(self_mask, -1e9)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not bool(valid.any()):
        return projection.sum() * 0.0
    per_anchor = -(log_probabilities * positive.float()).sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[valid].mean()


def _train_epochs(
    X: np.ndarray,
    y: np.ndarray,
    speakers: np.ndarray,
    seed: int,
    pair_weight: float,
    epochs: int,
) -> PairProjectionClassifier:
    _seed_everything(seed)
    model = PairProjectionClassifier(X.shape[1], len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
        torch.from_numpy(speakers.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(128, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    for _ in range(int(epochs)):
        model.train()
        for values, labels, speaker_ids in loader:
            optimizer.zero_grad()
            logits, projection = model(values)
            loss = nn.functional.cross_entropy(logits, labels)
            if pair_weight > 0:
                loss = loss + float(pair_weight) * _contrastive_loss(
                    projection, labels, speaker_ids
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def _predict(model: PairProjectionClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X.astype(np.float32, copy=False)))
        return torch.softmax(logits, dim=1).cpu().numpy()


def _select_epoch_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    pair_weight: float,
) -> tuple[PairProjectionClassifier, StandardScaler, int, float]:
    split = next(
        StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(
            X, y, groups
        )
    )
    fit_idx, validation_idx = split
    inner_scaler = StandardScaler().fit(X[fit_idx])
    X_fit = inner_scaler.transform(X[fit_idx]).astype(np.float32)
    X_validation = inner_scaler.transform(X[validation_idx]).astype(np.float32)
    speaker_encoder = {value: index for index, value in enumerate(np.unique(groups[fit_idx]))}
    fit_speakers = np.asarray([speaker_encoder[value] for value in groups[fit_idx]], dtype=np.int64)

    _seed_everything(seed)
    model = PairProjectionClassifier(X.shape[1], len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
    dataset = TensorDataset(
        torch.from_numpy(X_fit),
        torch.from_numpy(y[fit_idx].astype(np.int64)),
        torch.from_numpy(fit_speakers),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(128, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, Any] | None = None
    best_epoch = 1
    best_score = -1.0
    stale = 0
    for epoch in range(1, 121):
        model.train()
        for values, labels, speaker_ids in loader:
            optimizer.zero_grad()
            logits, projection = model(values)
            loss = nn.functional.cross_entropy(logits, labels)
            if pair_weight > 0:
                loss = loss + float(pair_weight) * _contrastive_loss(
                    projection, labels, speaker_ids
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        probabilities = _predict(model, X_validation)
        score = float(
            f1_score(y[validation_idx], np.argmax(probabilities, axis=1), average="macro")
        )
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 14 and epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError("Pair model did not produce a valid checkpoint.")

    full_scaler = StandardScaler().fit(X)
    X_full = full_scaler.transform(X).astype(np.float32)
    full_speaker_encoder = {value: index for index, value in enumerate(np.unique(groups))}
    full_speakers = np.asarray(
        [full_speaker_encoder[value] for value in groups], dtype=np.int64
    )
    final_model = _train_epochs(
        X_full,
        y,
        full_speakers,
        seed + 1000,
        pair_weight,
        best_epoch,
    )
    return final_model, full_scaler, best_epoch, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only grouped contrastive pair pilot")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    data = _sample_train(pd.read_csv(args.csv), args.max_per_class, args.seed)
    X = _load_embeddings(data, args.embedding_cache_dir)
    groups = data["speaker_id"].astype(str).to_numpy()
    encoder = LabelEncoder().fit(CLASSES)
    y = encoder.transform(data["emotion"].astype(str))
    n = len(y)
    logistic_probabilities = np.zeros((n, len(CLASSES)), dtype=float)
    ce_probabilities = np.zeros_like(logistic_probabilities)
    pair_probabilities = np.zeros_like(logistic_probabilities)
    fold_details: list[dict[str, Any]] = []
    started = time.perf_counter()
    outer = StratifiedGroupKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, validation_idx) in enumerate(outer.split(X, y, groups)):
        logistic, selected_c, scores = _fit_embedding_baseline(
            X[train_idx], y[train_idx], groups[train_idx], args.seed + fold, 3
        )
        logistic_probabilities[validation_idx] = logistic.predict_proba(X[validation_idx])

        ce_model, ce_scaler, ce_epoch, ce_inner = _select_epoch_and_fit(
            X[train_idx], y[train_idx], groups[train_idx], args.seed + 100 + fold, 0.0
        )
        ce_probabilities[validation_idx] = _predict(
            ce_model, ce_scaler.transform(X[validation_idx]).astype(np.float32)
        )
        pair_model, pair_scaler, pair_epoch, pair_inner = _select_epoch_and_fit(
            X[train_idx], y[train_idx], groups[train_idx], args.seed + 200 + fold, 0.20
        )
        pair_probabilities[validation_idx] = _predict(
            pair_model, pair_scaler.transform(X[validation_idx]).astype(np.float32)
        )
        fold_details.append(
            {
                "fold": fold,
                "logistic_c": selected_c,
                "logistic_inner_scores": scores,
                "ce_best_epoch": ce_epoch,
                "ce_inner_macro_f1": ce_inner,
                "pair_best_epoch": pair_epoch,
                "pair_inner_macro_f1": pair_inner,
            }
        )
        print(f"Fold {fold + 1}/{args.outer_splits} complete", flush=True)

    variants = {
        "logistic": logistic_probabilities,
        "mlp_cross_entropy": ce_probabilities,
        "mlp_cross_entropy_plus_pairs": pair_probabilities,
    }
    metrics = {
        name: _metrics(y, np.argmax(probabilities, axis=1))
        for name, probabilities in variants.items()
    }
    result = {
        "protocol": "official_train_only_speaker_grouped_contrastive_pair_pilot",
        "records": int(n),
        "speakers": int(len(np.unique(groups))),
        "official_test_used": False,
        "pair_definition": {
            "positive": "same emotion, different speaker",
            "hard_negative": ["anger_vs_joy", "calm_vs_sadness"],
            "pair_loss_weight": 0.20,
        },
        "metrics": metrics,
        "pair_gain_over_same_mlp": float(
            metrics["mlp_cross_entropy_plus_pairs"]["macro_f1"]
            - metrics["mlp_cross_entropy"]["macro_f1"]
        ),
        "pair_gain_over_logistic": float(
            metrics["mlp_cross_entropy_plus_pairs"]["macro_f1"]
            - metrics["logistic"]["macro_f1"]
        ),
        "fold_details": fold_details,
        "duration_sec": float(time.perf_counter() - started),
    }
    predictions = data[["file_path", "emotion", "speaker_id"]].copy()
    for name, probabilities in variants.items():
        predictions[f"{name}_class"] = encoder.inverse_transform(
            np.argmax(probabilities, axis=1)
        )
    predictions.to_csv(args.output_dir / "contrastive_pair_oof_predictions.csv", index=False)
    (args.output_dir / "contrastive_pair_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
