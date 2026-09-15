"""Полносвязная нейронная сеть для классификации эмоций."""

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .probability import ProbabilityPredictionMixin


class TorchMLP(ProbabilityPredictionMixin):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 32,
        use_gpu: bool = True,
        random_state: int = 42,
        **legacy_parameters: Any,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch не установлен. Установите torch из requirements.txt.")
        if "lr" in legacy_parameters:
            learning_rate = float(legacy_parameters.pop("lr"))
        if legacy_parameters:
            unexpected = ", ".join(sorted(legacy_parameters))
            raise TypeError(f"Неизвестные параметры TorchMLP: {unexpected}")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_sizes = tuple(hidden_sizes)
        self.learning_rate = float(learning_rate)
        self.lr = self.learning_rate
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.use_gpu = bool(use_gpu)
        self.random_state = int(random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = self._build().to(self.device)

    def _build(self) -> Any:
        layers: list[Any] = []
        input_size = self.input_dim
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.25))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, self.num_classes))
        return nn.Sequential(*layers)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TorchMLP":
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(feature_tensor, label_tensor)
        shuffle_generator = torch.Generator().manual_seed(self.random_state)
        data_loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=shuffle_generator,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.tensor(
                features, dtype=torch.float32
            ).to(self.device)
            logits = self.model(feature_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities
