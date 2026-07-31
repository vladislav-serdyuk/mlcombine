"""Hybrid (multi-modal) provider — late-fusion PyTorch network for image + text."""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm

from mlcombine.core.tensor import UnifiedTensor
from mlcombine.core.enums import ModelObjective, TaskType
from mlcombine.core.registry import registry
from mlcombine.core.protocols import SupportedModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True

    class HybridModel(nn.Module):
        """Late-fusion network for image + text multi-modal data."""

        def __init__(self, img_size: int, txt_size: int, out_dim: int, fusion_size: int = 128) -> None:
            super().__init__()
            self.image_branch = nn.Sequential(
                nn.Linear(img_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
            )
            self.text_branch = nn.Sequential(
                nn.Linear(txt_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
            )
            self.fusion = nn.Sequential(
                nn.Linear(256, fusion_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(fusion_size, out_dim),
            )

        def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = self.fusion(torch.cat([self.image_branch(image_features), self.text_branch(text_features)], dim=1))
            return out

    class HybridModelWrapper:
        """Wrapper making HybridModel conform to MLModelProtocol."""

        def __init__(
            self,
            img_size: int,
            txt_size: int,
            out_dim: int,
            fusion_size: int = 128,
            image_columns: list[str] | None = None,
            text_columns: list[str] | None = None,
            task_type: TaskType = TaskType.REGRESSION,
        ) -> None:
            self.model = HybridModel(img_size, txt_size, out_dim, fusion_size)
            self.img_size = img_size
            self.txt_size = txt_size
            self.out_dim = out_dim
            self.image_columns = image_columns
            self.text_columns = text_columns
            self.task_type = task_type
            self.is_fitted = False

        def _split_input(self, x: pd.DataFrame | np.ndarray | UnifiedTensor[Any]) -> tuple[torch.Tensor, torch.Tensor]:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            if isinstance(x, pd.DataFrame):
                if self.image_columns is not None:
                    img = x[self.image_columns].to_numpy()
                else:
                    img = x.iloc[:, : self.img_size].to_numpy()
                if self.text_columns is not None:
                    txt = x[self.text_columns].to_numpy()
                else:
                    txt = x.iloc[:, self.img_size : self.img_size + self.txt_size].to_numpy()
            else:
                img = x[:, : self.img_size]
                txt = x[:, self.img_size : self.img_size + self.txt_size]
            return torch.tensor(img).float(), torch.tensor(txt).float()

        def fit(
            self,
            x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
            **kwargs: Any,
        ) -> Self:
            epochs = int(kwargs.get("epochs", 10))
            lr = float(kwargs.get("lr", 0.001))
            batch_size = int(kwargs.get("batch_size", 32))

            img_tensor, txt_tensor = self._split_input(x)

            if isinstance(y, UnifiedTensor):
                y_arr = y.numpy()
            elif isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy()
            else:
                y_arr = y
            y_tensor = torch.tensor(y_arr).float()
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)

            criterion: nn.Module
            if self.task_type == TaskType.REGRESSION:
                criterion = nn.MSELoss()
            elif self.task_type == TaskType.CLASSIFICATION and self.out_dim > 1:
                criterion = nn.CrossEntropyLoss()
                y_tensor = y_tensor.squeeze(1).long()
            else:
                criterion = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            dataset = TensorDataset(img_tensor, txt_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model.train()
            for epoch in range(epochs):
                loader_tqdm = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
                for batch_img, batch_txt, batch_y in loader_tqdm:
                    optimizer.zero_grad()
                    output = self.model(batch_img, batch_txt)

                    if self.task_type == TaskType.CLASSIFICATION and self.out_dim > 1:
                        output = output.squeeze(1)

                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()

            self.is_fitted = True
            logger.info("Hybrid fitted: %d epochs, final loss=%.4f", epochs, loss.item())
            return self

        def _predict_tensor(self, x: pd.DataFrame | np.ndarray | UnifiedTensor[Any]) -> torch.Tensor:
            if isinstance(x, UnifiedTensor):
                x = x.numpy()
            img_tensor, txt_tensor = self._split_input(x)
            self.model.eval()
            with torch.no_grad():
                out: torch.Tensor = self.model(img_tensor, txt_tensor)
                return out

        def predict(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[Any]:
            if isinstance(x, pd.Series):
                x = x.to_frame().T
            output = self._predict_tensor(x)

            if self.task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
                if self.out_dim > 1:
                    return output.argmax(dim=1, keepdim=True).numpy()
                return (torch.sigmoid(output) >= 0.5).numpy().astype(int)
            return output.numpy()

        def predict_proba(self, x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any]) -> NDArray[np.float64]:
            if isinstance(x, pd.Series):
                x = x.to_frame().T
            output = self._predict_tensor(x)

            if self.out_dim > 1:
                return torch.softmax(output, dim=1).numpy()
            sigmoid_output = torch.sigmoid(output)
            return torch.cat([1 - sigmoid_output, sigmoid_output], dim=1).numpy()

except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@registry.model_provider("hybrid", package="torch", module="mlcombine.models.providers.hybrid")
def hybrid_provider(
    backbone: str = "hybrid",
    task_type: TaskType = TaskType.REGRESSION,
    objective: ModelObjective = ModelObjective.RMSE,
    num_classes: int | None = None,
    input_size: int | None = None,
    **params: Any,
) -> SupportedModel:
    """Create a hybrid PyTorch model for multi-modal data."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Install with: uv add torch")
        raise ImportError("PyTorch is required for HybridModelProvider")

    image_columns = params.pop("image_columns", None)
    text_columns = params.pop("text_columns", None)
    img_size = int(params.get("image_feature_size", 512))
    txt_size = int(params.get("text_feature_size", 768))
    out_dim = num_classes or 1
    if task_type in (TaskType.CLASSIFICATION, TaskType.MULTITASK):
        out_dim = num_classes or 2

    return HybridModelWrapper(
        img_size,
        txt_size,
        out_dim,
        image_columns=image_columns,
        text_columns=text_columns,
        task_type=task_type,
    )
