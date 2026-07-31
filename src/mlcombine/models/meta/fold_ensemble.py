"""FoldEnsemble — K-fold ensemble that averages fold model predictions.

Wraps any model provider with K-fold cross-validation training.
Each fold trains a separate model; ``predict()`` / ``predict_proba()``
averages all fold models.

Usage in YAML::

    model:
      - id: "base"
        provider: "catboost"
      - provider: "fold_ensemble"
        model: "base"
        params:
          n_folds: 5
          stratified: true
          group_col: "userId"
          target_encode_cols: ["cat_col"]
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from mlcombine.core.protocols import SupportedModel
from mlcombine.core.registry import registry
from mlcombine.core.schemas.blueprint import ModelBlueprint
from mlcombine.core.tensor import UnifiedTensor
from mlcombine.evaluators.cv import _target_encode

logger = logging.getLogger(__name__)


class FoldEnsemble:
    """K-fold ensemble: trains one model per fold, averages predictions.

    Args:
        blueprint: ``ModelBlueprint`` for the model to fold over.
        n_folds: Number of CV folds.
        stratified: Use ``StratifiedKFold`` / ``StratifiedGroupKFold``.
        group_col: Column for ``StratifiedGroupKFold`` (no group split if ``None``).
        target_encode_cols: Columns for OOF-safe target encoding.
        target_encode_smoothing: Smoothing factor for target encoding.
        random_state: Random seed.
    """

    def __init__(
        self,
        blueprint: ModelBlueprint,
        *,
        n_folds: int = 5,
        stratified: bool = True,
        group_col: str | None = None,
        target_encode_cols: list[str] | None = None,
        target_encode_smoothing: float = 10.0,
        random_state: int = 42,
        vote: str = "hard",
    ) -> None:
        self._blueprint = blueprint
        self._n_folds = n_folds
        self._stratified = stratified
        self._group_col = group_col
        self._target_encode_cols = list(target_encode_cols or [])
        self._target_encode_smoothing = target_encode_smoothing
        self._random_state = random_state
        self._vote = vote

        self.fold_models_: list[SupportedModel] = []
        self.oof_preds_: pd.Series | pd.DataFrame | None = None

    @property
    def is_fitted(self) -> bool:
        return len(self.fold_models_) > 0

    def fit(
        self,
        x: pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        y: pd.Series | pd.DataFrame | np.ndarray | UnifiedTensor[Any],
        **kwargs: Any,
    ) -> FoldEnsemble:
        if isinstance(x, UnifiedTensor):
            x = pd.DataFrame(x.numpy())
        elif not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))
        if isinstance(y, UnifiedTensor):
            y_arr: NDArray[Any] = y.numpy().ravel()
        else:
            y_arr = np.asarray(y).ravel()

        # ── Splitter ──
        groups = None
        if self._group_col and self._group_col in x.columns:
            groups = x[self._group_col].to_numpy()
            x = x.drop(columns=[self._group_col])
            try:
                if self._stratified:
                    from sklearn.model_selection import StratifiedGroupKFold

                    splitter = StratifiedGroupKFold(
                        n_splits=self._n_folds,
                        shuffle=True,
                        random_state=self._random_state,
                    )
                else:
                    from sklearn.model_selection import GroupKFold

                    splitter = GroupKFold(n_splits=self._n_folds)
            except ImportError:
                from sklearn.model_selection import GroupKFold

                splitter = GroupKFold(n_splits=self._n_folds)
            split_kwargs: dict[str, Any] = {"groups": groups}
        else:
            split_kwargs = {}
            if self._stratified:
                from sklearn.model_selection import StratifiedKFold

                splitter = StratifiedKFold(
                    n_splits=self._n_folds,
                    shuffle=True,
                    random_state=self._random_state,
                )
            else:
                from sklearn.model_selection import KFold

                splitter = KFold(
                    n_splits=self._n_folds,
                    shuffle=True,
                    random_state=self._random_state,
                )

        # ── K-fold training ──
        self.fold_models_ = []
        oof_preds_cols: list[NDArray[Any]] = []

        for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y_arr, **split_kwargs)):
            x_train = x.iloc[train_idx].copy()
            x_val = x.iloc[val_idx].copy()
            y_train = y_arr[train_idx]

            if self._target_encode_cols:
                x_all = pd.concat([x_train, x_val], axis=0)
                x_enc = _target_encode(
                    x_all,
                    y_arr,
                    self._target_encode_cols,
                    train_idx=np.arange(len(x_train)),
                    smoothing=self._target_encode_smoothing,
                )
                x_train = x_enc.iloc[: len(x_train)]
                x_val = x_enc.iloc[len(x_train) :]

            model = self._blueprint.build()
            model.fit(x_train, y_train, **kwargs)
            self.fold_models_.append(model)

            try:
                proba = np.asarray(model.predict_proba(x_val), dtype=np.float64)
            except RuntimeError, AttributeError:
                proba = np.asarray(model.predict(x_val), dtype=np.float64)
                proba = proba.reshape(-1, 1)
            oof_preds_cols.append(proba)

            logger.info(
                "Fold %d/%d: train=%d val=%d n_features=%d",
                fold + 1,
                self._n_folds,
                len(x_train),
                len(x_val),
                x_train.shape[1],
            )

        # ── Store OOF predictions ──
        is_multiclass = oof_preds_cols and oof_preds_cols[0].ndim > 1 and oof_preds_cols[0].shape[1] > 1
        if is_multiclass:
            n_classes = oof_preds_cols[0].shape[1]
            oof_full_mc = np.full((x.shape[0], n_classes), np.nan)
            for i, (_, val_idx) in enumerate(splitter.split(x, y_arr, **split_kwargs)):
                oof_full_mc[val_idx] = oof_preds_cols[i]
            self.oof_preds_ = pd.DataFrame(oof_full_mc, index=x.index)
        else:
            oof_full_bin = np.full(x.shape[0], np.nan)
            for i, (_, val_idx) in enumerate(splitter.split(x, y_arr, **split_kwargs)):
                oof_full_bin[val_idx] = oof_preds_cols[i].ravel()
            self.oof_preds_ = pd.Series(oof_full_bin, name="oof_preds", index=x.index)

        logger.info(
            "FoldEnsemble fitted: %d models, %d folds, oof_shape=%s",
            len(self.fold_models_),
            self._n_folds,
            self.oof_preds_.shape if hasattr(self.oof_preds_, "shape") else len(self.oof_preds_),
        )
        return self

    def _predict_fold(self, x: pd.DataFrame, fold_idx: int) -> NDArray[np.float64]:
        return np.asarray(self.fold_models_[fold_idx].predict(x), dtype=np.float64)

    def _predict_proba_fold(self, x: pd.DataFrame, fold_idx: int) -> NDArray[np.float64]:
        try:
            return np.asarray(
                self.fold_models_[fold_idx].predict_proba(x),
                dtype=np.float64,
            )
        except RuntimeError, AttributeError:
            preds = self._predict_fold(x, fold_idx)
            n_classes = 2
            if self.oof_preds_ is not None and isinstance(self.oof_preds_, pd.DataFrame):
                n_classes = self.oof_preds_.shape[1]
            one_hot = np.zeros((len(preds), n_classes))
            one_hot[np.arange(len(preds)), preds.astype(int).ravel()] = 1.0
            return one_hot

    def predict(
        self,
        x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any],
    ) -> NDArray[Any]:
        if not self.is_fitted:
            raise RuntimeError("FoldEnsemble must be fitted before prediction")
        if isinstance(x, UnifiedTensor):
            x = pd.DataFrame(x.numpy())
        elif not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        if self._vote == "soft":
            probas = self.predict_proba(x)
            return np.asarray(probas.argmax(axis=1))

        # hard — majority vote across folds
        all_preds = np.column_stack([f.predict(x).ravel() for f in self.fold_models_])
        return np.array([Counter(row).most_common(1)[0][0] for row in all_preds])

    def predict_proba(
        self,
        x: pd.DataFrame | np.ndarray | pd.Series | UnifiedTensor[Any],
    ) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("FoldEnsemble must be fitted before prediction")
        if isinstance(x, UnifiedTensor):
            x = pd.DataFrame(x.numpy())
        elif not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(np.asarray(x))

        probas = np.mean(
            [self._predict_proba_fold(x, i) for i in range(len(self.fold_models_))],
            axis=0,
        )
        return probas  # type: ignore[no-any-return]


@registry.model_provider("fold_ensemble")
def fold_ensemble_provider(model: ModelBlueprint, **params: Any) -> SupportedModel:
    """Create a ``FoldEnsemble`` wrapping a single model blueprint.

    ``params`` keys consumed by this provider:
        n_folds (int) — number of CV folds (default 5).
        stratified (bool) — use ``StratifiedKFold`` / ``StratifiedGroupKFold``.
        group_col (str) — column for group-based splitting.
        target_encode_cols (list[str]) — OOF-safe target encoding columns.
        target_encode_smoothing (float) — smoothing factor (default 10.0).
        random_state (int) — random seed (default 42).
        vote (str) — ``"hard"`` (majority, default) or ``"soft"`` (average probas).
    """
    n_folds: int = int(params.pop("n_folds", 5))
    stratified: bool = bool(params.pop("stratified", True))
    group_col: str | None = params.pop("group_col", None)
    target_encode_cols: list[str] = list(params.pop("target_encode_cols", []))
    target_encode_smoothing: float = float(params.pop("target_encode_smoothing", 10.0))
    random_state: int = int(params.pop("random_state", 42))
    vote: str = params.pop("vote", "hard")
    for key in ("task_type", "objective", "num_classes", "input_size", "backbone"):
        params.pop(key, None)
    return FoldEnsemble(
        model,
        n_folds=n_folds,
        stratified=stratified,
        group_col=group_col,
        target_encode_cols=target_encode_cols,
        target_encode_smoothing=target_encode_smoothing,
        random_state=random_state,
        vote=vote,
    )
