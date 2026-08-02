"""Preprocessing steps — ImputeStep and EncodeScaleStep as proper BaseStep pipeline entities.

Side Effects (both steps):
    - Mutate context.data.train_df and context.data.test_df in-place during run().
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    BaseStep,
    ConfigurationError,
    EncodeStrategy,
    FeatureMap,
    FeatureType,
    ImputeStrategy,
    MLCombineConfig,
    PipelineContext,
    ScaleStrategy,
)

logger = logging.getLogger(__name__)


@registry.step("ImputeStep")
class ImputeStep(BaseStep[PipelineContext]):
    """Impute missing numeric values using a fitted SimpleImputer.

    Numeric columns are selected from ``context.data.detected_types`` by
    ``FeatureType.NUMBER`` mask.  Scikit-learn's ``.transform()`` output is
    wrapped back into a DataFrame with the original row index to prevent
    index misalignment (NaN contamination).

    Side Effects:
        - Overwrites ``context.data.train_df`` and ``context.data.test_df``
          with imputed copies.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self.impute_strategy = cfg.handling.numbers.impute
        self.column_handling = cfg.handling.columns
        self.target_col = cfg.data.target_col if isinstance(cfg.data.target_col, str) else None
        self.treatment_col = cfg.data.treatment_col
        self.id_col = cfg.data.id_col
        self._drop_columns: list[str] = cfg.data.drop_columns or []
        self.input_features_: list[str] = []
        self.imputers_: dict[str, SimpleImputer] = {}
        self._predict_mode = predict

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        if cfg.handling.numbers.impute != ImputeStrategy.NONE:
            return True
        return any(col.impute is not None for col in cfg.handling.columns.values())

    def _fit(self, x: pd.DataFrame, detected_types: FeatureMap | None = None) -> None:
        """Fit imputer on numeric columns declared via detected_types.

        Args:
            x: Training DataFrame.
            detected_types: Feature map from pipeline context; columns with
                FeatureType.NUMBER are selected for imputation.

        Side Effects:
            - Sets ``self.input_features_`` and ``self.imputers_``.

        """
        if detected_types:
            self.input_features_ = [col for col, ft in detected_types.items() if ft == FeatureType.NUMBER]
            self.input_features_ = [c for c in self.input_features_ if c in x.columns]
        else:
            self.input_features_ = x.select_dtypes(include=[np.number]).columns.tolist()
        if self.treatment_col and self.treatment_col in self.input_features_:
            self.input_features_.remove(self.treatment_col)
        if self.target_col and self.target_col in self.input_features_:
            self.input_features_.remove(self.target_col)
        if self.id_col and self.id_col in self.input_features_:
            self.input_features_.remove(self.id_col)
        if self._drop_columns:
            self.input_features_ = [c for c in self.input_features_ if c not in self._drop_columns]
        for col in self.input_features_:
            override = self.column_handling.get(col)
            strategy = override.impute if override is not None and override.impute is not None else self.impute_strategy
            if strategy == ImputeStrategy.NONE:
                continue
            fill_value = None
            if strategy == ImputeStrategy.CONSTANT and override is not None:
                fill_value = override.fill_value
            imputer = SimpleImputer(strategy=strategy.value, fill_value=fill_value)
            imputer.fit(x[[col]])
            self.imputers_[col] = imputer
        self.input_features_ = list(self.imputers_.keys())

    def _transform_features(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation, restoring original index on output.

        Args:
            x: DataFrame to transform.

        Returns:
            Imputed DataFrame with the same index as *x*.

        Raises:
            ConfigurationError: If columns seen at fit are missing.

        """
        if not self.input_features_ or not self.imputers_:
            return x
        missing = [c for c in self.input_features_ if c not in x.columns]
        if missing:
            raise ConfigurationError(f"Columns seen at fit time are missing in transform: {missing}")
        X_copy = x.copy()
        for col, imputer in self.imputers_.items():
            imputed = pd.Series(imputer.transform(x[[col]]).flatten(), index=x.index)
            X_copy[col] = imputed
        return X_copy

    def run(self, context: PipelineContext) -> PipelineContext:
        """Fit on train, transform train + test, preserving indexes."""
        detected = context.data.detected_types

        if self._predict_mode:
            if context.artifacts.imputers:
                self.imputers_ = context.artifacts.imputers
                self.input_features_ = list(context.artifacts.imputers.keys())
            if not self.imputers_:
                logger.warning("No imputers in artifacts — skipping imputation")
                return context

        if context.data.train_df is not None:
            if not self._predict_mode:
                self._fit(context.data.train_df, detected)
                context.artifacts.imputers = self.imputers_
                context.artifacts.imputer_features = self.input_features_
                logger.info("Imputed %d numeric columns (strategy=%s)", len(self.input_features_), self.impute_strategy.value)
            context.data.train_df = self._transform_features(context.data.train_df)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df)
        if context.data.holdout_df is not None:
            context.data.holdout_df = self._transform_features(context.data.holdout_df)
        return context


@registry.step("EncodeScaleStep")
class EncodeScaleStep(BaseStep[PipelineContext]):
    """Encode categorical features and scale numeric features.

    ``EncodeStrategy.ORDINAL`` and ``EncodeStrategy.ONEHOT`` are supported
    for in-place encoding. ``EncodeStrategy.TARGET`` is **blocked** with
    ``NotImplementedError`` because it requires OOF isolation. Use
    ``_target_encode()`` (via the ``fold_ensemble`` meta-provider with
    ``target_encode_cols``) for OOF-safe target encoding instead.

    All transformation outputs are wrapped in ``pd.Series`` / ``pd.DataFrame``
    with the original row index to prevent index misalignment.

    Side Effects:
        - Overwrites ``context.data.train_df`` and ``context.data.test_df``
          with transformed copies.
    """

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self.encode_strategy = cfg.handling.categories.encode
        self.scale_strategy = cfg.handling.numbers.scale
        self.smoothing = cfg.handling.categories.smoothing
        self.column_handling = cfg.handling.columns
        self.target_col = cfg.data.target_col if isinstance(cfg.data.target_col, str) else None
        self.treatment_col = cfg.data.treatment_col
        self.id_col = cfg.data.id_col
        self._drop_columns: list[str] = cfg.data.drop_columns or []

        self.encoders_: dict[str, OrdinalEncoder | OneHotEncoder] = {}
        self.scalers_: dict[str, StandardScaler | RobustScaler | MinMaxScaler] = {}
        self.input_categorical_cols_: list[str] = []
        self.input_numeric_cols_: list[str] = []
        self._text_cols: list[str] = []
        self._predict_mode = predict

    def _fit(self, X: pd.DataFrame, y: pd.Series | pd.DataFrame | None = None) -> None:
        """Fit encoders and scaler on training data.

        Args:
            X: Training DataFrame.
            y: Target values (required only for TARGET encoding — currently blocked).

        Side Effects:
            - Populates ``self.input_categorical_cols_``, ``self.input_numeric_cols_``,
              ``self.encoders_``, ``self.scaler_``.

        """
        self.input_categorical_cols_ = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        self.input_numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col and self.target_col in self.input_numeric_cols_:
            self.input_numeric_cols_.remove(self.target_col)
        if self.treatment_col:
            if self.treatment_col in self.input_categorical_cols_:
                self.input_categorical_cols_.remove(self.treatment_col)
            if self.treatment_col in self.input_numeric_cols_:
                self.input_numeric_cols_.remove(self.treatment_col)
        if self.id_col:
            if self.id_col in self.input_categorical_cols_:
                self.input_categorical_cols_.remove(self.id_col)
            if self.id_col in self.input_numeric_cols_:
                self.input_numeric_cols_.remove(self.id_col)
        if self._drop_columns:
            self.input_categorical_cols_ = [c for c in self.input_categorical_cols_ if c not in self._drop_columns]
            self.input_numeric_cols_ = [c for c in self.input_numeric_cols_ if c not in self._drop_columns]

        if self.encode_strategy == EncodeStrategy.TARGET:
            raise NotImplementedError(
                "EncodeStrategy.TARGET requires cross-validation for OOF isolation. "
                "Use the 'fold_ensemble' meta-provider with target_encode_cols instead:\n\n"
                "  model:\n"
                "    - provider: 'fold_ensemble'\n"
                "      model: 'base'\n"
                "      params:\n"
                "        target_encode_cols: ['cat_col']\n"
                "        n_folds: 5\n"
            )

        if self.encode_strategy == EncodeStrategy.NONE:
            self.input_categorical_cols_ = []
        for col in self.input_categorical_cols_:
            override = self.column_handling.get(col)
            strategy = override.encode if override is not None and override.encode is not None else self.encode_strategy
            if strategy == EncodeStrategy.NONE:
                continue
            if strategy == EncodeStrategy.TARGET:
                raise NotImplementedError(
                    "EncodeStrategy.TARGET requires cross-validation for OOF isolation. "
                    "Use the 'fold_ensemble' meta-provider with target_encode_cols instead:\n\n"
                    "  model:\n"
                    "    - provider: 'fold_ensemble'\n"
                    "      model: 'base'\n"
                    "      params:\n"
                    "        target_encode_cols: ['cat_col']\n"
                    "        n_folds: 5\n"
                )
            if strategy == EncodeStrategy.ONEHOT:
                encoder: OrdinalEncoder | OneHotEncoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoder.fit(X[[col]].astype(object).fillna("__NaN__"))
            else:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                encoder.fit(X[[col]])
            self.encoders_[col] = encoder

        for col in self.input_numeric_cols_:
            override = self.column_handling.get(col)
            scale_strategy = override.scale if override is not None and override.scale is not None else self.scale_strategy
            if scale_strategy == ScaleStrategy.NONE:
                continue
            scaler_map: dict[ScaleStrategy, type[StandardScaler | RobustScaler | MinMaxScaler]] = {
                ScaleStrategy.STANDARD: StandardScaler,
                ScaleStrategy.ROBUST: RobustScaler,
                ScaleStrategy.MINMAX: MinMaxScaler,
            }
            scaler_cls = scaler_map.get(scale_strategy)
            if scaler_cls is not None:
                scaler = scaler_cls()
                scaler.fit(X[[col]])
                self.scalers_[col] = scaler
        self.input_numeric_cols_ = list(self.scalers_.keys())

    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders and scaler, preserving original index.

        Args:
            X: DataFrame to transform.

        Returns:
            Transformed DataFrame with the same index as *X*.

        Raises:
            ConfigurationError: If columns from fit are missing.

        """
        X_copy = X.copy()

        for col in self.input_categorical_cols_:
            if col not in X_copy.columns:
                continue
            enc = self.encoders_.get(col)
            if enc is None:
                continue
            if isinstance(enc, OneHotEncoder):
                encoded = enc.transform(X_copy[[col]].astype(object).fillna("__NaN__"))
                X_copy = X_copy.drop(columns=[col])
                for j, fname in enumerate(enc.get_feature_names_out(input_features=[col])):
                    X_copy[fname] = pd.Series(encoded[:, j], index=X.index)
            else:
                encoded = enc.transform(X_copy[[col]])
                X_copy[col] = pd.Series(encoded.flatten(), index=X.index)

        for col, scaler in self.scalers_.items():
            if col not in X_copy.columns:
                raise ConfigurationError(f"Numeric columns from fit missing in transform: {col}")
            scaled = pd.Series(scaler.transform(X_copy[[col]]).flatten(), index=X.index)
            X_copy[col] = scaled

        for col in self._text_cols:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(str)

        return X_copy

    def run(self, context: PipelineContext) -> PipelineContext:
        """Fit on train, transform train + test, preserving indexes."""
        self._text_cols = [col for col, ft in (context.data.detected_types or {}).items() if ft == FeatureType.TEXT]
        if self._predict_mode:
            if self.encode_strategy not in (None, EncodeStrategy.NONE) and context.artifacts.encoders:
                self.encoders_ = context.artifacts.encoders
                self.input_categorical_cols_ = list(context.artifacts.encoders.keys())
            if self.scale_strategy not in (None, ScaleStrategy.NONE) and context.artifacts.scalers:
                self.scalers_ = context.artifacts.scalers
                self.input_numeric_cols_ = list(context.artifacts.scalers.keys())

            needs_encoders = self.encode_strategy not in (None, EncodeStrategy.NONE)
            needs_scaler = self.scale_strategy not in (None, ScaleStrategy.NONE)
            if needs_encoders or needs_scaler:
                if not self.encoders_ and not self.scalers_:
                    raise RuntimeError("EncodeScaleStep: encoders/scaler not found in artifacts — retrain with encode/scale enabled or set handling to 'none'")

        if context.data.train_df is not None:
            if not self._predict_mode:
                y = self._extract_target(context.data.train_df)
                self._fit(context.data.train_df, y)
                context.artifacts.encoders = self.encoders_
                context.artifacts.scalers = self.scalers_
                context.artifacts.scaler_features = self.input_numeric_cols_
                n_cat = len(self.input_categorical_cols_)
                n_num = len(self.input_numeric_cols_)
                logger.info(
                    "Encoded %d categorical cols (strategy=%s), scaled %d numeric cols (strategy=%s)",
                    n_cat,
                    self.encode_strategy.value if self.encode_strategy else "none",
                    n_num,
                    self.scale_strategy.value if self.scale_strategy else "none",
                )
            context.data.train_df = self._transform_features(context.data.train_df)
        if context.data.test_df is not None:
            context.data.test_df = self._transform_features(context.data.test_df)
        if context.data.holdout_df is not None:
            context.data.holdout_df = self._transform_features(context.data.holdout_df)
        return context

    def _extract_target(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame | None:  # noqa: ANN202
        """Extract target column(s) from a DataFrame.

        Args:
            df: Source DataFrame.

        Returns:
            ``pd.Series`` for a single target column, ``pd.DataFrame`` for
            multi-target, or ``None`` when no target is configured.

        """
        if self.target_col is None:
            return None
        if isinstance(self.target_col, list):
            missing = [c for c in self.target_col if c not in df.columns]
            if missing:
                raise ConfigurationError(f"Target columns not found: {missing}")
            return df[self.target_col]
        if self.target_col in df.columns:
            return df[self.target_col] if self.encode_strategy == EncodeStrategy.ORDINAL else df[[self.target_col]]
        return None
