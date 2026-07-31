"""TypeDetectStep — automatic column feature-type detection using FeatureType enum.

No auto-guessing of target column; requires explicit target_col or raises ConfigurationError.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from mlcombine.core.types import (
    BaseStep,
    ConfigurationError,
    FeatureMap,
    FeatureType,
    MLCombineConfig,
    PipelineContext,
    TargetColumn,
    TaskType,
)
from mlcombine.core.registry import registry

logger = logging.getLogger(__name__)


@registry.step("TypeDetectStep")
class TypeDetectStep(BaseStep[PipelineContext]):
    """Pipeline step that detects column feature types and task type."""

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — requires target_col."""
        if cfg.data.target_col is None:
            raise ConfigurationError("target_col is required — auto-detection is forbidden")
        self.target_col: TargetColumn = cfg.data.target_col
        self.force_types: FeatureMap = cfg.force_types or {}
        self.detected_types: FeatureMap = {}
        self._explicit_task_type: TaskType | None = cfg.data.task_type

    def run(self, context: PipelineContext) -> PipelineContext:
        """Detect column types and task type, then store results on the context."""
        df = context.data.train_df
        if df is None:
            raise ConfigurationError("No train_df found in context for type detection")

        self.detected_types = self._detect_column_types(df)

        type_groups: dict[str, list[str]] = {}
        for col, ft in self.detected_types.items():
            label = ft.value if hasattr(ft, "value") else str(ft)
            type_groups.setdefault(label, []).append(col)
        for label, cols in sorted(type_groups.items()):
            logger.info("Type %-15s → %s", label, ", ".join(cols))

        if self._explicit_task_type is not None:
            self.task_type = self._explicit_task_type
        else:
            target_data = self._get_target_data(df)
            self.task_type = self._determine_task_type(target_data)

        context.data.detected_types = self.detected_types
        context.data.task_type = self.task_type
        context.data.target_col = self.target_col
        return context

    @staticmethod
    def _check_custom_handlers(series: pd.Series) -> str | None:
        """Check registry feature handlers; return the first matching type name or ``None``."""
        for type_name, handler_cls in registry.feature_handler_types.items():
            handler = handler_cls()
            if handler.detect(series):
                return type_name
        return None

    def _detect_column_types(self, df: pd.DataFrame) -> FeatureMap:
        detected: FeatureMap = {}
        nrows = len(df)

        for col in df.columns:
            if col in self.force_types:
                detected[col] = self.force_types[col]
                continue

            dtype = df[col].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_integer_dtype(dtype) and nrows > 0:
                    unique_ratio = df[col].nunique() / nrows
                    detected[col] = FeatureType.CATEGORY if unique_ratio < 0.05 else FeatureType.NUMBER
                else:
                    detected[col] = FeatureType.NUMBER

            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                detected[col] = self._classify_object_column(df[col])

            else:
                # Check registry feature handlers before marking UNKNOWN
                custom_type = self._check_custom_handlers(df[col])
                if custom_type is not None:
                    detected[col] = custom_type
                else:
                    logger.warning("Unrecognized dtype %s for column %s, marking UNKNOWN", dtype, col)
                    detected[col] = FeatureType.UNKNOWN

        return detected

    def _classify_object_column(self, series: pd.Series) -> FeatureType | str:
        if self._is_datetime_column(series):
            return FeatureType.DATETIME
        if self._is_image_path_column(series):
            return FeatureType.IMAGE_PATH
        if self._is_sequence_column(series):
            return FeatureType.SEQUENCE_TOKEN
        if self._is_text_column(series):
            return FeatureType.TEXT

        # Custom feature handlers from registry
        for type_name, handler_cls in registry.feature_handler_types.items():
            handler = handler_cls()
            if handler.detect(series):
                return type_name

        nrows = len(series)
        if nrows > 0:
            unique_ratio = series.nunique() / nrows
            if unique_ratio < 0.5:
                return FeatureType.CATEGORY

        return FeatureType.TEXT

    @staticmethod
    def _is_datetime_column(series: pd.Series) -> bool:
        try:
            pd.to_datetime(series.dropna().head(100), format="mixed")
            return True
        except ValueError, TypeError:
            return False

    @staticmethod
    def _is_image_path_column(series: pd.Series) -> bool:
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}
        count = sum(1 for val in sample if isinstance(val, str) and any(val.lower().endswith(ext) for ext in image_extensions))
        return (count / len(sample)) > 0.5

    @staticmethod
    def _is_sequence_column(series: pd.Series) -> bool:
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        hex_pat = re.compile(r"0x[0-9a-fA-F]+")
        space_pat = re.compile(r"^[\w\-.]+(\s+[\w\-.]+)*$")
        count = 0
        for val in sample:
            if isinstance(val, str):
                if hex_pat.search(val) or (space_pat.match(val.strip()) and len(val.split()) > 1):
                    count += 1
        return (count / len(sample)) > 0.3

    @staticmethod
    def _is_text_column(series: pd.Series) -> bool:
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        total = 0
        cnt = 0
        for val in sample:
            if isinstance(val, str):
                total += len(val)
                cnt += 1
        return cnt > 0 and (total / cnt) > 50

    def _get_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.target_col, list):
            missing = [c for c in self.target_col if c not in df.columns]
            if missing:
                raise ConfigurationError(f"Target columns not found: {missing}")
            return df[self.target_col]
        if isinstance(self.target_col, dict):
            cols = list(self.target_col.keys())
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ConfigurationError(f"Target columns not found: {missing}")
            return df[cols]
        if self.target_col in df.columns:
            return df[[self.target_col]]
        raise ConfigurationError(f"Target column '{self.target_col}' not found in dataframe")

    @staticmethod
    def _determine_task_type(target_data: pd.DataFrame) -> TaskType:
        if target_data.shape[1] > 1:
            return TaskType.MULTITASK
        series = target_data.iloc[:, 0]
        if pd.api.types.is_numeric_dtype(series.dtype):
            if pd.api.types.is_integer_dtype(series.dtype):
                unique_count = series.nunique()
                if unique_count <= 20:
                    return TaskType.CLASSIFICATION
            return TaskType.REGRESSION
        return TaskType.CLASSIFICATION
