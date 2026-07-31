"""DataLoaderStep — strict file-format validation with DatasetFormat enum, presence guards, and sep isolation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import (
    SUPPORTED_DATASET_SUFFIXES,
    BaseStep,
    DatasetFormat,
    DatasetNotFoundError,
    EmptyDatasetError,
    MLCombineConfig,
    PipelineContext,
)

logger = logging.getLogger(__name__)


_FORMAT_MAP: dict[str, DatasetFormat] = {
    ".csv": DatasetFormat.CSV,
    ".tsv": DatasetFormat.TSV,
    ".parquet": DatasetFormat.PARQUET,
    ".xlsx": DatasetFormat.XLSX,
    ".xls": DatasetFormat.XLS,
}


def _resolve_format(path: Path) -> DatasetFormat:
    for suffix, fmt in _FORMAT_MAP.items():
        if path.suffix == suffix or path.name.endswith(suffix):
            return fmt
    raise ValueError(f"Unsupported file format: {path.suffix}. Supported: {sorted(SUPPORTED_DATASET_SUFFIXES)}")


@registry.step("DataLoaderStep")
class DataLoaderStep(BaseStep[PipelineContext]):
    """Pipeline step that loads train/test DataFrames from disk."""

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — extracts separator and column names."""
        self.sep = cfg.data.sep
        self.target_col = cfg.data.target_col if isinstance(cfg.data.target_col, (str, list)) else None
        self.treatment_col = cfg.data.treatment_col
        self._train_df_path_str = cfg.data.train_df
        self._test_df_path_str = cfg.data.test_df

    def run(self, context: PipelineContext) -> PipelineContext:
        """Load DataFrames from the paths stored on the pipeline context."""
        train_path = context.data.train_df_path or Path(self._train_df_path_str)
        if train_path:
            context.data.train_df = self._load_file(train_path)
            logger.info("Loaded train_df: %d rows, %d columns", *context.data.train_df.shape)
        test_path = context.data.test_df_path or Path(self._test_df_path_str)
        if test_path:
            context.data.test_df = self._load_file(test_path)
            logger.info("Loaded test_df: %d rows, %d columns", *context.data.test_df.shape)
        context.data.treatment_col = self.treatment_col
        return context

    def _load_file(self, path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise DatasetNotFoundError(f"Dataset not found: {path}")

        fmt = _resolve_format(path)

        if fmt in (DatasetFormat.CSV, DatasetFormat.TSV):
            sep = "\t" if fmt == DatasetFormat.TSV else self.sep
            df = pd.read_csv(path, sep=sep)
        elif fmt == DatasetFormat.PARQUET:
            try:
                df = pd.read_parquet(path, engine="fastparquet")
            except ImportError:
                df = pd.read_parquet(str(path), engine="pyarrow", to_pandas_kwargs={})
        elif fmt in (DatasetFormat.XLSX, DatasetFormat.XLS):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        if df.empty:
            raise EmptyDatasetError(f"Loaded dataset is empty: {path}")

        return df
