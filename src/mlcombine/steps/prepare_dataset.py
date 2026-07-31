"""Pipeline step for data acquisition — downloads if URL, extracts if archive."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, DatasetNotFoundError, MLCombineConfig, PipelineContext

_ARCHIVE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
    }
)

logger = logging.getLogger(__name__)


@registry.step("PrepareDatasetStep")
class PrepareDatasetStep(BaseStep[PipelineContext]):
    """Ensures data is available locally: downloads if URL, extracts if archive."""

    train = True
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        """Initialize with config — extracts train/test paths and force flag."""
        self.train_df = cfg.data.train_df
        self.test_df = cfg.data.test_df
        self.force = cfg.data.force_prepare_dataset

    @classmethod
    def _is_remote_or_archive(cls, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return True
        name = source.lower()
        return any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)

    @classmethod
    def is_required(cls, cfg: MLCombineConfig) -> bool:
        return cls._is_remote_or_archive(cfg.data.train_df) or cls._is_remote_or_archive(cfg.data.test_df)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Resolve train and test datasets to local paths."""
        context.data.train_df_path = self._resolve(self.train_df)
        if self.test_df:
            context.data.test_df_path = self._resolve(self.test_df)
        return context

    def _resolve(self, source: str) -> Path:
        """Download (if URL) and extract (if archive), return local path."""
        local = self._fetch_if_remote(source)
        return self._extract_if_archive(local)

    def _fetch_if_remote(self, source: str) -> Path:
        """Download if remote source, otherwise return local path as-is."""
        if source.startswith(("http://", "https://")):
            logger.info("Downloading from: %s", source)
            return self._download(source)
        logger.info("Loading from: %s", source)
        local = Path(source)
        if not local.exists():
            raise DatasetNotFoundError(f"Local dataset not found: {local}")
        return local

    def _download(self, url: str) -> Path:
        """Download a file from a URL and return the local path."""
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "downloaded_file"
        if "." not in filename:
            filename += ".tmp"

        download_dir = Path(".")
        download_dir.mkdir(parents=True, exist_ok=True)
        file_path = download_dir / filename

        if file_path.exists() and not self.force:
            logger.info("Already downloaded: %s", file_path)
            return file_path

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return file_path

    @staticmethod
    def _is_archive(path: Path) -> bool:
        name = path.name.lower()
        return any(name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)

    def _extract_if_archive(self, path: Path) -> Path:
        if not self._is_archive(path):
            return path
        if not self.force and self._already_extracted(path):
            logger.info("Already extracted: %s", path)
            return path.parent / path.stem
        logger.info("Extracting archive: %s", path)
        return self._extract(path)

    @staticmethod
    def _already_extracted(archive: Path) -> bool:
        extract_to = archive.parent / archive.stem
        return extract_to.exists() and any(extract_to.iterdir())

    @staticmethod
    def _extract(archive_path: Path) -> Path:
        extract_to = archive_path.parent
        extract_to.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive_path), str(extract_to))
        return extract_to
