"""Shared utilities — merge guard and reusable helpers."""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from mlcombine.core.exceptions import RowExplosionError

logger = logging.getLogger(__name__)


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["left", "right", "outer", "inner", "cross"] = "left",
    on: str | list[str] | None = None,
    left_on: str | None = None,
    right_on: str | None = None,
    right_index: bool = False,
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """Merge two DataFrames with row-explosion and suffix-bleeding guards.

    Args:
        left: Left (master) DataFrame.
        right: Right (lookup) DataFrame.
        how: Join type (``"left"``, ``"inner"``, …).
        on: Column(s) to join on.
        left_on: Left-side join column.
        right_on: Right-side join column.
        right_index: Use right DataFrame index as join key.
        suffixes: Suffix pair for overlapping non-key columns.

    Returns:
        Merged DataFrame.

    Raises:
        RowExplosionError: If a left join unexpectedly produces more rows
            than the original left DataFrame.

    Side Effects:
        Logs a warning about non-key column overlap that may produce
        ``_x`` / ``_y`` suffixes.
    """
    len_before = len(left)

    key_cols: set[str] = set()
    if on is not None:
        key_cols = {on} if isinstance(on, str) else set(on)
    if left_on is not None:
        key_cols.add(left_on)
    right_keys = {right_on} if right_on else (set(right.columns) if right_index else set())
    non_key_overlap = set(left.columns) - key_cols & set(right.columns) - right_keys
    if non_key_overlap:
        logger.warning(
            "Column overlap detected (may produce _x/_y suffixes): %s",
            non_key_overlap,
        )

    merged = left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        right_index=right_index,
        suffixes=suffixes,
    )

    if how == "left" and len(merged) > len_before:
        raise RowExplosionError(f"Left join exploded from {len_before} to {len(merged)} rows")
    return merged


__all__ = [
    "safe_merge",
]
