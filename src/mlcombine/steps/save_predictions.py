"""SavePredictionsStep — write pre-computed predictions to CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from mlcombine.core.registry import registry
from mlcombine.core.types import BaseStep, MLCombineConfig, PipelineContext

logger = logging.getLogger(__name__)


@registry.step("SavePredictionsStep")
class SavePredictionsStep(BaseStep[PipelineContext]):
    """Write pre-computed predictions from context to CSV.

    Configure via ``step_config.save_predictions``::

        save_predictions:
          target_col: rent_price   # default: data.target_col
          sep: ","

    Side Effects:
        - Creates parent directories and writes a CSV file to disk.
    """

    train = False
    predict = True

    def __init__(self, cfg: MLCombineConfig, *, predict: bool = False, weights: str | None = None) -> None:
        self._output_path = weights or cfg.trainer.output_file
        sp = getattr(cfg.step_config, "save_predictions", None) or {}
        default_target = cfg.data.target_col if isinstance(cfg.data.target_col, str) else "prediction"
        self._target_column = sp.get("target_col") or default_target
        self._sep = sp.get("sep", None)

    def run(self, context: PipelineContext) -> PipelineContext:
        output_path = Path(self._output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        preds = context.data.predictions
        if preds is None:
            raise RuntimeError("No predictions in context — ModelPredictStep must run before SavePredictionsStep")

        out: dict[str, object] = {}
        ids = context.data.prediction_ids
        if ids is not None:
            col_name = str(ids.name) if ids.name is not None else "id"
            out[col_name] = ids

        out[self._target_column] = preds

        sub = pd.DataFrame(out)
        sep = self._sep if self._sep else ("\t" if output_path.suffix == ".tsv" else ",")
        sub.to_csv(output_path, sep=sep, index=False)
        logger.info(
            "Predictions saved to %s — %d rows, columns: %s",
            output_path,
            len(sub),
            list(sub.columns),
        )
        return context


__all__ = [
    "SavePredictionsStep",
]
