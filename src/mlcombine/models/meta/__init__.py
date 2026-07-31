"""Meta-estimators — composite model strategies built on top of base providers."""

from mlcombine.models.meta.uplift import SLearner, TLearner
from mlcombine.models.meta.tuner import TunerWrapper
from mlcombine.models.meta.ensemble import EnsembleWrapper
from mlcombine.models.meta.stacking import StackingWrapper
from mlcombine.models.meta.fold_ensemble import FoldEnsemble

__all__ = [
    "SLearner",
    "TLearner",
    "TunerWrapper",
    "EnsembleWrapper",
    "StackingWrapper",
    "FoldEnsemble",
]
