"""Load all built-in modules so their ``@registry.*`` decorators fire.

This module is imported by ``mlcombine.__init__`` before anything else,
so logging must be configured here to capture registration DEBUG messages.
"""

from __future__ import annotations

import logging  # noqa: E402
import sys  # noqa: E402

logging.basicConfig(  # noqa: E402
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

# Built-in pipeline steps — import the package so all @registry.step() decorators fire
import mlcombine.steps  # noqa: F401, E402

# Built-in model providers — import the package so all @registry.model_provider() decorators fire
import mlcombine.models.providers  # noqa: F401, E402

# Uplift meta-model providers (t_learner, s_learner)
import mlcombine.models.meta  # noqa: F401, E402

# Built-in evaluators (holdout, cv, etc.)
import mlcombine.evaluators  # noqa: F401, E402

# Built-in tensor adapters
try:  # noqa: E402
    from mlcombine.core.tensor import TorchAdapter  # noqa: F401
except ImportError:
    pass
