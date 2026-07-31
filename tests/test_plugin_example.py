"""Test plugin for verifying plugin loading machinery.

Registers a dummy feature handler and a dummy step via the registry.
"""

import pandas as pd

from mlcombine.core.registry import FeatureHandler, registry


@registry.feature_handler("test_plugin_type")
class TestPluginHandler(FeatureHandler):
    """Matches series containing the word 'plugin'."""

    def detect(self, series: pd.Series) -> bool:
        return bool(series.astype(str).str.contains("plugin", case=False, na=False).mean() > 0.5)


@registry.step("test_plugin_step")
class TestPluginStep:
    """Dummy step that does nothing."""

    pass
