"""Trying to prototype fitting everything into olmo core."""

import logging

from olmoearth_pretrain.evals.models import CopernicusFMConfig
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
)

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> CopernicusFMConfig:
    """Build the model config for an experiment."""
    model_config = CopernicusFMConfig()
    return model_config
