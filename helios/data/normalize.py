"""Normalize the data."""

import numpy as np

from helios.data.constants import ModalitySpec

# With predefined, we should be able to get the min & max values for each band
# With computed, we will be getting the mean & std values for each band

# For value large values, do we need to cut them off?


class Normalizer:
    """Normalize the data."""

    def __init__(self, modality: ModalitySpec, computed_values: bool = True) -> None:
        """Initialize the normalizer.

        Args:
            modality: The modality to normalize.
            computed_values: Whether to use computed values or predefined values.

        Returns:
            None
        """
        self.modality = modality
        self.computed_values = computed_values

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data."""
        pass
