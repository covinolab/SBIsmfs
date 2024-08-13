from typing import Any, Dict, Optional
import numpy as np
from sbi_smfs.utils.summary_stats import build_transition_matricies

class Observation:
    def __init__(self, config: Dict[str, Any], q: np.ndarray) -> None:
        self.config = config
        self._raw_q = q
        self._summary_stats = None
        self.calculate_summary_stats()

    def calculate_summary_stats(self) -> None:
        """
        Calculate summary statistics from raw q data.
        """
        self._summary_stats = build_transition_matricies(
            self._raw_q,
            self.config.posterior.summary_statistics.lag_times,
            self.config.posterior.summary_statistics.min_bin,
            self.config.posterior.summary_statistics.max_bin,
            self.config.posterior.summary_statistics.num_bins
        )

    @property
    def q(self) -> Optional[np.ndarray]:
        """
        Returns the summary statistics of the observation.
        """
        return self._summary_stats

    @property
    def raw_q(self) -> np.ndarray:
        """
        Returns the raw q data of the observation.
        """
        return self._raw_q

    def __repr__(self) -> str:
        return f"Observation(raw_q_shape={self._raw_q.shape}, summary_stats_shape={self._summary_stats.shape if self._summary_stats is not None else None})"