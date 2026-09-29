"""Abstract analyzer contract for all pluggable modules."""

from abc import ABC, abstractmethod
from typing import Any, List

from core.data_types import ClipCandidate


class BaseAnalyzer(ABC):
    """Base class for every analyzer in Auto-Clipper Engine."""

    @abstractmethod
    def analyze(self, source: Any, **kwargs: Any) -> List[ClipCandidate]:
        """Analyze a source and return clip candidates.

        Args:
            source: Input data. Concrete analyzers define the expected type.
            **kwargs: Optional analyzer-specific parameters.

        Returns:
            List of clip candidates. Empty list on failure.
        """
        raise NotImplementedError