"""Shared data contracts for Auto-Clipper Engine."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TranscriptSegment:
    """A single timed transcript segment."""

    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)


@dataclass
class ClipCandidate:
    """A potential clip candidate produced by an analyzer."""

    start_time: float
    end_time: float
    text: str
    volume_dbfs: Optional[float] = None
    score: float = 0.0
    reason: str = ""
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipCandidate":
        """Build a candidate from a dictionary."""
        return cls(
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            text=str(data.get("text", "")),
            volume_dbfs=data.get("volume_dbfs"),
            score=float(data.get("score", 0.0)),
            reason=str(data.get("reason", "")),
            source=str(data.get("source", "")),
            metadata=dict(data.get("metadata", {})),
        )