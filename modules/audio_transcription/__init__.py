"""Audio and transcript analyzer module."""

import logging
from typing import Any, List, Optional

from core.base_analyzer import BaseAnalyzer
from core.data_types import ClipCandidate

from .speech_engine import SpeechEngine
from .wave_analyzer import WaveAnalyzer

logger = logging.getLogger(__name__)


class AudioTranscriptAnalyzer(BaseAnalyzer):
    """Combine volume spike detection with Faster-Whisper transcription."""

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        volume_threshold_dbfs: float = -20.0,
    ) -> None:
        """Initialize the audio/transcript analyzer.

        Args:
            whisper_model: Faster-Whisper model size.
            device: "cpu" or "cuda".
            compute_type: Faster-Whisper compute type.
            volume_threshold_dbfs: Minimum dBFS for spike detection.
        """
        self.wave_analyzer = WaveAnalyzer(threshold_dbfs=volume_threshold_dbfs)
        self.speech_engine = SpeechEngine(
            model_size=whisper_model,
            device=device,
            compute_type=compute_type,
        )

    def analyze(
        self,
        source: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> List[ClipCandidate]:
        """Analyze audio and return transcript-enriched candidates.

        Args:
            source: Path to the audio file.
            language: Optional ISO language code for Whisper.
            **kwargs: Optional overrides for wave analysis.

        Returns:
            List of clip candidates. Empty list on failure.
        """
        try:
            spikes = self.wave_analyzer.analyze(source, **kwargs)
            if not spikes:
                logger.info("No volume spikes found.")
                return []

            segments = self.speech_engine.transcribe(source, language=language)
            if not segments:
                logger.warning("No transcript segments found.")
                return []

            candidates: List[ClipCandidate] = []
            for spike in spikes:
                text = self.speech_engine.text_for_range(
                    segments,
                    spike.start_time,
                    spike.end_time,
                )
                if not text:
                    continue

                candidates.append(
                    ClipCandidate(
                        start_time=spike.start_time,
                        end_time=spike.end_time,
                        text=text,
                        volume_dbfs=spike.volume_dbfs,
                        source="audio_transcription",
                        metadata={"spike": spike.metadata},
                    )
                )

            return candidates
        except Exception as exc:
            logger.exception("Audio transcript analysis failed: %s", exc)
            return []