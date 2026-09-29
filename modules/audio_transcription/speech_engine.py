"""Faster-Whisper speech transcription engine."""

import logging
import os
from typing import Any, List, Optional

from core.data_types import TranscriptSegment
from core.media_utils import extract_audio_to_wav, is_video

logger = logging.getLogger(__name__)


class SpeechEngine:
    """Thin wrapper around Faster-Whisper for timed transcription."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """Initialize the speech engine."""
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    def _get_model(self) -> Any:
        """Lazily create the Faster-Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe an audio or video file and return timed segments."""
        tmp_wav: Optional[str] = None
        source_path = audio_path

        try:
            if is_video(audio_path):
                tmp_wav = extract_audio_to_wav(audio_path, target_sample_rate=16000, mono=True)
                source_path = tmp_wav

            model = self._get_model()
            segments, _ = model.transcribe(
                source_path,
                language=language,
                vad_filter=True,
            )

            result: List[TranscriptSegment] = []
            for segment in segments:
                text = str(segment.text).strip()
                if text:
                    result.append(
                        TranscriptSegment(
                            start=float(segment.start),
                            end=float(segment.end),
                            text=text,
                        )
                    )
            return result
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            return []
        finally:
            if tmp_wav:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    logger.warning("Could not delete temp file: %s", tmp_wav)

    @staticmethod
    def text_for_range(
        segments: List[TranscriptSegment],
        start: float,
        end: float,
    ) -> str:
        """Return transcript text overlapping a time range."""
        parts = [
            segment.text
            for segment in segments
            if segment.end >= start and segment.start <= end and segment.text
        ]
        return " ".join(parts).strip()