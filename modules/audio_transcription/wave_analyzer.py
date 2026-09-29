"""Audio waveform and volume spike analyzer."""

import logging
import os
from typing import Any, Dict, List

import numpy as np
from pydub import AudioSegment

from core.base_analyzer import BaseAnalyzer
from core.data_types import ClipCandidate
from core.media_utils import extract_audio_to_wav, is_video

logger = logging.getLogger(__name__)


def _load_audio(audio_path: str) -> AudioSegment:
    """Load an audio file, extracting from video if necessary.

    Args:
        audio_path: Path to audio or video file.

    Returns:
        AudioSegment with mono audio loaded.
    """
    if is_video(audio_path):
        wav_path = extract_audio_to_wav(audio_path, target_sample_rate=16000, mono=True)
        try:
            return AudioSegment.from_wav(wav_path)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                logger.warning("Could not delete temp file: %s", wav_path)

    return AudioSegment.from_file(audio_path)


def _to_mono_samples(segment: AudioSegment) -> np.ndarray:
    """Convert an AudioSegment to normalized mono float samples."""
    mono = segment.set_channels(1)
    samples = np.array(mono.get_array_of_samples(), dtype=np.float32)

    if mono.sample_width == 2:
        samples /= 32768.0
    elif mono.sample_width == 4:
        samples /= 2147483648.0
    else:
        samples /= float(2 ** (8 * mono.sample_width - 1))

    return samples


def _rms_dbfs(samples: np.ndarray) -> float:
    """Calculate RMS dBFS for a sample window."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms <= 0.0:
        return -120.0
    return float(20.0 * np.log10(rms))


def _frame_dbfs(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    hop_ms: int,
) -> List[Dict[str, float]]:
    """Compute dBFS over sliding windows."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))

    frames: List[Dict[str, float]] = []
    limit = max(1, len(samples) - frame_len + 1)

    for start in range(0, limit, hop_len):
        frame = samples[start:start + frame_len]
        frames.append({"time": start / sample_rate, "dbfs": _rms_dbfs(frame)})

    return frames


def detect_volume_spikes(
    audio_path: str,
    threshold_dbfs: float = -14.0,   # ← naikkan
    frame_ms: int = 100,             # ← window lebih besar (kurang sensitif)
    hop_ms: int = 50,
    min_duration: float = 1.2,       # ← minimal 1.2 detik
    merge_gap: float = 1.0,          # ← gabung spike berdekatan
) -> List[Dict[str, Any]]:
    """Detect loud regions such as laughter, shouting, or sudden spikes."""
    try:
        segment = _load_audio(audio_path)
        samples = _to_mono_samples(segment)
        frames = _frame_dbfs(samples, segment.frame_rate, frame_ms, hop_ms)

        active = [frame for frame in frames if frame["dbfs"] >= threshold_dbfs]
        if not active:
            return []

        ranges: List[Dict[str, Any]] = []
        start = prev_time = active[0]["time"]
        peak = active[0]["dbfs"]

        for frame in active[1:]:
            current = frame["time"]
            if current - prev_time <= merge_gap:
                prev_time = current
                peak = max(peak, frame["dbfs"])
                continue

            if prev_time - start >= min_duration:
                ranges.append({
                    "start": start,
                    "end": prev_time + frame_ms / 1000.0,
                    "volume_dbfs": peak,
                })

            start = prev_time = current
            peak = frame["dbfs"]

        if prev_time - start >= min_duration:
            ranges.append({
                "start": start,
                "end": prev_time + frame_ms / 1000.0,
                "volume_dbfs": peak,
            })

        return ranges
    except Exception as exc:
        logger.exception("Volume spike detection failed: %s", exc)
        return []


class WaveAnalyzer(BaseAnalyzer):
    """Analyzer that extracts loud regions from audio or video."""

    def __init__(
        self,
        threshold_dbfs: float = -14.0,
        frame_ms: int = 100,
        hop_ms: int = 50,
        min_duration: float = 1.2,
        merge_gap: float = 1.0,
    ) -> None:
        """Initialize the analyzer."""
        self.threshold_dbfs = threshold_dbfs
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.min_duration = min_duration
        self.merge_gap = merge_gap

    def analyze(self, source: str, **kwargs: Any) -> List[ClipCandidate]:
        """Analyze media and return volume-based candidates."""
        spikes = detect_volume_spikes(
            source,
            threshold_dbfs=float(kwargs.get("threshold_dbfs", self.threshold_dbfs)),
            frame_ms=int(kwargs.get("frame_ms", self.frame_ms)),
            hop_ms=int(kwargs.get("hop_ms", self.hop_ms)),
            min_duration=float(kwargs.get("min_duration", self.min_duration)),
            merge_gap=float(kwargs.get("merge_gap", self.merge_gap)),
        )
        return [
            ClipCandidate(
                start_time=spike["start"],
                end_time=spike["end"],
                text="",
                volume_dbfs=spike["volume_dbfs"],
                source="wave_analyzer",
                metadata={"spike": True},
            )
            for spike in spikes
        ]