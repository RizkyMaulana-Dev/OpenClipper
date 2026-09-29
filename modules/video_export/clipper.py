"""FFmpeg-based video clipper."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_analyzer import BaseAnalyzer
from core.data_types import ClipCandidate
from core.media_utils import _ffmpeg_cmd, ensure_ffmpeg_available

logger = logging.getLogger(__name__)


class VideoClipper(BaseAnalyzer):
    """Cut video files into clips based on candidate timestamps."""

    def __init__(
        self,
        output_dir: str = "output",
        vertical: bool = False,
        re_encode: bool = False,
    ) -> None:
        """Initialize the video clipper.

        Args:
            output_dir: Folder to save output clips.
            vertical: Convert to 9:16 vertical for shorts.
            re_encode: Force re-encode (slower but accurate cuts).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vertical = vertical
        self.re_encode = re_encode

    def analyze(self, source: Any, **kwargs: Any) -> List[ClipCandidate]:
        """Export clips from video based on candidates.

        Args:
            source: Path to source video.
            **kwargs: Must contain `candidates` list of ClipCandidate.

        Returns:
            List of candidates with `clip_path` added to metadata.
        """
        video_path = str(source)
        candidates = kwargs.get("candidates", [])

        if not candidates:
            logger.warning("No candidates to export")
            return []

        ensure_ffmpeg_available()
        exported: List[ClipCandidate] = []

        for index, candidate in enumerate(candidates):
            output = self.output_dir / f"clip_{index:03d}_{int(candidate.score)}.mp4"
            try:
                self._export_one(video_path, candidate, output)
                candidate.metadata["clip_path"] = str(output)
                exported.append(candidate)
                logger.info("Exported clip %d -> %s", index, output.name)
            except Exception as exc:
                logger.error("Failed to export clip %d: %s", index, exc)

        return exported

    def _export_one(
        self,
        video_path: str,
        candidate: ClipCandidate,
        output: Path,
    ) -> None:
        """Cut a single clip with FFmpeg."""
        duration = candidate.duration
        cmd = [_ffmpeg_cmd(), "-y"]

        # Fast seek before -i (accurate when combined with re-encode)
        if self.re_encode:
            cmd += ["-ss", f"{candidate.start_time:.3f}", "-i", video_path,
                    "-t", f"{duration:.3f}"]
        else:
            cmd += ["-ss", f"{candidate.start_time:.3f}", "-i", video_path,
                    "-t", f"{duration:.3f}", "-c", "copy"]

        if self.vertical:
            cmd += [
                "-vf",
                "crop=ih*9/16:ih,scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
            ]

        cmd += ["-loglevel", "error", str(output)]

        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg clip failed: {stderr[:400]}")