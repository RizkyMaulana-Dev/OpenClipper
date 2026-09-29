"""FFmpeg-based media utilities for audio extraction."""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

# Fallback: kalau PATH tidak terbaca Python, cari manual di lokasi umum
_FFMPEG_SEARCH_DIRS = [
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\Program Files (x86)\ffmpeg\bin",
    os.path.expanduser(r"~\ffmpeg\bin"),
    os.path.expanduser(r"~\Downloads\ffmpeg\bin"),
]


def _find_binary(name: str) -> str | None:
    """Find a binary in PATH or common fallback locations.

    Args:
        name: Binary name without extension (e.g., "ffmpeg").

    Returns:
        Full path to binary, or None if not found.
    """
    # 1. Coba PATH dulu
    found = shutil.which(name)
    if found:
        return found

    exe_name = f"{name}.exe" if os.name == "nt" else name

    # 2. Scan folder umum
    for folder in _FFMPEG_SEARCH_DIRS:
        candidate = Path(folder) / exe_name
        if candidate.exists():
            return str(candidate)

    # 3. Scan khusus Winget (nested path)
    if os.name == "nt":
        winget_root = Path(os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages"
        ))
        if winget_root.exists():
            for exe in winget_root.glob(f"**/{exe_name}"):
                return str(exe)

    return None


def ensure_ffmpeg_available() -> None:
    """Verify that ffmpeg and ffprobe exist somewhere reachable.

    Raises:
        RuntimeError: If either binary cannot be found.
    """
    missing = [name for name in ("ffmpeg", "ffprobe") if _find_binary(name) is None]
    if missing:
        raise RuntimeError(
            f"Missing binaries: {', '.join(missing)}. "
            "Install FFmpeg and make sure it's in PATH. "
            "Download: https://www.gyan.dev/ffmpeg/builds/"
        )


def _ffmpeg_cmd() -> str:
    """Get the ffmpeg binary absolute path."""
    binary = _find_binary("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg not found")
    return binary


def is_video(path: str | Path) -> bool:
    """Return True if the path has a known video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_audio_to_wav(
    media_path: str,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> str:
    """Extract audio from any media file to a temp WAV file.

    Args:
        media_path: Path to input media (video or audio).
        target_sample_rate: Output sample rate in Hz.
        mono: Downmix to mono if True.

    Returns:
        Path to the temporary WAV file.

    Raises:
        RuntimeError: If FFmpeg fails or input file is missing.
    """
    ensure_ffmpeg_available()

    src = Path(media_path)
    if not src.exists():
        raise RuntimeError(f"Media file not found: {media_path}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="acl_")
    os.close(tmp_fd)
    Path(tmp_path).unlink(missing_ok=True)

    cmd = [
        _ffmpeg_cmd(),
        "-y",
        "-i", str(src),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-loglevel", "error", tmp_path]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg binary not found.") from exc

    if result.returncode != 0:
        Path(tmp_path).unlink(missing_ok=True)
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg extraction failed: {stderr[:500]}")

    logger.info("Extracted audio to %s", tmp_path)
    return tmp_path