"""Configuration for Auto-Clipper Engine."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"   # ← sudah diupdate
OLLAMA_HOST = "http://localhost:11434"

VOLUME_THRESHOLD_DBFS = -14.0   # ← naikkan supaya kandidat lebih sedikit
MIN_AI_SCORE = 60.0