"""Simple test pipeline for Auto-Clipper Engine."""

import argparse
import json
import logging
import sys

from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WHISPER_MODEL,
    MIN_AI_SCORE,
    OLLAMA_HOST,
    VOLUME_THRESHOLD_DBFS,
)
from core.clip_merger import merge_candidates
from core.media_utils import ensure_ffmpeg_available
from modules.ai_transcript import OllamaEvaluator
from modules.audio_transcription import AudioTranscriptAnalyzer
from modules.video_export import VideoClipper

logger = logging.getLogger(__name__)


def main() -> int:
    """Run full pipeline: analyze, evaluate, merge, export."""
    parser = argparse.ArgumentParser(description="Auto-Clipper Engine pipeline")
    parser.add_argument("media_path", help="Path to audio or video file")
    parser.add_argument("--language", default=None, help="ISO language code")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--min-score", type=float, default=MIN_AI_SCORE)
    parser.add_argument("--export", action="store_true", help="Export video clips")
    parser.add_argument("--vertical", action="store_true", help="9:16 crop for shorts")
    parser.add_argument("--re-encode", action="store_true", help="Accurate cut (slower)")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        ensure_ffmpeg_available()
    except RuntimeError as exc:
        logger.error("FFmpeg not ready: %s", exc)
        return 1

    # Step 1: Audio + transcript
    audio_analyzer = AudioTranscriptAnalyzer(
        whisper_model=args.whisper_model,
        volume_threshold_dbfs=VOLUME_THRESHOLD_DBFS,
    )
    candidates = audio_analyzer.analyze(args.media_path, language=args.language)
    logger.info("Audio candidates: %d", len(candidates))
    if not candidates:
        return 0

    # Step 2: AI evaluation
    evaluator = OllamaEvaluator(
        model=args.ollama_model,
        host=OLLAMA_HOST,
        min_score=args.min_score,
    )
    viral = evaluator.analyze(candidates)
    logger.info("Viral candidates: %d", len(viral))
    if not viral:
        return 0

    # Step 3: Merge nearby candidates into longer clips
    merged = merge_candidates(
        viral,
        max_gap=8.0,
        min_duration=15.0,
        max_duration=60.0,
        padding=1.5,
    )
    logger.info("Merged clips: %d", len(merged))

    # Step 4: Export (optional)
    final = merged
    if args.export:
        clipper = VideoClipper(
            output_dir=args.output_dir,
            vertical=args.vertical,
            re_encode=args.re_encode,
        )
        final = clipper.analyze(args.media_path, candidates=merged)
        logger.info("Exported: %d clips to %s", len(final), args.output_dir)

    logger.info(
        "Result JSON:\n%s",
        json.dumps([c.to_dict() for c in final], indent=2, ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())