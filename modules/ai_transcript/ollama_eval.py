"""Ollama-based viral transcript evaluator."""

import json
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional
from urllib import request

from core.base_analyzer import BaseAnalyzer
from core.data_types import ClipCandidate

logger = logging.getLogger(__name__)


class OllamaEvaluator(BaseAnalyzer):
    """Evaluate each transcript candidate with a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
        min_score: float = 60.0,
    ) -> None:
        """Initialize the Ollama evaluator.

        Args:
            model: Ollama model name.
            host: Ollama server base URL.
            timeout: Request timeout in seconds.
            min_score: Minimum AI score to keep a candidate.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.min_score = min_score

    def analyze(self, source: Any, **kwargs: Any) -> List[ClipCandidate]:
        """Evaluate clip candidates one-by-one using Ollama.

        Args:
            source: List of ClipCandidate objects.
            **kwargs: Optional overrides for model, host, timeout, min_score.

        Returns:
            Filtered list of viral candidates. Empty list on failure.
        """
        if not isinstance(source, list) or not source:
            return []

        candidates = [item for item in source if isinstance(item, ClipCandidate)]
        if not candidates:
            return []

        model = str(kwargs.get("model", self.model))
        host = str(kwargs.get("host", self.host)).rstrip("/")
        timeout = int(kwargs.get("timeout", self.timeout))
        min_score = float(kwargs.get("min_score", self.min_score))

        results: List[ClipCandidate] = []
        total = len(candidates)

        for index, candidate in enumerate(candidates):
            if not candidate.text.strip():
                continue

            logger.info("Evaluating candidate %d/%d", index + 1, total)
            try:
                evaluation = self._evaluate_one(candidate, model, host, timeout)
            except Exception as exc:
                logger.error("Candidate %d eval failed: %s", index, exc)
                continue

            if not evaluation:
                continue

            try:
                score = float(evaluation.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0

            is_viral = bool(evaluation.get("is_viral", False))
            reason = str(evaluation.get("reason", ""))

            logger.debug(
                "Candidate %d -> score=%.1f is_viral=%s reason=%s",
                index, score, is_viral, reason,
            )

            if not is_viral or score < min_score:
                continue

            results.append(
                replace(
                    candidate,
                    score=score,
                    reason=reason,
                    source="ai_transcript",
                    metadata={**candidate.metadata, "ai": evaluation},
                )
            )

        return results

    def _evaluate_one(
        self,
        candidate: ClipCandidate,
        model: str,
        host: str,
        timeout: int,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single candidate and return parsed AI verdict."""
        prompt = self._build_prompt(candidate)
        raw = self._call_ollama(prompt, model, host, timeout)
        if not raw:
            return None
        return self._parse_single(raw)

    @staticmethod
    def _build_prompt(candidate: ClipCandidate) -> str:
        """Build a per-candidate prompt for viral scoring."""
        text = candidate.text.strip()[:600]
        return (
            "You are a viral clip editor for short-form content "
            "(TikTok, YouTube Shorts, Reels).\n"
            "Evaluate the transcript segment below for viral potential.\n"
            "Consider: comedy, surprise, emotion, conflict, memorable quote, "
            "relatable moments, funny delivery.\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            '{"score": <integer 0-100>, "is_viral": <true|false>, '
            '"reason": "<short reason in max 15 words>"}\n\n'
            f"Transcript:\n{text}"
        )

    def _call_ollama(
        self,
        prompt: str,
        model: str,
        host: str,
        timeout: int,
    ) -> str:
        """Call Ollama /api/generate with JSON mode."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 200},
        }

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")

        parsed = json.loads(body)
        return str(parsed.get("response", ""))

    def _parse_single(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a single-candidate JSON verdict."""
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON from Ollama: %s | raw=%s", exc, text[:300])
            return None

        if not isinstance(data, dict):
            logger.error("Expected dict, got %s", type(data).__name__)
            return None

        # Normalize: kalau model bungkus dalam key lain (e.g. "results")
        if "score" not in data:
            for key in ("result", "verdict", "evaluation", "data", "output"):
                nested = data.get(key)
                if isinstance(nested, dict) and "score" in nested:
                    return nested

        return data