"""VADER-based sentiment analyzer with custom urgency lexicon.

Public API:
    SENTIMENT_CLASSES    : ordered tuple of the 4 locked label names.
    SentimentAnalyzer    : thin wrapper around VADER's `SentimentIntensityAnalyzer`
                           augmented with a hand-curated urgency / escalation
                           lexicon for support tickets. Exposes:
                             - analyze(text) -> SentimentResult
                             - urgency_signals(text) -> list[str]
                             - negative_intensity(scores) -> float
                           The wrapper is intentionally tiny and stateless so it
                           can be reused by `priority.py` (Day 10) and the
                           inference pipeline (Day 11) without coupling.

Design notes
------------
VADER ships a lexicon tuned for social media, not customer support. Tickets
use escalation vocabulary that VADER under-weights ("charged twice",
"unacceptable", "lawsuit", "urgent"). We extend the analyzer by adding a
small urgency lexicon with stronger negative polarity, then run a second
pass that:
  1) collects the urgency phrases actually present in the text
     (returned as `urgency_signals` for the API),
  2) escalates the label when urgency matches are dense or compound is
     already strongly negative.

Label mapping (locked in docs/PROJECT_PLAN.md §6):
    Positive    compound >= +0.20  (after custom boost)
    Neutral     compound in [-0.05, +0.20)
    Frustrated  compound in [-0.50, -0.05)
    Angry       compound <  -0.50 OR (>=2 urgency hits AND neg>=0.35)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENTIMENT_CLASSES: tuple[str, ...] = ("Positive", "Neutral", "Frustrated", "Angry")

# Custom urgency lexicon: phrases common in support tickets that signal
# escalation. Values are negative deltas added to the compound score on a
# match (capped per text). Keep this list small and high-signal; larger
# lexicons drift toward sentiment-analysis-as-keywords.
URGENCY_LEXICON: dict[str, float] = {
    # Money / billing escalation
    "charged twice": -0.30,
    "double charged": -0.30,
    "double charge": -0.30,
    "overcharged": -0.20,
    "wrong charge": -0.20,
    "unauthorized charge": -0.35,
    "refund": -0.05,
    "lost money": -0.25,
    # Service disruption
    "unacceptable": -0.30,
    "ridiculous": -0.25,
    "outage": -0.20,
    "down for hours": -0.25,
    "completely down": -0.30,
    "data loss": -0.35,
    "data breach": -0.35,
    "lost data": -0.30,
    # Threats / escalation
    "lawsuit": -0.40,
    "legal action": -0.40,
    "cancel my subscription": -0.25,
    "cancel my account": -0.25,
    "switching to a competitor": -0.30,
    "switching to competitor": -0.25,
    "competitor": -0.10,
    # Time pressure
    "urgent": -0.15,
    "asap": -0.15,
    "immediately": -0.15,
    "right now": -0.10,
    # Profanity-light strong negatives
    "furious": -0.30,
    "infuriating": -0.25,
    "horrible": -0.20,
    "worst": -0.20,
    "terrible": -0.20,
    "garbage": -0.25,
    "scam": -0.30,
    "fraud": -0.35,
}

# How much to boost the compound score per urgency hit, capped below.
_URGENCY_BOOST_CAP: float = 0.60

# Label thresholds on the (possibly boosted) compound score.
_POS_THRESHOLD: float = 0.20
_NEG_FRU_THRESHOLD: float = -0.05
_NEG_ANG_THRESHOLD: float = -0.50

# Per-text max urgency hits before we declare escalation regardless of VADER.
_URGENCY_ESCALATION_HITS: int = 2
_URGENCY_ESCALATION_NEG: float = 0.35


@dataclass(frozen=True)
class SentimentScores:
    """Raw VADER scores plus the post-lexicon compound used for labeling."""

    neg: float
    neu: float
    pos: float
    compound: float
    urgency_hits: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "neg": float(self.neg),
            "neu": float(self.neu),
            "pos": float(self.pos),
            "compound": float(self.compound),
        }


@dataclass(frozen=True)
class SentimentResult:
    """Structured output of `SentimentAnalyzer.analyze`."""

    label: str
    scores: SentimentScores
    urgency_signals: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "scores": self.scores.to_dict(),
            "urgency_signals": list(self.urgency_signals),
        }


class SentimentAnalyzer:
    """VADER + custom urgency-lexicon sentiment analyzer.

    Instances are cheap to construct (one underlying VADER analyzer each)
    and safe to share across threads because VADER's lexicon is read-only
    after construction. For request paths, prefer a module-level singleton
    via `get_default_analyzer()` to avoid re-loading the lexicon.
    """

    def __init__(self, urgency_lexicon: dict[str, float] | None = None) -> None:
        self._vader = SentimentIntensityAnalyzer()
        self.urgency_lexicon: dict[str, float] = (
            dict(URGENCY_LEXICON) if urgency_lexicon is None else dict(urgency_lexicon)
        )

    def _vader_scores(self, text: str) -> dict[str, float]:
        return self._vader.polarity_scores(text)

    def urgency_signals(self, text: str) -> list[str]:
        """Return urgency phrases from the lexicon present in `text`."""
        if not text:
            return []
        lowered = text.lower()
        hits: list[str] = []
        for phrase in self.urgency_lexicon:
            if phrase in lowered:
                hits.append(phrase)
        return hits

    def negative_intensity(self, scores: SentimentScores) -> float:
        """Return the negative-intensity scalar used by the priority engine.

        Blends VADER's `neg` probability with a small bonus driven by
        urgency hits, so escalation-heavy tickets score higher than VADER
        alone would predict. Output is clamped to [0, 1].
        """
        bonus = min(0.10 * scores.urgency_hits, 0.30)
        value = float(scores.neg) + bonus
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _boost_compound(
        self, base_compound: float, urgency_hits: int, text_lower: str
    ) -> tuple[float, float]:
        """Apply per-phrase negative deltas and return (new_compound, total_delta)."""
        total_delta = 0.0
        if not urgency_hits:
            return base_compound, 0.0
        for phrase, delta in self.urgency_lexicon.items():
            if phrase in text_lower:
                total_delta += delta
        # Cap the magnitude so a flood of urgency phrases doesn't push the
        # score to nonsense territory.
        if total_delta < -_URGENCY_BOOST_CAP:
            total_delta = -_URGENCY_BOOST_CAP
        if total_delta > _URGENCY_BOOST_CAP:
            total_delta = _URGENCY_BOOST_CAP
        new = base_compound + total_delta
        if new < -1.0:
            new = -1.0
        if new > 1.0:
            new = 1.0
        return new, total_delta

    @staticmethod
    def _label_from(compound: float, neg: float, urgency_hits: int) -> str:
        if (
            compound < _NEG_ANG_THRESHOLD
            or (urgency_hits >= _URGENCY_ESCALATION_HITS and neg >= _URGENCY_ESCALATION_NEG)
        ):
            return "Angry"
        if compound < _NEG_FRU_THRESHOLD:
            return "Frustrated"
        if compound >= _POS_THRESHOLD:
            return "Positive"
        return "Neutral"

    def analyze(self, text: str) -> SentimentResult:
        """Return a `SentimentResult` for `text`.

        Empty / non-string inputs yield a Neutral result with zero scores so
        downstream code never has to special-case them.
        """
        if not isinstance(text, str) or not text.strip():
            empty = SentimentScores(neg=0.0, neu=1.0, pos=0.0, compound=0.0, urgency_hits=0)
            return SentimentResult(label="Neutral", scores=empty, urgency_signals=())

        vader = self._vader_scores(text)
        signals = self.urgency_signals(text)
        boosted_compound, _ = self._boost_compound(
            float(vader["compound"]), len(signals), text.lower()
        )
        scores = SentimentScores(
            neg=float(vader["neg"]),
            neu=float(vader["neu"]),
            pos=float(vader["pos"]),
            compound=boosted_compound,
            urgency_hits=len(signals),
        )
        label = self._label_from(scores.compound, scores.neg, scores.urgency_hits)
        return SentimentResult(label=label, scores=scores, urgency_signals=tuple(signals))

    def analyze_batch(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.analyze(t) for t in texts]


_DEFAULT: SentimentAnalyzer | None = None


def get_default_analyzer() -> SentimentAnalyzer:
    """Return a process-wide singleton `SentimentAnalyzer`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SentimentAnalyzer()
    return _DEFAULT


def analyze_text(text: str) -> SentimentResult:
    """Convenience wrapper using the default analyzer."""
    return get_default_analyzer().analyze(text)


__all__ = [
    "SENTIMENT_CLASSES",
    "URGENCY_LEXICON",
    "SentimentScores",
    "SentimentResult",
    "SentimentAnalyzer",
    "get_default_analyzer",
    "analyze_text",
]
