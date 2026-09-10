"""Consensus scoring and the review-routing decision table.

Author: Perijn
Summary: Fuses the four OCR engine outputs with Consensus Entropy and routes each page to accept or a typed review queue.
Usage: Imported by app.main. Thresholds arrive as constructor arguments sourced from environment variables.

The table encodes one insight that a plain agreement score misses: the two classical
engines and the two VLMs fail in different ways. When the classical engines agree
with each other but disagree with MinerU, the classical pair is the more credible
witness, because a 5M-parameter non-autoregressive detector cannot invent fluent
prose. That specific shape is the hallucination signature and gets top priority.

Thresholds are deliberately environment-driven. They must be calibrated against
this corpus before being trusted; the defaults are starting points, not verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .engines import EngineResult, numeric_tokens

# Verdicts, ordered by review priority. Lower priority number means review first.
ACCEPT = "accept"
ACCEPT_WEIGHTED = "accept_weighted"
REVIEW_HALLUCINATION = "review_hallucination"
REVIEW_HARD_PAGE = "review_hard_page"
REVIEW_ENTROPY = "review_entropy"

_PRIORITY = {
    REVIEW_HALLUCINATION: 0,
    REVIEW_HARD_PAGE: 1,
    REVIEW_ENTROPY: 2,
    ACCEPT_WEIGHTED: 3,
    ACCEPT: 4,
}


def similarity(left: str, right: str) -> float:
    """Return a normalized agreement score in ``[0, 1]`` between two texts.

    Whitespace is collapsed first so that the spaced-letter artifact (``c a r b o n
    a t e``) does not read as agreement with the correct ``carbonate``. Sequence
    matcher is used rather than a raw edit distance so the score is bounded without
    pulling in a heavier dependency for the CPU-only fallback path.

    Args:
        left: First OCR text.
        right: Second OCR text.

    Returns:
        Agreement in ``[0, 1]``; 1 means identical after normalization.
    """
    a = " ".join(left.split())
    b = " ".join(right.split())
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class FusionConfig:
    """Thresholds governing the decision table.

    Args:
        agree: Pairwise similarity at or above which two engines are said to agree.
        classical_pair_agree: Minimum similarity between the two classical engines
            for them to count as a mutually corroborating witness pair.
        classical_pair_hard: Below this, the classical engines disagree with each
            other and the page is treated as genuinely unreadable.
        entropy_accept: Maximum Consensus Entropy eligible for outright acceptance.
        entropy_review_floor: Above this, review is forced regardless of agreement.
    """

    agree: float = 0.92
    classical_pair_agree: float = 0.90
    classical_pair_hard: float = 0.75
    entropy_accept: float = 0.25
    entropy_review_floor: float = 0.60


@dataclass
class FusionResult:
    """The fused verdict for one page.

    Attributes:
        verdict: One of the module-level verdict constants.
        review: Whether a human must look at this page.
        priority: Review queue ordering; 0 is most urgent.
        text: The selected text, the consensus medoid when one exists.
        entropy: Consensus Entropy over the engine outputs, or ``None`` when fewer
            than two engines produced output.
        pairwise: Mapping of ``"a|b"`` to similarity for every engine pair.
        ungrounded: Numeric tokens present in the selected text but absent from
            every classical engine output.
        reasons: Human-readable rules that fired, in evaluation order.
        engines: Per-engine status and normalized OCR blocks for the audit trail.
    """

    verdict: str
    review: bool
    priority: int
    text: str
    entropy: float | None
    pairwise: dict[str, float] = field(default_factory=dict)
    ungrounded: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    engines: dict[str, Any] = field(default_factory=dict)


class FusionEngine:
    """Applies Consensus Entropy and the decision table to a page's engine outputs."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        """Create a fusion engine.

        Args:
            config: Threshold set. Defaults to ``FusionConfig()``.
        """
        self.config = config or FusionConfig()
        self._consensus = _load_consensus_entropy()

    def fuse(
        self, results: list[EngineResult], *, allow_auto_accept: bool = True
    ) -> FusionResult:
        """Fuse per-engine results into one verdict.

        Args:
            results: Every engine result for the page, including failures.
            allow_auto_accept: Whether this request ran the complete validation
                ensemble. Diagnostic subsets always return a review verdict.

        Returns:
            The verdict, with the audit trail needed to act on it.
        """
        usable = [r for r in results if r.text.strip() and not r.error]
        status = {
            r.engine: {
                "kind": r.kind,
                "ok": not r.error,
                "characters": len(r.text),
                "confidence": r.confidence,
                "blocks": r.blocks,
                "error": r.error,
            }
            for r in results
        }

        if not usable:
            return FusionResult(
                verdict=REVIEW_ENTROPY,
                review=True,
                priority=0,
                text="",
                entropy=None,
                reasons=["no engine produced any text"],
                engines=status,
            )

        pairwise = _pairwise(usable)
        entropy = self._entropy([r.text for r in usable])
        selected = self._select(usable)

        mineru = _by_name(usable, "mineru")
        classical = [r for r in usable if r.kind == "classical"]
        classical_pair_sim = (
            similarity(classical[0].text, classical[1].text)
            if len(classical) >= 2
            else None
        )
        mineru_vs_classical = (
            min(similarity(mineru.text, c.text) for c in classical)
            if mineru and classical
            else None
        )
        mineru_best_classical = (
            max(similarity(mineru.text, c.text) for c in classical)
            if mineru and classical
            else None
        )

        selected_ungrounded = _ungrounded_numbers(selected, classical)
        mineru_ungrounded = (
            _ungrounded_numbers(mineru, classical) if mineru is not None else []
        )
        ungrounded = list(dict.fromkeys(selected_ungrounded + mineru_ungrounded))
        reasons: list[str] = []

        if ungrounded:
            reasons.append(
                f"{len(ungrounded)} numeric token(s) in the selected or MinerU "
                "parse appear in no classical engine output"
            )

        # R0: A MinerU number that a mutually corroborating classical pair never
        # saw is never eligible for automatic acceptance. A single invented dosage
        # has negligible whole-page edit distance, so pairwise similarity alone
        # cannot safely detect this failure mode. When the classical pair itself
        # disagrees, preserve the R1 hard-page verdict: neither is a trusted witness.
        if not allow_auto_accept:
            reasons.append(
                "partial engine selection cannot produce a validated acceptance"
            )
            verdict = REVIEW_ENTROPY
        elif len(usable) != len(results):
            reasons.append("one or more validation engines failed or produced no text")
            verdict = REVIEW_ENTROPY
        elif ungrounded and (
            classical_pair_sim is None
            or classical_pair_sim >= self.config.classical_pair_agree
        ):
            reasons.append("ungrounded numeric value forces high-priority review")
            verdict = REVIEW_HALLUCINATION
        else:
            verdict = self._decide(
                classical_pair_sim=classical_pair_sim,
                mineru_vs_classical=mineru_vs_classical,
                mineru_best_classical=mineru_best_classical,
                pairwise=pairwise,
                entropy=entropy,
                engine_count=len(usable),
                reasons=reasons,
            )

        review = verdict != ACCEPT and verdict != ACCEPT_WEIGHTED
        return FusionResult(
            verdict=verdict,
            review=review,
            priority=_PRIORITY[verdict],
            text=selected.text,
            entropy=entropy,
            pairwise=pairwise,
            ungrounded=ungrounded,
            reasons=reasons,
            engines=status,
        )

    def _decide(
        self,
        *,
        classical_pair_sim: float | None,
        mineru_vs_classical: float | None,
        mineru_best_classical: float | None,
        pairwise: dict[str, float],
        entropy: float | None,
        engine_count: int,
        reasons: list[str],
    ) -> str:
        """Evaluate the decision table in priority order.

        Args:
            classical_pair_sim: Agreement between the two classical engines.
            mineru_vs_classical: MinerU's worst agreement against any classical engine.
            mineru_best_classical: MinerU's best agreement against a classical engine.
            pairwise: All pairwise engine similarities.
            entropy: Consensus Entropy, or ``None`` when underdetermined.
            engine_count: Engines that produced text.
            reasons: Accumulator appended to for each rule that fires.

        Returns:
            The verdict constant.
        """
        cfg = self.config

        # R1: The classical engines disagree with each other. Neither can corroborate
        # the other, so nothing downstream is verifiable. This outranks the
        # hallucination rule because a page the classical pair cannot read gives no
        # witness to trust.
        if (
            classical_pair_sim is not None
            and classical_pair_sim < cfg.classical_pair_hard
        ):
            reasons.append(
                f"classical engines disagree with each other "
                f"({classical_pair_sim:.3f} < {cfg.classical_pair_hard})"
            )
            return REVIEW_HARD_PAGE

        # R2: The classical pair corroborates itself but not MinerU. Two independent
        # non-generative engines agreeing on something MinerU did not say is the
        # hallucination signature.
        if (
            classical_pair_sim is not None
            and classical_pair_sim >= cfg.classical_pair_agree
            and mineru_vs_classical is not None
            and mineru_vs_classical < cfg.agree
        ):
            reasons.append(
                f"classical pair agrees ({classical_pair_sim:.3f}) but MinerU "
                f"diverges ({mineru_vs_classical:.3f} < {cfg.agree})"
            )
            return REVIEW_HALLUCINATION

        # R3: High Consensus Entropy always means the outputs contain too much
        # divergent information for the weighted-accept path below.
        if entropy is not None and entropy >= cfg.entropy_review_floor:
            reasons.append(
                f"consensus entropy {entropy:.3f} exceeds review floor "
                f"{cfg.entropy_review_floor}"
            )
            return REVIEW_ENTROPY

        # R4: Unanimous agreement across every live engine, with low entropy.
        if engine_count >= 3 and pairwise and min(pairwise.values()) >= cfg.agree:
            if entropy is None or entropy <= cfg.entropy_accept:
                reasons.append(f"all {engine_count} engines agree above {cfg.agree}")
                return ACCEPT
            reasons.append(
                f"pairwise agreement but entropy {entropy:.3f} exceeds "
                f"{cfg.entropy_accept}"
            )
            return REVIEW_ENTROPY

        # R5: MinerU agrees with at least one classical engine.
        if mineru_best_classical is not None and mineru_best_classical >= cfg.agree:
            reasons.append("MinerU corroborated by at least one classical engine")
            return ACCEPT_WEIGHTED

        # R5: Not enough corroboration to accept, not enough signal to classify.
        reasons.append("insufficient corroboration across live engines")
        return REVIEW_ENTROPY

    def _entropy(self, texts: list[str]) -> float | None:
        """Compute Consensus Entropy, or ``None`` when underdetermined.

        Args:
            texts: Engine outputs for one page.

        Returns:
            Entropy as a float, or ``None`` with fewer than two texts.
        """
        if len(texts) < 2 or self._consensus is None:
            return None
        calculate = self._consensus
        value = calculate(texts, task_type="ocr")
        return _as_scalar(value)

    def _select(self, usable: list[EngineResult]) -> EngineResult:
        """Pick the consensus medoid, falling back to MinerU.

        Args:
            usable: Engines that produced text.

        Returns:
            The engine result closest to the center of the others; MinerU when only
            one engine is live, since it is the designated primary parser.
        """
        if len(usable) == 1:
            return usable[0]

        mineru = _by_name(usable, "mineru")
        best: EngineResult | None = None
        best_score = -1.0
        for candidate in usable:
            others = [o for o in usable if o is not candidate]
            if not others:
                continue
            score = sum(similarity(candidate.text, o.text) for o in others) / len(
                others
            )
            if score > best_score:
                best_score = score
                best = candidate
        if best is None:
            return mineru or usable[0]
        # On a tie, prefer MinerU: it owns reading order and table structure that
        # the classical engines flatten into a single text stream.
        if mineru is not None:
            mineru_score = sum(
                similarity(mineru.text, o.text) for o in usable if o is not mineru
            ) / max(1, len(usable) - 1)
            if mineru_score >= best_score - 1e-9:
                return mineru
        return best


def _pairwise(results: list[EngineResult]) -> dict[str, float]:
    """Compute similarity for every unordered engine pair."""
    out: dict[str, float] = {}
    for i, left in enumerate(results):
        for right in results[i + 1 :]:
            out[f"{left.engine}|{right.engine}"] = similarity(left.text, right.text)
    return out


def _by_name(results: list[EngineResult], name: str) -> EngineResult | None:
    """Return the result for one engine name, when present."""
    for result in results:
        if result.engine == name:
            return result
    return None


def _ungrounded_numbers(
    selected: EngineResult, classical: list[EngineResult]
) -> list[str]:
    """Find numeric tokens in the parse that no classical engine saw.

    A dosage or identifier present in the VLM output but absent from every
    non-generative engine is the concrete form of the invented-value failure. This
    check is structural rather than statistical, so it carries no false-alarm cost
    from model disagreement.

    Args:
        selected: The chosen output text.
        classical: Classical engine results for the same page.

    Returns:
        Numeric tokens with no classical corroboration. Empty when no classical
        engine ran, since there is then nothing to ground against.
    """
    if not classical:
        return []
    corpus = " ".join(c.text for c in classical)
    grounded = set(numeric_tokens(corpus))
    return [t for t in numeric_tokens(selected.text) if t not in grounded]


def _as_scalar(value: Any) -> float | None:
    """Reduce a possibly-sequence consensus metric to a single float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        last = value[-1]
        return float(last) if isinstance(last, (int, float)) else None
    return None


def _load_consensus_entropy() -> Any:
    """Import the Consensus Entropy calculator or fail loudly.

    The metric is load-bearing for the routing decision. Silently substituting a
    different score would change the meaning of every verdict emitted, so an
    import failure raises here rather than degrading quietly.
    """
    try:
        from consensus_entropy import calculate_consensus_entropy
    except ImportError as exc:  # pragma: no cover - packaging boundary.
        raise RuntimeError(
            "consensus-entropy is required for OCR routing and is not installed. "
            "Install it with: pip install consensus-entropy"
        ) from exc
    return calculate_consensus_entropy
