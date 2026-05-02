"""
idiosyncrasy_detector.py
Detects AI writing idiosyncrasies in prompts and LLM outputs.

Grounded in course readings:
  - "Can AI writing be salvaged?" (CHI 2024)
  - "People who frequently use ChatGPT ... are accurate detectors of AI text"
  - "Generative AI enhances individual creativity but reduces collective diversity"

Pattern catalogue informed by empirical AI-text-detection research on
overused vocabulary, structural monotony, and stylistic tells.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Any, Dict, List, Tuple


# ── Pattern Definitions ───────────────────────────────────────────────────────

@dataclass
class IdiosyncracyPattern:
    """One detectable AI stylistic pattern."""
    category: str
    label: str
    phrases: list[str]
    severity: float = 1.0


# ---------- PHRASE-BASED PATTERNS ----------

PATTERNS: list[IdiosyncracyPattern] = [

    # ── 1. Hedging / signposting ──
    IdiosyncracyPattern(
        category="hedging",
        label="Unnecessary hedging",
        severity=1.5,
        phrases=[
            "it's important to note that",
            "it is important to note that",
            "it's worth mentioning",
            "it is worth mentioning",
            "it should be noted that",
            "it's crucial to understand",
            "it is crucial to understand",
            "it's essential to recognize",
            "it is essential to recognize",
            "it bears noting that",
            "one should keep in mind",
            "it is important to remember",
            "it's worth highlighting",
            "it is worth highlighting",
            "it's important to consider",
            "it is important to consider",
            "it must be emphasized that",
            "this underscores the importance of",
            "this highlights the need for",
            "this serves as a reminder",
        ],
    ),

    # ── 2. Sycophantic / assistant-like ──
    IdiosyncracyPattern(
        category="sycophancy",
        label="Assistant-style phrasing",
        severity=2.0,
        phrases=[
            "i'd be happy to",
            "i'd be glad to",
            "let me help you",
            "let me explain",
            "let me break this down",
            "let me walk you through",
            "let's dive in",
            "let's break it down",
            "let's explore",
            "great question",
            "that's a great question",
            "that's an excellent question",
            "excellent question",
            "good question",
            "absolutely!",
            "of course!",
            "sure thing",
            "certainly!",
            "i hope this helps",
            "i hope that helps",
            "feel free to ask",
            "don't hesitate to",
            "happy to help",
            "glad you asked",
            "here's a comprehensive",
            "here is a comprehensive",
        ],
    ),

    # ── 3. Formulaic transitions ──
    IdiosyncracyPattern(
        category="transition",
        label="Formulaic transitions",
        severity=1.2,
        phrases=[
            "furthermore,",
            "moreover,",
            "in conclusion,",
            "additionally,",
            "in summary,",
            "to summarize,",
            "to sum up,",
            "it is also worth noting",
            "with that being said,",
            "having said that,",
            "that being said,",
            "on the other hand,",
            "in light of the above,",
            "in light of this,",
            "with this in mind,",
            "given the above,",
            "as mentioned earlier,",
            "as previously mentioned,",
            "as noted above,",
            "building on this,",
            "to elaborate,",
            "to put it simply,",
            "to be more specific,",
            "in other words,",
            "ultimately,",
            "by the same token,",
        ],
    ),

    # ── 4. Overqualification / vagueness ──
    IdiosyncracyPattern(
        category="overqualification",
        label="Vague overqualification",
        severity=1.3,
        phrases=[
            "it depends on various factors",
            "there are many aspects to consider",
            "this is a complex topic",
            "there are several perspectives",
            "this is a nuanced issue",
            "the answer depends on context",
            "there are many ways to approach",
            "it varies depending on",
            "this can vary widely",
            "there are multiple considerations",
            "there is no one-size-fits-all",
            "it's not a simple answer",
            "it is not a simple answer",
            "there are pros and cons",
            "both sides have merit",
            "the situation is complex",
            "it requires careful consideration",
            "there are trade-offs",
        ],
    ),

    # ── 5. Hollow filler / cliché openers ──
    IdiosyncracyPattern(
        category="filler",
        label="Hollow filler phrases",
        severity=1.4,
        phrases=[
            "in today's world",
            "in today's digital age",
            "in today's digital era",
            "in today's fast-paced world",
            "in today's rapidly evolving",
            "in the modern era",
            "in the digital age",
            "in the digital landscape",
            "throughout history",
            "since time immemorial",
            "in the ever-evolving landscape",
            "in the ever-changing landscape",
            "in this day and age",
            "as we all know",
            "as is well known",
            "needless to say",
            "it goes without saying",
            "plays a crucial role",
            "plays a vital role",
            "plays an important role",
            "plays a pivotal role",
            "plays a key role",
            "plays a significant role",
            "is of paramount importance",
            "is of utmost importance",
            "at the end of the day",
            "when it comes to",
            "when all is said and done",
            "in the grand scheme of things",
            "the bottom line is",
            "in the realm of",
            "in the world of",
            "in the landscape of",
            "at its core",
        ],
    ),

    # ── 6. Artificial parallelism / formulaic structure ──
    IdiosyncracyPattern(
        category="parallelism",
        label="Formulaic structural phrases",
        severity=1.0,
        phrases=[
            "first and foremost",
            "last but not least",
            "each and every",
            "above and beyond",
            "one of the most important",
            "without a doubt",
            "not only",  # "not only X but also Y" — very common AI pattern
            "it's not just",
            "it is not just",
            "whether you're a",
            "whether you are a",
            "from x to y",  # generic range pattern caught by regex below
        ],
    ),

    # ── 7. AI-signature vocabulary (verbs) ──
    IdiosyncracyPattern(
        category="ai_vocabulary",
        label="AI-signature verbs",
        severity=1.6,
        phrases=[
            "delve into",
            "delve deeper",
            "delving into",
            "delves into",
            "embark on",
            "embarking on",
            "embarks on",
            "unleash",
            "unleashing",
            "leverage",
            "leveraging",
            "leverages",
            "foster",
            "fostering",
            "fosters",
            "navigate",
            "navigating",
            "navigates",
            "underscore",
            "underscores",
            "underscoring",
            "streamline",
            "streamlining",
            "streamlines",
            "facilitate",
            "facilitating",
            "facilitates",
            "spearhead",
            "spearheading",
            "catalyze",
            "catalyzing",
            "revolutionize",
            "revolutionizing",
            "harness",
            "harnessing",
            "harnesses",
            "elevate",
            "elevating",
            "elevates",
            "empower",
            "empowering",
            "empowers",
            "optimize",
            "optimizing",
            "synergize",
            "synergizing",
            "illuminate",
            "illuminating",
            "illuminates",
            "demystify",
            "demystifying",
            "unpack",
            "unpacking",
            "unravel",
            "unraveling",
            "bolster",
            "bolstering",
            "bolsters",
            "dovetail",
            "dovetails",
            "epitomize",
            "epitomizes",
            "epitomizing",
            "galvanize",
            "galvanizing",
        ],
    ),

    # ── 8. AI-signature vocabulary (adjectives / nouns) ──
    IdiosyncracyPattern(
        category="ai_vocabulary",
        label="AI-signature descriptors",
        severity=1.5,
        phrases=[
            "multifaceted",
            "ever-evolving",
            "cutting-edge",
            "groundbreaking",
            "game-changing",
            "game-changer",
            "transformative",
            "holistic",
            "comprehensive",
            "nuanced",
            "robust",
            "pivotal",
            "paramount",
            "indispensable",
            "meticulous",
            "meticulously",
            "intricate",
            "intricacies",
            "myriad",
            "plethora",
            "tapestry",
            "symphony",
            "cornerstone",
            "linchpin",
            "bedrock",
            "landscape",  # metaphorical use
            "realm",
            "testament",
            "paradigm",
            "synergy",
            "synergies",
            "ecosystem",
            "stakeholder",
            "stakeholders",
            "actionable",
            "scalable",
            "seamless",
            "seamlessly",
            "unparalleled",
            "commendable",
            "noteworthy",
            "groundwork",
            "underpinnings",
            "interplay",
            "juxtaposition",
        ],
    ),

    # ── 9. Balanced contrast (AI hedging structure) ──
    IdiosyncracyPattern(
        category="balanced_contrast",
        label="Balanced contrast hedging",
        severity=1.3,
        phrases=[
            "while it is true that",
            "while this is true",
            "while this may be",
            "while there are",
            "although this is",
            "on one hand",
            "on the one hand",
            "it is equally important to",
            "it's equally important to",
            "at the same time,",
            "however, it is important",
            "however, it's important",
            "that said,",
            "be that as it may,",
            "notwithstanding,",
            "conversely,",
            "by contrast,",
        ],
    ),

    # ── 10. Conclusory grandstanding ──
    IdiosyncracyPattern(
        category="grandstanding",
        label="Conclusory grandstanding",
        severity=1.4,
        phrases=[
            "pave the way for",
            "paving the way for",
            "paves the way for",
            "shape the future",
            "shaping the future",
            "a step in the right direction",
            "stands as a testament",
            "serves as a testament",
            "serves as a reminder",
            "a beacon of",
            "the key takeaway",
            "the takeaway here is",
            "the crux of the matter",
            "marks a significant",
            "represents a significant",
            "heralds a new era",
            "ushering in a new",
            "a watershed moment",
            "redefine the way we",
            "redefining the way we",
            "the implications are profound",
            "the possibilities are endless",
            "only time will tell",
        ],
    ),
]


# ---------- WORD-LEVEL INDICATORS (regex-based) ----------
# These are single words that appear far more often in AI text than human text.
# Matched as whole words with word boundaries.

AI_INDICATOR_WORDS: list[tuple[str, float]] = [
    # (regex pattern, severity weight)
    (r"\bdelve\b", 2.0),
    (r"\btapestry\b", 2.0),
    (r"\bmultifaceted\b", 1.8),
    (r"\bholistic\b", 1.5),
    (r"\bpivotal\b", 1.5),
    (r"\bparamount\b", 1.5),
    (r"\bmeticulous(?:ly)?\b", 1.5),
    (r"\bseamless(?:ly)?\b", 1.3),
    (r"\bunparalleled\b", 1.5),
    (r"\bcommendable\b", 1.5),
    (r"\bnoteworthy\b", 1.3),
    (r"\bunderpinnings?\b", 1.5),
    (r"\binterplay\b", 1.5),
    (r"\bjuxtaposition\b", 1.5),
    (r"\bcornerstone\b", 1.5),
    (r"\blinchpin\b", 1.8),
    (r"\bbedrock\b", 1.5),
    (r"\bmyriad\b", 1.5),
    (r"\bplethora\b", 1.8),
    (r"\bsymphony\b", 1.5),
    (r"\brobust\b", 1.2),
    (r"\bintricate\b", 1.3),
    (r"\bnuanced\b", 1.3),
    (r"\bdaunting\b", 1.3),
    (r"\bvibrant\b", 1.2),
    (r"\bactionable\b", 1.5),
    (r"\bscalable\b", 1.3),
    (r"\bsynerg(?:y|ies|ize|izing)\b", 1.8),
    (r"\bparadigm\b", 1.5),
    (r"\becosystem\b", 1.2),
    (r"\bstakeholders?\b", 1.3),
    (r"\btransformative\b", 1.3),
    (r"\bgroundbreaking\b", 1.3),
    (r"\bgame-?chang(?:er|ing)\b", 1.5),
    (r"\bcutting-?edge\b", 1.3),
    (r"\bindispensable\b", 1.5),
    (r"\btestament\b", 1.3),
    (r"\blandscape\b", 1.0),  # lower — can be literal
    (r"\brealm\b", 1.2),
    (r"\bever-evolving\b", 1.8),
    (r"\bgroundwork\b", 1.3),
]

# Pre-compile for performance
_COMPILED_WORD_PATTERNS = [(re.compile(p, re.IGNORECASE), w) for p, w in AI_INDICATOR_WORDS]


# ── Core Detection ────────────────────────────────────────────────────────────

@dataclass
class PatternMatch:
    """A single pattern occurrence found in text."""
    category: str
    label: str
    phrase: str
    severity: float
    position: int


@dataclass
class IdiosyncracyReport:
    """Full analysis of one text."""
    ai_style_score: float
    total_matches: int
    matches_per_100_words: float
    pattern_matches: list[dict] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    category_labels: list[str] = field(default_factory=list)
    top_patterns: list[str] = field(default_factory=list)
    structural_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _check_structural_patterns(text: str, lines: list[str]) -> list[PatternMatch]:
    """Detect structural AI tells beyond phrase matching."""
    matches: list[PatternMatch] = []
    total_lines = max(len(lines), 1)
    non_empty = [l for l in lines if l.strip()]
    total_non_empty = max(len(non_empty), 1)

    # 1. Bullet-list dominance
    bullet_lines = [l for l in lines if re.match(r"^\s*[-•*]\s", l)]
    bullet_ratio = len(bullet_lines) / total_non_empty
    if len(bullet_lines) > 4 and bullet_ratio > 0.5:
        matches.append(PatternMatch(
            "structure", "List-heavy structure",
            f"({len(bullet_lines)}/{total_non_empty} lines are bullets)",
            1.0 if bullet_ratio > 0.7 else 0.6, 0,
        ))

    # 2. Numbered list dominance
    numbered_lines = [l for l in lines if re.match(r"^\s*\d+[\.\)]\s", l)]
    if len(numbered_lines) > 4 and len(numbered_lines) / total_non_empty > 0.4:
        matches.append(PatternMatch(
            "structure", "Numbered-list dominance",
            f"({len(numbered_lines)} numbered items)", 0.6, 0,
        ))

    # 3. Em-dash overuse (strong AI signal in 2024-2025)
    em_dash_count = text.count("—") + text.count("--")
    words = max(len(text.split()), 1)
    em_dash_per_100 = (em_dash_count / words) * 100
    if em_dash_count >= 3 and em_dash_per_100 > 1.0:
        matches.append(PatternMatch(
            "structure", "Em-dash overuse",
            f"({em_dash_count} em-dashes, {em_dash_per_100:.1f}/100 words)",
            1.5, 0,
        ))

    # 4. Low burstiness (uniform sentence lengths = AI monotone)
    sentences = re.split(r'[.!?]+', text)
    sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
    if len(sentence_lengths) >= 5:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        if avg_len > 0:
            variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
            cv = (variance ** 0.5) / avg_len  # coefficient of variation
            if cv < 0.25:  # very uniform = robotic
                matches.append(PatternMatch(
                    "structure", "Low burstiness (uniform sentences)",
                    f"(CV={cv:.2f}, avg={avg_len:.0f} words/sentence)",
                    1.2, 0,
                ))

    # 5. Heading-heavy structure (##, **, etc.)
    heading_lines = [l for l in lines if re.match(r"^\s*(#{1,4}\s|(\*\*).+\2\s*$)", l)]
    if len(heading_lines) > 3 and len(heading_lines) / total_non_empty > 0.15:
        matches.append(PatternMatch(
            "structure", "Heading-heavy formatting",
            f"({len(heading_lines)} headings)", 0.5, 0,
        ))

    # 6. Colon-intro pattern (AI loves "Key Point: explanation")
    colon_intros = [l for l in lines if re.match(r"^\s*\*?\*?[A-Z][^:]{2,30}:\s", l)]
    if len(colon_intros) > 3:
        matches.append(PatternMatch(
            "structure", "Colon-introduction pattern",
            f"({len(colon_intros)} label: content lines)", 0.5, 0,
        ))

    return matches


def detect_idiosyncrasies(text: str) -> IdiosyncracyReport:
    """Scan text for AI writing idiosyncrasies and return a structured report."""
    if not text or not text.strip():
        return IdiosyncracyReport(ai_style_score=0.0, total_matches=0, matches_per_100_words=0.0)

    text_lower = text.lower()
    word_count = max(len(text.split()), 1)
    matches: list[PatternMatch] = []

    # Phase 1: phrase-based matching
    for pattern in PATTERNS:
        for phrase in pattern.phrases:
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break
                matches.append(PatternMatch(
                    category=pattern.category, label=pattern.label,
                    phrase=phrase, severity=pattern.severity, position=pos,
                ))
                start = pos + len(phrase)

    # Phase 2: word-level indicators (regex, deduplicated against phrase matches)
    matched_positions: set[int] = {m.position for m in matches}
    for compiled_re, severity in _COMPILED_WORD_PATTERNS:
        for m in compiled_re.finditer(text):
            if m.start() not in matched_positions:
                matches.append(PatternMatch(
                    category="ai_vocabulary", label="AI-signature word",
                    phrase=m.group().lower(), severity=severity, position=m.start(),
                ))
                matched_positions.add(m.start())

    # Phase 3: structural analysis
    lines = text.strip().split("\n")
    structural = _check_structural_patterns(text, lines)
    matches.extend(structural)

    # Aggregate
    category_counts: dict[str, int] = {}
    for match in matches:
        category_counts[match.category] = category_counts.get(match.category, 0) + 1

    # Scoring: weighted density, stricter calibration
    weighted_sum = sum(m.severity for m in matches)
    density = (weighted_sum / word_count) * 100
    # Calibrated: 3 matches in 100 words ≈ 30, 6 matches ≈ 55, 10+ ≈ 80+
    ai_style_score = min(100.0, round(density * 8, 1))

    # Top pattern labels
    label_counts: dict[str, int] = {}
    for match in matches:
        label_counts[match.label] = label_counts.get(match.label, 0) + 1
    top_patterns = sorted(label_counts.keys(), key=lambda l: label_counts[l], reverse=True)[:6]

    structural_flags = [m.phrase for m in structural]

    return IdiosyncracyReport(
        ai_style_score=ai_style_score,
        total_matches=len(matches),
        matches_per_100_words=round((len(matches) / word_count) * 100, 2),
        pattern_matches=[
            {"category": m.category, "label": m.label, "phrase": m.phrase, "severity": m.severity}
            for m in matches[:30]
        ],
        category_counts=category_counts,
        category_labels=sorted(category_counts.keys()),
        top_patterns=top_patterns,
        structural_flags=structural_flags,
    )


def compute_ai_style_score(text: str) -> float:
    """Return a single 0–100 AI-ness score."""
    return detect_idiosyncrasies(text).ai_style_score


# ── Comparison Utilities ──────────────────────────────────────────────────────

@dataclass
class IdiosyncracyComparison:
    """Comparison of AI-ness between two texts."""
    before_score: float
    after_score: float
    reduction_pct: float
    categories_fixed: list[str]
    categories_introduced: list[str]
    before_report: dict
    after_report: dict

    def to_dict(self) -> dict:
        return asdict(self)


def compare_idiosyncrasies(before: str, after: str) -> IdiosyncracyComparison:
    """Compare AI-ness of two texts and classify what was fixed."""
    before_report = detect_idiosyncrasies(before)
    after_report = detect_idiosyncrasies(after)

    reduction = 0.0
    if before_report.ai_style_score > 0:
        reduction = round(
            ((before_report.ai_style_score - after_report.ai_style_score) / before_report.ai_style_score) * 100, 1
        )

    before_cats = set(before_report.category_counts.keys())
    after_cats = set(after_report.category_counts.keys())
    categories_fixed = [
        cat for cat in before_cats
        if after_report.category_counts.get(cat, 0) < before_report.category_counts.get(cat, 0)
    ]
    categories_introduced = list(after_cats - before_cats)

    return IdiosyncracyComparison(
        before_score=before_report.ai_style_score,
        after_score=after_report.ai_style_score,
        reduction_pct=reduction,
        categories_fixed=categories_fixed,
        categories_introduced=categories_introduced,
        before_report=before_report.to_dict(),
        after_report=after_report.to_dict(),
    )


# ── Candidate Diversity Analysis ─────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard_distance(a: str, b: str) -> float:
    a_tok, b_tok = _tokenize(a), _tokenize(b)
    union = a_tok | b_tok
    if not union:
        return 0.0
    return 1.0 - len(a_tok & b_tok) / len(union)


def _structural_distance(a: str, b: str) -> float:
    """Measure structural divergence: line count ratio, bullet ratio, avg sentence length."""
    def _features(text: str) -> tuple:
        lines = [l for l in text.strip().split("\n") if l.strip()]
        bullets = sum(1 for l in lines if re.match(r"^\s*[-•*\d]+[\.\)]\s", l))
        sentences = re.split(r'[.!?]+', text)
        avg_sent = sum(len(s.split()) for s in sentences if len(s.split()) > 2) / max(len(sentences), 1)
        return (len(lines), bullets / max(len(lines), 1), avg_sent)

    fa, fb = _features(a), _features(b)
    diffs = []
    for va, vb in zip(fa, fb):
        denom = max(abs(va), abs(vb), 1)
        diffs.append(abs(va - vb) / denom)
    return sum(diffs) / len(diffs)


def candidate_diversity_report(
    candidate_a: str, candidate_b: str, candidate_c: str
) -> Dict[str, Any]:
    """
    Analyze whether three rewriter candidates converge on
    similar AI patterns or genuinely diverge.
    Uses both token-level Jaccard distance AND structural distance
    AND idiosyncrasy-pattern divergence for a multi-dimensional view.
    """
    reports = {
        "A": detect_idiosyncrasies(candidate_a),
        "B": detect_idiosyncrasies(candidate_b),
        "C": detect_idiosyncrasies(candidate_c),
    }
    candidates = {"A": candidate_a, "B": candidate_b, "C": candidate_c}

    scores = {k: r.ai_style_score for k, r in reports.items()}
    all_categories: dict[str, set[str]] = {}
    for key, report in reports.items():
        for cat in report.category_labels:
            all_categories.setdefault(cat, set()).add(key)

    shared = [cat for cat, agents in all_categories.items() if len(agents) == 3]
    unique = [cat for cat, agents in all_categories.items() if len(agents) == 1]

    # Multi-dimensional diversity
    keys = ["A", "B", "C"]
    pairs = list(combinations(keys, 2))

    jaccard_scores = [_jaccard_distance(candidates[a], candidates[b]) for a, b in pairs]
    structural_scores = [_structural_distance(candidates[a], candidates[b]) for a, b in pairs]
    style_diffs = [abs(scores[a] - scores[b]) for a, b in pairs]

    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    avg_structural = sum(structural_scores) / len(structural_scores) if structural_scores else 0
    avg_style_diff = sum(style_diffs) / len(style_diffs) if style_diffs else 0

    score_spread = max(scores.values()) - min(scores.values()) if scores else 0.0

    # Combined convergence assessment
    convergence_label = (
        "high" if avg_jaccard < 0.3 and score_spread < 8 and len(shared) >= 2
        else "moderate" if avg_jaccard < 0.5 or score_spread < 20
        else "diverse"
    )

    return {
        "scores": scores,
        "score_spread": round(score_spread, 1),
        "shared_categories": shared,
        "unique_categories": unique,
        "convergence_label": convergence_label,
        "avg_token_diversity": round(avg_jaccard * 100, 1),
        "avg_structural_diversity": round(avg_structural * 100, 1),
        "avg_style_divergence": round(avg_style_diff, 1),
        "per_candidate": {k: r.to_dict() for k, r in reports.items()},
    }
