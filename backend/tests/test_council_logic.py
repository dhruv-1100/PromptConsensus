"""
Tests for the pure council computations: ranking extraction, consensus
diagnostics, label handling and the chairman's bare-label guard.
These make no network calls and need no API key.
"""
import pytest

from agents.council import (
    _consensus_diagnostics,
    _index_to_label,
    _looks_like_bare_label,
    _parse_ranking,
    _response_labels,
)


THREE = ["Response A", "Response B", "Response C"]


class TestParseRanking:
    def test_reads_the_final_ranking_block(self):
        text = (
            "Response A is thorough but unstructured.\n"
            "Response B is well structured.\n\n"
            "FINAL RANKING:\n1. Response B\n2. Response C\n3. Response A"
        )
        assert _parse_ranking(text, THREE) == ["Response B", "Response C", "Response A"]

    def test_prefers_the_final_ranking_over_earlier_mentions(self):
        text = (
            "1. Response A looks best at first glance\n"
            "2. Response B\n3. Response C\n\n"
            "FINAL RANKING:\n1. Response C\n2. Response B\n3. Response A"
        )
        assert _parse_ranking(text, THREE)[0] == "Response C"

    @pytest.mark.parametrize(
        "block",
        [
            "FINAL RANKING:\n1) Response B\n2) Response A\n3) Response C",
            "FINAL RANKING\n1 - Response B\n2 - Response A\n3 - Response C",
            "final ranking:\n* 1. Response B\n* 2. Response A\n* 3. Response C",
            "**FINAL RANKING:**\n1. Response B\n2. Response A\n3. Response C",
        ],
    )
    def test_tolerates_formatting_variants(self, block):
        assert _parse_ranking(block, THREE) == ["Response B", "Response A", "Response C"]

    def test_falls_back_to_numbered_list_without_a_header(self):
        assert _parse_ranking("1. Response C\n2. Response A\n3. Response B", THREE) == [
            "Response C",
            "Response A",
            "Response B",
        ]

    def test_falls_back_to_mention_order(self):
        text = "I prefer Response B, then Response A, and finally Response C."
        assert _parse_ranking(text, THREE) == ["Response B", "Response A", "Response C"]

    def test_drops_duplicates_keeping_first_position(self):
        text = "FINAL RANKING:\n1. Response B\n2. Response B\n3. Response A\n4. Response C"
        assert _parse_ranking(text, THREE) == ["Response B", "Response A", "Response C"]

    def test_ignores_labels_outside_the_expected_set(self):
        text = "FINAL RANKING:\n1. Response D\n2. Response B\n3. Response A\n4. Response C"
        assert _parse_ranking(text, THREE) == ["Response B", "Response A", "Response C"]

    def test_returns_empty_when_the_count_does_not_match(self):
        # A partial ranking is worse than none: run_pipeline rejects the review
        # rather than aggregating over a guess.
        assert _parse_ranking("FINAL RANKING:\n1. Response B", THREE) == []

    def test_returns_empty_on_unusable_text(self):
        assert _parse_ranking("I cannot comply with this request.", THREE) == []

    def test_handles_double_letter_labels(self):
        labels = _response_labels(27)
        text = "FINAL RANKING:\n" + "\n".join(
            f"{i}. {label}" for i, label in enumerate(reversed(labels), start=1)
        )
        assert _parse_ranking(text, labels) == list(reversed(labels))


class TestLabels:
    @pytest.mark.parametrize(
        "index,expected",
        [(0, "A"), (1, "B"), (25, "Z"), (26, "AA"), (27, "AB"), (51, "AZ"), (52, "BA")],
    )
    def test_spreadsheet_style_labels(self, index, expected):
        assert _index_to_label(index) == expected

    def test_rejects_negative_index(self):
        with pytest.raises(ValueError):
            _index_to_label(-1)

    def test_response_labels_requires_at_least_one(self):
        with pytest.raises(ValueError):
            _response_labels(0)


def _reviews(*rankings):
    return [
        {"reviewer": f"R{i}", "model": "m", "evaluation": "ok", "parsed_ranking": list(r)}
        for i, r in enumerate(rankings)
    ]


def _aggregate(*labels):
    return [
        {"label": label, "candidate": label[-1], "average_rank": float(i + 1), "votes": 3}
        for i, label in enumerate(labels)
    ]


class TestConsensusDiagnostics:
    def test_unanimous_ranking_is_full_consensus(self):
        d = _consensus_diagnostics(_reviews(THREE, THREE, THREE), _aggregate(*THREE))
        assert d["first_place_support_pct"] == 100.0
        assert d["reviewer_agreement_pct"] == 100.0
        assert d["consensus_strength_pct"] == 100.0
        assert d["consensus_label"] == "high"
        assert d["is_unanimous_winner"] is True
        assert d["needs_human_review"] is False

    def test_total_disagreement_is_weak_consensus(self):
        d = _consensus_diagnostics(
            _reviews(
                ["Response A", "Response B", "Response C"],
                ["Response B", "Response C", "Response A"],
                ["Response C", "Response A", "Response B"],
            ),
            _aggregate(*THREE),
        )
        assert d["consensus_label"] == "weak"
        assert d["needs_human_review"] is True
        assert d["is_unanimous_winner"] is False

    def test_split_vote_lands_between_the_extremes(self):
        d = _consensus_diagnostics(
            _reviews(THREE, THREE, ["Response B", "Response A", "Response C"]),
            _aggregate(*THREE),
        )
        assert d["first_place_support_pct"] == pytest.approx(66.7, abs=0.1)
        assert 0 < d["consensus_strength_pct"] < 100
        assert d["consensus_label"] in {"moderate", "high"}

    def test_winner_margin_is_the_rank_gap_to_the_runner_up(self):
        aggregate = [
            {"label": "Response A", "candidate": "A", "average_rank": 1.33, "votes": 3},
            {"label": "Response B", "candidate": "B", "average_rank": 2.0, "votes": 3},
        ]
        d = _consensus_diagnostics(_reviews(THREE, THREE, THREE), aggregate)
        assert d["winner_margin"] == pytest.approx(0.67)

    def test_empty_aggregate_flags_human_review(self):
        d = _consensus_diagnostics([], [])
        assert d["winner_label"] is None
        assert d["consensus_strength_pct"] == 0.0
        assert d["needs_human_review"] is True

    def test_single_candidate_does_not_divide_by_zero(self):
        d = _consensus_diagnostics(_reviews(["Response A"]), _aggregate("Response A"))
        assert d["winner_label"] == "Response A"
        assert d["winner_margin"] == pytest.approx(-1.0)  # no runner-up to compare against


class TestBareLabelGuard:
    @pytest.mark.parametrize(
        "reply",
        [
            "Response Z",
            "Candidate C",
            "**Candidate C**",
            "  response aa  ",
            "The winner is Response A",
            "Response B.",
            "",
            "   ",
            "Use Candidate A instead",  # too short to be a usable prompt
        ],
    )
    def test_detects_a_reply_that_is_only_a_label(self, reply):
        assert _looks_like_bare_label(reply) is True

    @pytest.mark.parametrize(
        "reply",
        [
            # Regression: these were discarded before, because the guard matched the
            # substrings "Candidate " and "Response " anywhere in the prompt.
            "You are a hiring manager reviewing an application. The Candidate submitted the "
            "attached resume; score each competency from one to five and justify every score "
            "with a direct quotation from the document.",
            "Draft a customer support Response that acknowledges the delay, explains the root "
            "cause in plain language, and offers two concrete remedies the reader can choose "
            "between without contacting us again.",
            "Summarise the trial results in four labelled sections: methodology, key findings, "
            "limitations and remaining uncertainty. Flag every claim that the data cannot "
            "support directly.",
        ],
    )
    def test_keeps_a_real_synthesised_prompt(self, reply):
        assert _looks_like_bare_label(reply) is False
