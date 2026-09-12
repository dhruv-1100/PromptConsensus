"""
Tests for the pre-execution safety heuristics and the tolerant JSON extraction
used to read structured model output.
"""
import pytest

from live_mode_utils import coerce_message_content, extract_prompt_and_perspective, parse_json_response
from safety_checks import run_safety_checks


LONG_GENERIC_PROMPT = " ".join(["word"] * 60)


class TestSafetyChecks:
    def test_short_prompt_is_flagged(self):
        report = run_safety_checks("Summarise this.", domain="general")
        titles = [check["title"] for check in report["checks"]]
        assert "Prompt is still quite short" in titles
        assert report["risk_level"] == "low"
        assert report["requires_acknowledgement"] is False

    def test_unstructured_output_is_flagged(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT, domain="general")
        assert "Output structure is underspecified" in [c["title"] for c in report["checks"]]

    def test_structure_keyword_clears_that_check(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT + " use these sections", domain="general")
        assert "Output structure is underspecified" not in [c["title"] for c in report["checks"]]

    def test_unresolved_intent_gaps_are_surfaced(self):
        report = run_safety_checks(
            LONG_GENERIC_PROMPT,
            domain="general",
            intent={"missing_info": ["patient age", "current medications"]},
        )
        gap_check = next(c for c in report["checks"] if "unresolved gaps" in c["title"])
        assert "patient age" in gap_check["action"]

    def test_medication_content_without_guardrails_is_high_risk(self):
        report = run_safety_checks(
            "Write a discharge summary listing every medication and dose in a table, "
            "with follow-up instructions reviewed by the treating provider. " + LONG_GENERIC_PROMPT,
            domain="healthcare",
        )
        assert report["risk_level"] == "high"
        assert report["requires_acknowledgement"] is True
        assert "Medication content lacks verification guardrails" in [
            c["title"] for c in report["checks"]
        ]

    def test_medication_guardrail_phrase_clears_the_high_finding(self):
        report = run_safety_checks(
            "Write a discharge summary listing medications and doses. Use only provided "
            "medication facts and mark unknown values. Require review by a licensed clinician "
            "and include follow-up red-flag instructions. " + LONG_GENERIC_PROMPT,
            domain="healthcare",
        )
        assert report["risk_level"] != "high"

    def test_healthcare_medium_findings_require_acknowledgement(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT + " with sections", domain="healthcare")
        assert report["risk_level"] == "medium"
        assert report["requires_acknowledgement"] is True

    def test_legal_prompt_without_jurisdiction_is_high_risk(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT + " in sections", domain="legal")
        assert report["risk_level"] == "high"
        assert "Jurisdiction is missing" in [c["title"] for c in report["checks"]]

    def test_research_prompt_wants_citations_and_limitations(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT + " in sections", domain="research")
        titles = [c["title"] for c in report["checks"]]
        assert "Evidence requirements are missing" in titles
        assert "Uncertainty handling is missing" in titles
        # research is sensitive, but medium findings there do not force an ack
        assert report["requires_acknowledgement"] is False

    def test_education_prompt_wants_a_learner_level(self):
        report = run_safety_checks(LONG_GENERIC_PROMPT + " in sections", domain="education")
        assert "Learner level is not explicit" in [c["title"] for c in report["checks"]]

    @pytest.mark.parametrize("domain", ["healthcare", "legal", "research", "education"])
    def test_sensitive_domains_are_labelled(self, domain):
        assert run_safety_checks(LONG_GENERIC_PROMPT, domain=domain)["sensitive_domain"] is True

    def test_general_domain_is_not_sensitive(self):
        assert run_safety_checks(LONG_GENERIC_PROMPT, domain="general")["sensitive_domain"] is False

    @pytest.mark.parametrize("domain", ["", None, "  HEALTHCARE  "])
    def test_domain_is_normalised(self, domain):
        report = run_safety_checks(LONG_GENERIC_PROMPT, domain=domain)
        assert report["domain"] in {"general", "healthcare"}

    def test_report_counts_words(self):
        report = run_safety_checks("one two three", domain="general", raw_query="a b")
        assert report["checked_prompt_length"] == 3
        assert report["raw_query_length"] == 2

    def test_empty_prompt_does_not_crash(self):
        report = run_safety_checks("", domain="healthcare")
        assert report["risk_level"] in {"low", "medium", "high"}


class TestParseJsonResponse:
    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_wrapped_json(self):
        assert parse_json_response('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    def test_trailing_comma_is_repaired(self):
        assert parse_json_response('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_smart_quotes_are_repaired(self):
        assert parse_json_response('{“a”: 1}') == {"a": 1}

    def test_python_literal_fallback(self):
        assert parse_json_response("{'a': 1, 'b': None}") == {"a": 1, "b": None}

    @pytest.mark.parametrize("text", ["", "   ", "I cannot help with that.", "{unclosed"])
    def test_unparseable_text_raises(self, text):
        with pytest.raises(ValueError):
            parse_json_response(text)


class TestCoerceMessageContent:
    def test_string_passthrough(self):
        assert coerce_message_content("hello") == "hello"

    def test_joins_content_blocks(self):
        blocks = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
        assert coerce_message_content(blocks) == "part one\npart two"

    def test_reads_alternate_block_keys(self):
        assert coerce_message_content([{"type": "output_text", "content": "hi"}]) == "hi"

    def test_dict_content(self):
        assert coerce_message_content({"text": " hi "}) == "hi"

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_values(self, value):
        assert coerce_message_content(value) == ""


class TestExtractPromptAndPerspective:
    def test_reads_structured_output(self):
        raw = '{"optimised_prompt": "Do the thing", "perspective_used": "CoT"}'
        assert extract_prompt_and_perspective(raw, "default") == ("Do the thing", "CoT")

    def test_falls_back_to_the_default_perspective(self):
        raw = '{"optimised_prompt": "Do the thing"}'
        assert extract_prompt_and_perspective(raw, "default") == ("Do the thing", "default")

    def test_unstructured_output_is_used_verbatim(self, tmp_path, monkeypatch):
        # Parse failures are logged, so keep the log out of the repo copy.
        monkeypatch.setattr("live_mode_utils.PARSE_FAILURE_LOG", str(tmp_path / "failures.json"))
        prompt, perspective = extract_prompt_and_perspective(
            "```\nJust a plain prompt\n```", "default", model_name="m", step="rewriter_a"
        )
        assert prompt == "Just a plain prompt"
        assert perspective == "default"
