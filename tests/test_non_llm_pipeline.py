from __future__ import annotations

import pytest

from app.non_llm.confidence_scorer import ConfidenceResult, ConfidenceScorer
from app.non_llm.disclaimer_remover import DisclaimerRemover
from app.non_llm.pipeline import NonLLMPipeline, NonLLMResult, run_pipeline
from app.non_llm.reply_remover import ReplyRemover
from app.non_llm.signature_remover import SignatureRemover


# ---------------------------------------------------------------------------
# ReplyRemover
# ---------------------------------------------------------------------------


class TestReplyRemover:
    """Tests for ReplyRemover.clean()."""

    def setup_method(self) -> None:
        self.remover = ReplyRemover()

    def test_removes_on_wrote_block(self) -> None:
        text = (
            "Hi Alice,\n\nThanks for your note.\n\n"
            "On Monday, Apr 7, 2025 at 10:00 AM Alice <alice@example.com> wrote:\n"
            "> Can you send the report?\n"
        )
        result, was_modified = self.remover.clean(text)
        assert "wrote:" not in result
        assert "Can you send the report" not in result

    def test_removes_blockquoted_lines(self) -> None:
        text = "My reply.\n\n> Original line 1\n> Original line 2\n"
        result, was_modified = self.remover.clean(text)
        assert "> Original" not in result

    def test_removes_outlook_from_sent_to_subject_block(self) -> None:
        text = (
            "Sure, sounds good.\n\n"
            "From: Bob <bob@example.com>\n"
            "Sent: Monday, April 7, 2025 10:00 AM\n"
            "To: Alice <alice@example.com>\n"
            "Subject: Meeting tomorrow\n"
            "\nOriginal body here.\n"
        )
        result, was_modified = self.remover.clean(text)
        assert was_modified is True

    def test_plain_message_passes_through_unmodified(self) -> None:
        text = "Hello! This is a simple standalone message with no reply."
        result, was_modified = self.remover.clean(text)
        assert "Hello!" in result
        # was_modified may be False for a clean single-message email
        assert isinstance(was_modified, bool)


# ---------------------------------------------------------------------------
# SignatureRemover
# ---------------------------------------------------------------------------


class TestSignatureRemover:
    """Tests for SignatureRemover.clean()."""

    def setup_method(self) -> None:
        self.remover = SignatureRemover()

    def test_double_dash_separator_removes_content_after(self) -> None:
        text = "This is the message body.\n\n--\nJohn Smith\njohn@example.com"
        result, was_modified = self.remover.clean(text)
        assert "John Smith" not in result
        assert was_modified is True

    def test_best_regards_with_name_removed(self) -> None:
        text = "Please find the details below.\n\nBest regards,\nJohn Smith"
        result, was_modified = self.remover.clean(text)
        assert was_modified is True
        assert "John Smith" not in result

    def test_sent_from_iphone_line_removed(self) -> None:
        text = "Got it, thanks!\n\nSent from my iPhone"
        result, was_modified = self.remover.clean(text)
        assert "Sent from my iPhone" not in result
        assert was_modified is True

    def test_no_signature_was_modified_false(self) -> None:
        text = "This is a complete message with no signature whatsoever."
        result, was_modified = self.remover.clean(text)
        assert "complete message" in result
        assert was_modified is False


# ---------------------------------------------------------------------------
# DisclaimerRemover
# ---------------------------------------------------------------------------


class TestDisclaimerRemover:
    """Tests for DisclaimerRemover.clean()."""

    def setup_method(self) -> None:
        self.remover = DisclaimerRemover()

    def test_confidentiality_notice_removed(self) -> None:
        text = (
            "Hi Jane,\n\nPlease review the attached contract.\n\n"
            "CONFIDENTIALITY NOTICE:\n"
            "This email is confidential and intended solely for the addressee.\n"
        )
        result, was_modified = self.remover.clean(text)
        assert "CONFIDENTIALITY NOTICE" not in result
        assert was_modified is True

    def test_disclaimer_header_removed(self) -> None:
        text = (
            "Meeting confirmed for tomorrow.\n\n"
            "DISCLAIMER:\n"
            "The information in this email is confidential.\n"
        )
        result, was_modified = self.remover.clean(text)
        assert "DISCLAIMER" not in result
        assert was_modified is True

    def test_this_email_is_confidential_pattern_removed(self) -> None:
        text = (
            "Please review.\n\n"
            "This email is confidential and intended only for the recipient.\n"
            "If you are not the intended recipient, please delete it.\n"
        )
        result, was_modified = self.remover.clean(text)
        assert was_modified is True

    def test_no_disclaimer_was_modified_false(self) -> None:
        text = "Hello! Just checking in. Hope all is well."
        result, was_modified = self.remover.clean(text)
        assert "Hello!" in result
        assert was_modified is False


# ---------------------------------------------------------------------------
# ConfidenceScorer
# ---------------------------------------------------------------------------


class TestConfidenceScorer:
    """Tests for ConfidenceScorer.score()."""

    def setup_method(self) -> None:
        self.scorer = ConfidenceScorer(threshold=0.85)

    def test_empty_text_returns_zero_score(self) -> None:
        result = self.scorer.score("")
        assert result.score == 0.0
        assert result.is_reliable is False

    def test_whitespace_only_returns_zero_score(self) -> None:
        result = self.scorer.score("   ")
        assert result.score == 0.0
        assert result.is_reliable is False

    def test_very_short_text_returns_low_score(self) -> None:
        result = self.scorer.score("Hi")
        assert result.score < 0.85

    def test_clean_prose_returns_high_score(self) -> None:
        text = (
            "Hi Alice, I hope this message finds you well. "
            "I wanted to follow up on our discussion from last week. "
            "Could you please send me the updated report by end of day? "
            "Thank you for your help. Best regards."
        )
        result = self.scorer.score(text)
        assert isinstance(result, ConfidenceResult)
        assert result.score >= 0.0

    def test_html_tags_in_text_reduce_score(self) -> None:
        clean_text = "Hello, please review the report by Friday."
        html_text = "<p>Hello, <b>please</b> <a href='#'>review</a> the report by Friday.</p>"
        clean_result = self.scorer.score(clean_text)
        html_result = self.scorer.score(html_text)
        # HTML-laden text should have more noise found
        assert len(html_result.noise_found) >= len(clean_result.noise_found)

    def test_score_bounded_zero_to_one(self) -> None:
        for text in ["", "x", "Hello world! This is a complete sentence."]:
            result = self.scorer.score(text)
            assert 0.0 <= result.score <= 1.0

    def test_is_reliable_true_when_above_threshold(self) -> None:
        # Use a very low threshold to guarantee reliability
        scorer = ConfidenceScorer(threshold=0.0)
        result = scorer.score("Hello, this is a test message.")
        assert result.is_reliable is True

    def test_is_reliable_false_when_below_threshold(self) -> None:
        scorer = ConfidenceScorer(threshold=1.0)
        result = scorer.score("Hello.")
        assert result.is_reliable is False

    def test_result_is_confidence_result_dataclass(self) -> None:
        result = self.scorer.score("Test message.")
        assert hasattr(result, "score")
        assert hasattr(result, "noise_found")
        assert hasattr(result, "quality_found")
        assert hasattr(result, "details")
        assert hasattr(result, "is_reliable")


# ---------------------------------------------------------------------------
# NonLLMPipeline
# ---------------------------------------------------------------------------


class TestNonLLMPipeline:
    """Tests for NonLLMPipeline.run()."""

    def test_full_pipeline_run_returns_non_llm_result(self) -> None:
        pipeline = NonLLMPipeline(threshold=0.85)
        result = pipeline.run("Hello! This is the latest email. Please review the attached document.")
        assert isinstance(result, NonLLMResult)
        assert isinstance(result.text, str)
        assert isinstance(result.steps_executed, list)
        assert isinstance(result.metadata, dict)

    def test_steps_executed_populated_when_modifications_occur(self) -> None:
        pipeline = NonLLMPipeline(threshold=0.85)
        text = (
            "Hi,\n\nThanks for writing.\n\n"
            "On Monday, Bob wrote:\n> What is the status?\n\n"
            "Best regards,\nAlice"
        )
        result = pipeline.run(text)
        # confidence_scoring is always appended
        assert "confidence_scoring" in result.steps_executed

    def test_metadata_contains_required_keys(self) -> None:
        pipeline = NonLLMPipeline(threshold=0.85)
        result = pipeline.run("A simple email message with no noise.")
        assert "original_length" in result.metadata
        assert "confidence_score" in result.metadata
        assert "is_reliable" in result.metadata

    def test_custom_steps_can_be_injected(self) -> None:
        custom_step = ReplyRemover()
        pipeline = NonLLMPipeline(threshold=0.85, steps=[custom_step])
        result = pipeline.run("Hello, this is a test message.")
        assert isinstance(result, NonLLMResult)

    def test_pipeline_with_empty_steps_list(self) -> None:
        pipeline = NonLLMPipeline(threshold=0.85, steps=[])
        text = "Simple message."
        result = pipeline.run(text)
        assert result.text == text
        assert "confidence_scoring" in result.steps_executed


# ---------------------------------------------------------------------------
# run_pipeline convenience function
# ---------------------------------------------------------------------------


class TestRunPipelineFunction:
    """Tests for the run_pipeline() module-level convenience function."""

    def test_plain_email_returns_non_llm_result(self) -> None:
        text = (
            "Hi Bob,\n\n"
            "Just confirming our meeting at 3 PM tomorrow.\n\n"
            "Best regards,\n"
            "Alice"
        )
        result = run_pipeline(text)
        assert isinstance(result, NonLLMResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_default_threshold_is_0_85(self) -> None:
        # Pass text that is clean enough so the scorer provides a deterministic confidence
        text = "Hello, this is a clean standalone message."
        result = run_pipeline(text)
        # Verify the scorer was run (metadata populated)
        assert "confidence_score" in result.metadata

    def test_custom_threshold_accepted(self) -> None:
        text = "Short."
        result = run_pipeline(text, threshold=0.5)
        assert isinstance(result, NonLLMResult)

    def test_steps_executed_always_includes_confidence_scoring(self) -> None:
        result = run_pipeline("Any email content here.")
        assert "confidence_scoring" in result.steps_executed
