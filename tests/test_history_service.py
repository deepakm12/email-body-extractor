from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.history_service import HistoryEntry, HistoryRepository


class TestHistoryRepositorySave:
    """Tests for HistoryRepository.save()."""

    def test_save_creates_entry_with_uuid(self, tmp_history_file: HistoryRepository) -> None:
        entry = tmp_history_file.save(
            mode="non_llm",
            content="Hello, this is a test email.",
            flow_used="non_llm",
            confidence=0.9,
            latest_message="Hello, this is a test email.",
        )
        assert isinstance(entry, HistoryEntry)
        assert entry.id is not None
        assert len(entry.id) > 0

    def test_save_creates_entry_with_timestamp(self, tmp_history_file: HistoryRepository) -> None:
        entry = tmp_history_file.save(
            mode="non_llm",
            content="Test content",
            flow_used="non_llm",
            confidence=0.8,
            latest_message="Test content",
        )
        assert entry.timestamp is not None
        assert "T" in entry.timestamp  # ISO 8601 format

    def test_save_creates_content_preview(self, tmp_history_file: HistoryRepository) -> None:
        long_content = "A" * 200
        entry = tmp_history_file.save(
            mode="non_llm",
            content=long_content,
            flow_used="non_llm",
            confidence=0.9,
            latest_message="Extracted message",
        )
        assert len(entry.content_preview) <= 120

    def test_save_stores_provider(self, tmp_history_file: HistoryRepository) -> None:
        entry = tmp_history_file.save(
            mode="llm",
            content="Email content",
            flow_used="llm",
            confidence=0.95,
            latest_message="Extracted",
            provider="openai",
        )
        assert entry.provider == "openai"


class TestHistoryRepositoryLoad:
    """Tests for HistoryRepository.load()."""

    def test_load_returns_saved_entry(self, tmp_history_file: HistoryRepository) -> None:
        tmp_history_file.save(
            mode="non_llm",
            content="Sample email content",
            flow_used="non_llm",
            confidence=0.9,
            latest_message="Sample email content",
        )
        entries = tmp_history_file.load()
        assert len(entries) == 1
        assert entries[0].mode == "non_llm"

    def test_load_returns_empty_list_when_file_absent(self, tmp_path: Path) -> None:
        repo = HistoryRepository(file_path=tmp_path / "nonexistent.json")
        entries = repo.load()
        assert entries == []

    def test_load_returns_empty_list_on_corrupt_json(self, tmp_path: Path) -> None:
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{not valid json[[[")
        repo = HistoryRepository(file_path=corrupt_file)
        entries = repo.load()
        assert entries == []

    def test_multiple_saves_newest_first(self, tmp_history_file: HistoryRepository) -> None:
        tmp_history_file.save(
            mode="non_llm",
            content="First email",
            flow_used="non_llm",
            confidence=0.7,
            latest_message="First",
        )
        tmp_history_file.save(
            mode="non_llm",
            content="Second email",
            flow_used="non_llm",
            confidence=0.8,
            latest_message="Second",
        )
        entries = tmp_history_file.load()
        assert len(entries) == 2
        # Newest first
        assert entries[0].latest_message == "Second"
        assert entries[1].latest_message == "First"


class TestHistoryRepositoryClear:
    """Tests for HistoryRepository.clear()."""

    def test_clear_empties_history(self, tmp_history_file: HistoryRepository) -> None:
        tmp_history_file.save(
            mode="non_llm",
            content="Email to be cleared",
            flow_used="non_llm",
            confidence=0.9,
            latest_message="Email to be cleared",
        )
        assert len(tmp_history_file.load()) == 1
        tmp_history_file.clear()
        assert tmp_history_file.load() == []


class TestHistoryRepositoryMaxEntries:
    """Tests for max_entries enforcement."""

    def test_max_entries_enforced(self, tmp_path: Path) -> None:
        repo = HistoryRepository(file_path=tmp_path / "history.json", max_entries=3)
        for i in range(5):
            repo.save(
                mode="non_llm",
                content=f"Email number {i}",
                flow_used="non_llm",
                confidence=0.9,
                latest_message=f"Message {i}",
            )
        entries = repo.load()
        assert len(entries) == 3

    def test_max_entries_keeps_newest(self, tmp_path: Path) -> None:
        repo = HistoryRepository(file_path=tmp_path / "history.json", max_entries=2)
        for i in range(4):
            repo.save(
                mode="non_llm",
                content=f"Email {i}",
                flow_used="non_llm",
                confidence=0.9,
                latest_message=f"Message {i}",
            )
        entries = repo.load()
        assert len(entries) == 2
        # The two most-recently inserted entries should be present
        assert entries[0].latest_message == "Message 3"
        assert entries[1].latest_message == "Message 2"


class TestHistoryRepositorySequentialOperations:
    """Basic sequential/thread-safety tests."""

    def test_sequential_save_and_load(self, tmp_history_file: HistoryRepository) -> None:
        for i in range(5):
            tmp_history_file.save(
                mode="non_llm",
                content=f"Content {i}",
                flow_used="non_llm",
                confidence=float(i) / 10,
                latest_message=f"Message {i}",
            )
        entries = tmp_history_file.load()
        assert len(entries) == 5

    def test_save_then_clear_then_save(self, tmp_history_file: HistoryRepository) -> None:
        tmp_history_file.save(
            mode="non_llm",
            content="First",
            flow_used="non_llm",
            confidence=0.9,
            latest_message="First",
        )
        tmp_history_file.clear()
        tmp_history_file.save(
            mode="non_llm",
            content="After clear",
            flow_used="non_llm",
            confidence=0.8,
            latest_message="After clear",
        )
        entries = tmp_history_file.load()
        assert len(entries) == 1
        assert entries[0].latest_message == "After clear"
