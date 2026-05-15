"""Extraction history persistence service."""

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter

from app.models.schemas import ExtractionRequest
from app.router.extraction_router import RouterResult

HISTORY_FILE = Path(__file__).resolve().parents[3] / "extraction_history.json"
MAX_HISTORY = 1000


@dataclass
class HistoryEntry:
    """Represents a single entry in the extraction history."""

    id: str
    mode: str
    timestamp: str
    flow_used: str
    confidence: float
    latest_message: str
    content_preview: str
    provider: str | None = None


class HistoryRepository:
    """Manages persistence of extraction history entries."""

    _file_path: Path
    _max_entries: int
    _lock: threading.Lock
    _type_adapter: TypeAdapter[list[HistoryEntry]]

    def __init__(self, file_path: Path = HISTORY_FILE, max_entries: int = MAX_HISTORY) -> None:
        self._file_path = file_path
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._type_adapter = TypeAdapter(list[HistoryEntry])

    def save(
        self,
        mode: str,
        content: str,
        flow_used: str,
        confidence: float,
        latest_message: str,
        provider: str | None = None,
    ) -> HistoryEntry:
        """Saves a new history entry and returns it."""
        entry = HistoryEntry(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            content_preview=content[:120].replace("\n", " "),
            mode=mode,
            flow_used=flow_used,
            confidence=confidence,
            latest_message=latest_message,
            provider=provider,
        )
        with self._lock:
            history = self._load_raw()
            history.insert(0, asdict(entry))
            self._write(history[: self._max_entries])
        return entry

    def load(self) -> list[HistoryEntry]:
        """Loads all history entries."""
        return self._type_adapter.validate_python(self._load_raw())

    def clear(self) -> None:
        """Clears all history entries."""
        self._write([])

    def _load_raw(self) -> list[dict[str, object]]:
        """Loads raw history data from the file."""
        if not self._file_path.exists():
            return []
        try:
            data: list[dict[str, object]] = json.loads(self._file_path.read_text())
            return data
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, entries: list[dict[str, object]]) -> None:
        """Writes the given entries to the history file."""
        try:
            self._file_path.write_text(json.dumps(entries, indent=2))
        except OSError as exc:
            raise IOError("Failed to write history file") from exc


_repository = HistoryRepository()


def save_entry(result: RouterResult, request: ExtractionRequest) -> HistoryEntry:
    """Saves a new history entry and returns it."""
    return _repository.save(
        content=request.content,
        mode=request.mode.value,
        provider=request.provider,
        flow_used=result.flow_used,
        confidence=result.confidence,
        latest_message=result.latest_message,
    )


def load_history() -> list[HistoryEntry]:
    """Loads all history entries."""
    return _repository.load()


def clear_history() -> None:
    """Clears all history entries."""
    _repository.clear()
