"""Job and part storage for the API.

Deliberately simple: an in-process registry with an optional disk spill, not a
database. The API's job is to plan welds, and a plan is cheap to recompute from
the CAD file -- so the store keeps recent work in memory for the plugins to
poll, writes uploaded CAD and generated programs to disk, and evicts the oldest
entries when it fills.

Threading: the whole store is guarded by one lock. The planner holds the GIL
for most of its work anyway (NumPy and OCCT both release it in places, but not
predictably), so finer-grained locking would buy nothing and risk a race for
the sake of it.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model import Part, WeldJob

logger = logging.getLogger(__name__)

__all__ = ["JobRecord", "Store", "get_store"]


@dataclass
class JobRecord:
    """One planning run, with its state and whatever it produced."""

    id: str
    status: str = "pending"  # 'pending' | 'running' | 'done' | 'failed'
    job: WeldJob | None = None
    part: Part | None = None
    error: str = ""
    stage: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    source_path: Path | None = None
    exports: dict[str, Path] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.created_at

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "has_job": self.job is not None,
            "exports": sorted(self.exports),
        }


class Store:
    """In-memory registry with a working directory on disk."""

    def __init__(self, root: Path | None = None, capacity: int = 64) -> None:
        self.root = Path(root) if root else Path(tempfile.gettempdir()) / "weldforge"
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._records: OrderedDict[str, JobRecord] = OrderedDict()
        self._parts: OrderedDict[str, Part] = OrderedDict()
        self._lock = threading.RLock()

    # -- parts ---------------------------------------------------------------

    def put_part(self, part: Part) -> Part:
        with self._lock:
            self._parts[part.id] = part
            self._parts.move_to_end(part.id)
            self._evict(self._parts)
        return part

    def get_part(self, part_id: str) -> Part | None:
        with self._lock:
            part = self._parts.get(part_id)
            if part is not None:
                self._parts.move_to_end(part_id)
            return part

    def list_parts(self) -> list[Part]:
        with self._lock:
            return list(reversed(self._parts.values()))

    # -- jobs ----------------------------------------------------------------

    def create_record(self, job_id: str, **kwargs: Any) -> JobRecord:
        with self._lock:
            record = JobRecord(id=job_id, **kwargs)
            self._records[job_id] = record
            self._records.move_to_end(job_id)
            self._evict(self._records)
            return record

    def get_record(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is not None:
                self._records.move_to_end(job_id)
            return record

    def list_records(self) -> list[JobRecord]:
        with self._lock:
            return list(reversed(self._records.values()))

    def update_record(self, job_id: str, **changes: Any) -> JobRecord | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            for key, value in changes.items():
                setattr(record, key, value)
            return record

    # -- files ---------------------------------------------------------------

    def workspace(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, job_id: str, filename: str, data: bytes) -> Path:
        """Store an uploaded CAD file, keeping only its basename.

        The basename is taken apart deliberately: an upload named
        ``../../etc/passwd`` must not escape the workspace, and a client that
        sends a full Windows path should not create a directory tree.
        """
        safe = Path(filename).name or "upload.step"
        target = self.workspace(job_id) / safe
        target.write_bytes(data)
        logger.info("stored upload %s (%d bytes)", target, len(data))
        return target

    def save_export(self, job_id: str, filename: str, text: str) -> Path:
        safe = Path(filename).name or "program.txt"
        target = self.workspace(job_id) / "export"
        target.mkdir(parents=True, exist_ok=True)
        path = target / safe
        path.write_text(text)
        with self._lock:
            record = self._records.get(job_id)
            if record is not None:
                record.exports[safe] = path
        return path

    def purge(self, job_id: str) -> bool:
        with self._lock:
            record = self._records.pop(job_id, None)
        folder = self.root / job_id
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        return record is not None

    # -- housekeeping --------------------------------------------------------

    def _evict(self, mapping: OrderedDict) -> None:
        while len(mapping) > self.capacity:
            key, _ = mapping.popitem(last=False)
            logger.debug("store evicted %s", key)
            folder = self.root / str(key)
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._parts.clear()


_store: Store | None = None


def get_store(root: Path | None = None) -> Store:
    """The process-wide store, created on first use."""
    global _store
    if _store is None:
        _store = Store(root)
    return _store
