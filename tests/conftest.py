"""Shared fixtures.

Nothing here touches the network or a GPU: the segmenter and the Hub API are
both stubbed.  A test suite that needs either would not be run, and the bugs
worth catching here are in the plumbing around them.
"""

from __future__ import annotations

import numpy as np
import pytest


class FakeHfApi:
    """Records what would have been sent to the Hub."""

    def __init__(self, *args, **kwargs):
        self.files: list[str] = []
        self.uploads: list[str] = []
        self.commits: list[list[str]] = []
        self.created = False

    def create_repo(self, *args, **kwargs):
        self.created = True

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def upload_file(self, path_in_repo=None, **kwargs):
        self.uploads.append(path_in_repo)
        if path_in_repo not in self.files:
            self.files.append(path_in_repo)

    def create_commit(self, operations=None, **kwargs):
        paths = [op.path_in_repo for op in (operations or [])]
        self.commits.append(paths)
        self.files.extend(paths)


@pytest.fixture
def fake_api(monkeypatch):
    """Patch HfApi so HubWriter talks to a recorder instead of the Hub."""
    import huggingface_hub

    api = FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: api)
    return api


@pytest.fixture
def keep_shards(monkeypatch):
    """Stop HubWriter deleting shards, so a test can read them back."""
    from pathlib import Path

    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: None)


def make_record(index: int = 0, **overrides):
    rec = dict(
        segment_id=f"ibrahim-aldosari__087__{index:04d}",
        reciter_slug="ibrahim-aldosari",
        source_id="ibrahim-aldosari/087",
        source_path="audio/ibrahim-aldosari/087.mp3",
        index=index,
        start_sample=index * 160000,
        end_sample=index * 160000 + 128000,
        start_seconds=float(index * 10),
        end_seconds=float(index * 10 + 8),
        duration_seconds=8.0,
        sample_rate=16000,
        audio_path=None,
        source_is_complete=True,
        is_last_of_source=False,
    )
    rec.update(overrides)
    return rec


def make_wave(seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * seconds)) * 0.05).astype(np.float32)
