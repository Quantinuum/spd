"""Evaluation-owned disk storage for execution checkpoints.

This module knows nothing about circuit operations, backend mathematics, or
checkpoint grouping. The runner chooses what to save and when to restore it.
"""
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory


class CheckpointStore:
    """A one-shot collection of snapshots, with an opaque execution signature.

    Only one requested snapshot is loaded at a time. close() is idempotent;
    TemporaryDirectory also cleans up when an abandoned store is collected.
    """

    def __init__(self, signature, directory=None):
        self.signature = signature
        self._temporary = TemporaryDirectory(prefix="spd-channels-", dir=directory)
        self.directory = Path(self._temporary.name)
        self.closed = False

    def save(self, key, state):
        if self.closed:
            raise ValueError("Checkpoint store is closed.")
        with (self.directory / f"{key}.pickle").open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, key):
        if self.closed:
            raise ValueError("Checkpoints have already been consumed or closed.")
        with (self.directory / f"{key}.pickle").open("rb") as handle:
            return pickle.load(handle)

    def close(self):
        self._temporary.cleanup()
        self.closed = True
