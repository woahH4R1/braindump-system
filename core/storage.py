"""Read/write access to the ``braindump-data`` repository.

Two interchangeable backends sit behind the :class:`Storage` facade:

* :class:`GitHubBackend` — used in production (Hugging Face Spaces). It
  talks to the private ``braindump-data`` repo through the GitHub API
  using a token supplied via the ``GITHUB_TOKEN`` secret. The token is
  NEVER hardcoded.
* :class:`LocalBackend` — used for local development and testing when no
  token is present. It mirrors the same directory layout under
  ``_local_data/`` on disk so the whole app runs without any secrets.

The backend is selected automatically: if ``GITHUB_TOKEN`` is set we use
GitHub, otherwise we fall back to local disk.

Every layer in the system only ever touches data through this module, so
the storage details stay in one place.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

# Directory layout inside braindump-data (mirrored locally).
#
# Browsable by design: one self-contained folder per dump under dumps/, with
# all machine-derived corpus state cordoned off under _system/ so the human
# view (your actual inputs) stays clean.
DUMPS_DIR = "dumps"           # dumps/<id>/{raw.txt, meta.json, keywords.json, graph.json}
SYSTEM_DIR = "_system"        # machine-derived state — safe to ignore when browsing
INDEX_PATH = "index.json"

SEMANTIC_DIR = f"{SYSTEM_DIR}/semantic"
GLOBAL_IDF_PATH = f"{SYSTEM_DIR}/global_idf.json"
GLOBAL_GRAPH_PATH = f"{SYSTEM_DIR}/global_graph.json"


def dump_dir(dump_id: str) -> str:
    return f"{DUMPS_DIR}/{dump_id}"


def raw_path(dump_id: str) -> str:
    return f"{dump_dir(dump_id)}/raw.txt"


def meta_path(dump_id: str) -> str:
    return f"{dump_dir(dump_id)}/meta.json"


def keywords_path(dump_id: str) -> str:
    return f"{dump_dir(dump_id)}/keywords.json"


def dump_graph_path(dump_id: str) -> str:
    return f"{dump_dir(dump_id)}/graph.json"


class LocalBackend:
    """Filesystem backend mirroring the braindump-data layout."""

    def __init__(self, root: str = "_local_data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _full(self, path: str) -> Path:
        return self.root / path

    def read_bytes(self, path: str) -> bytes | None:
        p = self._full(path)
        return p.read_bytes() if p.exists() else None

    def write_bytes(self, path: str, data: bytes, message: str = "") -> None:
        p = self._full(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list_dir(self, path: str) -> list[str]:
        p = self._full(path)
        if not p.exists():
            return []
        return sorted(f.name for f in p.iterdir() if f.is_file())

    def exists(self, path: str) -> bool:
        return self._full(path).exists()


class GitHubBackend:
    """GitHub API backend for the private braindump-data repo."""

    def __init__(self, token: str, repo_full_name: str, branch: str = "main"):
        from github import Github  # imported lazily so local dev needs no PyGithub

        self._gh = Github(token)
        self.repo = self._gh.get_repo(repo_full_name)
        self.branch = branch

    def read_bytes(self, path: str) -> bytes | None:
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
        except Exception:
            return None
        # decoded_content handles base64 for both text and binary blobs.
        return contents.decoded_content

    def write_bytes(self, path: str, data: bytes, message: str = "") -> None:
        message = message or f"Update {path}"
        try:
            existing = self.repo.get_contents(path, ref=self.branch)
            self.repo.update_file(path, message, data, existing.sha, branch=self.branch)
        except Exception:
            # File does not exist yet -> create it (also creates parent dirs).
            self.repo.create_file(path, message, data, branch=self.branch)

    def list_dir(self, path: str) -> list[str]:
        try:
            contents = self.repo.get_contents(path, ref=self.branch)
        except Exception:
            return []
        if not isinstance(contents, list):
            contents = [contents]
        return sorted(c.name for c in contents if c.type == "file")

    def exists(self, path: str) -> bool:
        try:
            self.repo.get_contents(path, ref=self.branch)
            return True
        except Exception:
            return False


class Storage:
    """Facade over a backend with typed helpers for text/json/npy."""

    def __init__(self, backend=None):
        if backend is not None:
            self.backend = backend
        else:
            token = os.environ.get("GITHUB_TOKEN")
            repo = os.environ.get("GITHUB_DATA_REPO")
            if token and repo:
                branch = os.environ.get("GITHUB_DATA_BRANCH", "main")
                self.backend = GitHubBackend(token, repo, branch)
            else:
                self.backend = LocalBackend()

    @property
    def kind(self) -> str:
        return type(self.backend).__name__

    # -- text ----------------------------------------------------------
    def read_text(self, path: str) -> str | None:
        data = self.backend.read_bytes(path)
        return data.decode("utf-8") if data is not None else None

    def write_text(self, path: str, text: str, message: str = "") -> None:
        self.backend.write_bytes(path, text.encode("utf-8"), message)

    # -- json ----------------------------------------------------------
    def read_json(self, path: str, default=None):
        data = self.backend.read_bytes(path)
        if data is None:
            return default
        return json.loads(data.decode("utf-8"))

    def write_json(self, path: str, obj, message: str = "") -> None:
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        self.backend.write_bytes(path, text.encode("utf-8"), message)

    # -- numpy ---------------------------------------------------------
    def read_npy(self, path: str):
        data = self.backend.read_bytes(path)
        if data is None:
            return None
        return np.load(io.BytesIO(data), allow_pickle=False)

    def write_npy(self, path: str, array, message: str = "") -> None:
        buf = io.BytesIO()
        np.save(buf, np.asarray(array), allow_pickle=False)
        self.backend.write_bytes(path, buf.getvalue(), message)

    # -- listing -------------------------------------------------------
    def list_dir(self, path: str) -> list[str]:
        return self.backend.list_dir(path)

    def exists(self, path: str) -> bool:
        return self.backend.exists(path)

    # -- master index --------------------------------------------------
    def load_index(self) -> dict:
        return self.read_json(INDEX_PATH, default={"dumps": []})

    def save_index(self, index: dict) -> None:
        self.write_json(INDEX_PATH, index, message="Update master index")

    def add_to_index(self, entry: dict) -> dict:
        index = self.load_index()
        index["dumps"] = [d for d in index["dumps"] if d.get("id") != entry["id"]]
        index["dumps"].append(entry)
        index["dumps"].sort(key=lambda d: d["id"])
        self.save_index(index)
        return index
