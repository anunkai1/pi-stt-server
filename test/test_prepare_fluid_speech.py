"""Provisioning regression with fake downloads and checksum verification."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

import prepare_fluid_speech


def test_prepare_downloads_pinned_revision_and_verifies_checksums(monkeypatch, tmp_path):
    payload = b"fixture"
    monkeypatch.setattr(prepare_fluid_speech, "SHA256", {
        name: hashlib.sha256(payload).hexdigest()
        for name in prepare_fluid_speech.SHA256
    })
    calls = []

    def download(repo, name, *, revision, local_dir):
        assert repo == prepare_fluid_speech.REPO
        assert revision == prepare_fluid_speech.REVISION
        calls.append(name)
        (Path(local_dir) / name).write_bytes(payload)

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    out = tmp_path / "model"
    monkeypatch.setattr(sys, "argv", ["prepare_fluid_speech.py", str(out)])
    prepare_fluid_speech.main()
    assert set(calls) == set(prepare_fluid_speech.SHA256)
    for name in prepare_fluid_speech.SHA256:
        assert (out / name).read_bytes() == payload


def test_prepare_rejects_checksum_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(prepare_fluid_speech, "SHA256", {"vocab.txt": "0" * 64})

    def download(repo, name, *, revision, local_dir):
        (Path(local_dir) / name).write_bytes(b"corrupt")

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    out = tmp_path / "model"
    monkeypatch.setattr(sys, "argv", ["prepare_fluid_speech.py", str(out)])
    with pytest.raises(SystemExit, match="vocab.txt: sha256"):
        prepare_fluid_speech.main()
    assert not out.exists()
