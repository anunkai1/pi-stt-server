"""Provisioning regression with fake downloads, loader and processor."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import prepare_bark


def test_processor_uses_the_exact_pinned_revision(monkeypatch, tmp_path):
    payload = b"fixture"
    monkeypatch.setattr(prepare_bark, "SHA256", {
        name: hashlib.sha256(payload).hexdigest()
        for name in ("model.bark", "bark_loader.py", "NOTICE")
    })
    def download(repo, name, *, revision, local_dir):
        assert revision == prepare_bark.REVISION
        (Path(local_dir) / name).write_bytes(payload)
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    def save(path):
        Path(path).mkdir(exist_ok=True)
    processor = SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(save_pretrained=save)))
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = processor
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    def load(module):
        module.load_model = lambda path: SimpleNamespace(save_pretrained=save)
    monkeypatch.setattr(prepare_bark.importlib.util, "spec_from_file_location", lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=load)))
    monkeypatch.setattr(prepare_bark.importlib.util, "module_from_spec", lambda spec: SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["prepare_bark.py", str(tmp_path / "model")])
    prepare_bark.main()
    processor.from_pretrained.assert_called_once_with(
        "nvidia/parakeet-tdt-0.6b-v3", revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    )
