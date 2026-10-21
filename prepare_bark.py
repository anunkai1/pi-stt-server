#!/usr/bin/env python3
"""Download Underdog Bark 3.0 Plus and unpack it into a plain transformers model directory.

    prepare_bark.py OUT_DIR

The files are fetched at a pinned Hugging Face revision and checksum-verified.
Decoding the entropy-coded model.bark takes ~25 s, so it is done once here
rather than on every cold start; the result loads in well under a second and
is bit-identical to what the official loader produces. The processor files
(feature extractor, tokenizer) are saved alongside so the server never touches
the network. Bark 3.0 is CC-BY-4.0; its NOTICE is copied into OUT_DIR.
"""

import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = "ConwayResearch/Underdog-Bark-3.0-Plus"
REVISION = "d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb"
SHA256 = {
    "model.bark": "fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857",
    "bark_loader.py": "e86b7f130bec1a8fa14dd315cfce95785b890dbd30000f6ceb2597fdd4312b02",
    "NOTICE": "0d4bc9bcb50fc5b239e573a7ec5765534636c13b7d312866a9b423ad0b8621cf",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])

    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir)
        for name, digest in SHA256.items():
            hf_hub_download(REPO, name, revision=REVISION, local_dir=src)
            actual = hashlib.sha256((src / name).read_bytes()).hexdigest()
            if actual != digest:
                raise SystemExit(f"{name}: sha256 {actual} != pinned {digest}")
        spec = importlib.util.spec_from_file_location("bark_loader", src / "bark_loader.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)

        partial = out.with_name(out.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        loader.load_model(str(src / "model.bark")).save_pretrained(partial)
        AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3").save_pretrained(partial)
        shutil.copy(src / "NOTICE", partial / "NOTICE")
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    print(f"bark model prepared at {out}")


if __name__ == "__main__":
    main()
