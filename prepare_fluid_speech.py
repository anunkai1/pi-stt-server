#!/usr/bin/env python3
"""Download Fluid Speech Pro ONNX (int8) and verify pinned SHA-256 checksums.

    prepare_fluid_speech.py OUT_DIR

All weight, tokenizer and licence files are revision-pinned and checksum-verified;
the server itself never touches the network. Fluid Speech Pro is CC BY 4.0;
LICENSE and NOTICE are saved alongside the weights in OUT_DIR.
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

REPO = "FluidVoice-Labs/fluid-speech-pro-onnx"
REVISION = "fa588e9f87f1b9ad0ca2e2d8f7eaf84dc3803581"
SHA256 = {
    "encoder-model.int8.onnx": "07ad77459bd3bb165c2c53e978ed1dfd64c6e129b18af2ec5ee38343be1d4451",
    "encoder-model.int8.onnx.data": "d2f8bfbb1fa399f207cde85aa3fabee0d67f6730cb31cbf31832c10de69f9c3c",
    "decoder.int8.onnx": "0bd3e225124a8a90373e500f62516270fd85fdf42237aa569489b232d21cac08",
    "decoder.int8.onnx.data": "632c3cbf59d3e52d2f7f4dda3bf8e28c045f7a3bb3dbef0f44529ba63926efaa",
    "joiner.int8.onnx": "c0e995b659ed5e15fc0c38408ee56cd9a03e6d3329f304c32b9d5017c8cdd381",
    "joiner.int8.onnx.data": "7473427ab99417fae8cf7025ba11507cb5dcc223194bae2f79064dde0936d0af",
    "vocab.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    "LICENSE": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    "NOTICE": "fb46bca8b337792a755ec6c77e0e33ebd27d181f46540e4b88382d179942625c",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])

    from huggingface_hub import hf_hub_download

    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir)
            for name, digest in SHA256.items():
                hf_hub_download(REPO, name, revision=REVISION, local_dir=src)
                actual = hashlib.sha256((src / name).read_bytes()).hexdigest()
                if actual != digest:
                    raise SystemExit(f"{name}: sha256 {actual} != pinned {digest}")
                shutil.copy2(src / name, partial / name)
        shutil.rmtree(out, ignore_errors=True)
        partial.rename(out)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    print(f"fluid_speech model prepared at {out}")


if __name__ == "__main__":
    main()
