"""
Provision the weights this lab needs that dev/models/ does not already have,
and check that the rest are in place.

Downloaded (commercial-use licences):
    ip-adapter-plus-face_sd15 / ip-adapter-full-face_sd15   h94/IP-Adapter, Apache-2.0
    image_encoder (OpenCLIP ViT-H/14, ~2.5 GB)               h94/IP-Adapter, MIT (LAION)

    python fetch_models.py            # download what is missing, then check all
    python fetch_models.py --check    # only report
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

# Everything else is shared with dev/ and should already be provisioned there.
REQUIRED = {
    "SD1.5 (unet)": Path(C.MODEL_PATHS["sd15"]) / "unet",
    "ControlNet lineart": Path(C.MODEL_PATHS["controlnet"]) / "config.json",
    "style LoRA": Path(C.MODEL_PATHS["lora"]),
    "LCM-LoRA (--lcm)": Path(C.MODEL_PATHS["lcm_lora"]) / "pytorch_lora_weights.safetensors",
    "AuraFace detector": Path(C.MODEL_PATHS["auraface"]) / "scrfd_10g_bnkps.onnx",
    "AuraFace 3D landmarks": Path(C.MODEL_PATHS["auraface"]) / "1k3d68.onnx",
    "lineart annotator": Path(C.MODEL_PATHS["annotators_lineart"]) / "sk_model.pth",
    "MediaPipe face landmarker": Path(C.MODEL_PATHS["mediapipe_face_landmarker"]),
}


def download():
    from huggingface_hub import hf_hub_download
    for (repo, filename), rel in C.DOWNLOADS.items():
        dest = C.MODELS_ROOT / rel
        if dest.exists():
            continue
        print(f"downloading {repo}/{filename} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            got = hf_hub_download(repo, filename, local_dir=tmp)
            shutil.move(got, dest)


def check():
    rows = [(name, p) for name, p in REQUIRED.items()]
    rows += [(rel, C.MODELS_ROOT / rel) for rel in C.DOWNLOADS.values()]
    missing = 0
    for name, p in rows:
        ok = p.exists()
        missing += not ok
        print(f"  [{'ok' if ok else 'MISSING'}] {name}: {p}")
    return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, no downloads")
    args = ap.parse_args()
    print(f"MODELS_ROOT = {C.MODELS_ROOT}")
    if not args.check:
        download()
    missing = check()
    if missing:
        print(f"{missing} missing (everything except the IP-Adapter files comes from dev/'s setup)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
