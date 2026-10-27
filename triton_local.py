"""
Production direct-sketch pipeline run locally, with a choice of identity adapter.

    # FRU (face sketch): production full-face vs plus-face vs our AuraFace adapter
    python triton_local.py --place fru --images mul_face/p1 mul_face/p2 \
        --adapters full-face plus-face aura --aura-file <step-XXXXXXX/faceid_aura_sd15.safetensors>

    # SAU (full body + face pass), one adapter, stronger identity
    python triton_local.py --place sau --images photos/ --adapters aura --aura-file ... --scale 0.6

    # 512 cap on the generation size (aspect kept), as FRU needs; 0 = production (no cap)
    python triton_local.py --place fru --images ... --adapters full-face full+aura --max-side 512 ...

    # Stage T parity: production IPAdapterFull vs the diffusers-native port (must match),
    # then the port with our AuraFace adapter added (full-face scale + --aura-scale)
    python triton_local.py --place fru --images ... --adapters full-face full-face-native full+aura \
        --aura-file <...safetensors> --aura-scale 0.6

Stages are the production code itself, not copies:
  1. preprocess   dev/pipeline/{fru,sau}_preprocess.py   (identical to sketch-service/pipeline)
  2. generate     triton-server/model_repository/{fru,sau}_generator/1/model.py, loaded in-process
                  behind a stub triton_python_backend_utils; config from triton-server/config.py
  3. postprocess  dev/pipeline/{fru,sau}_postprocess.py  (identical to sketch-service/pipeline)

The only change to the generator is which IP-Adapter class it builds:
  full-face  IPAdapterFull, as in production
  plus-face  IPAdapterPlus (16 tokens), same CLIP image input
  aura       our FaceID-format adapter on the AuraFace embedding of the same face crop
  full-face-native   the production model.py with SKETCH_IP_BACKEND=diffusers (pipeline/ip_native.py)
  full+aura          ... plus SKETCH_ID_ADAPTER=aura; the request carries the photo's AuraFace
                     vector as `face_id` and --aura-scale as `id_strength`
When full-face and full-face-native both run, summary.json gets a "parity" block
(pixel difference and AuraFace cosine between their outputs, per image).
Everything else (LoRA weights, LCM 4 steps, guidance, ControlNet scales, prompts,
TAESD) is whatever triton-server/config.py and model.py say. torch.compile and
the warm-up are off by default (speed only; --compile restores them).

Seeds are fixed (--seed), the same for every adapter. Identity score: AuraFace
cosine between the visitor photo and the generated face sketch.
Output: <out>/<adapter>/<image>_{raw,final}.jpg, scores.csv, summary.json,
sheet_raw.jpg / sheet_final.jpg (rows = images; columns = input | each adapter).
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
DEV = _HERE.parent
TRITON = DEV.parent / "triton-server"
FACEID_REPO = DEV.parents[1] / "faceid-adapter"       # sibling repo (AuraFace helper); --faceid-repo
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ADAPTERS = ["full-face", "plus-face", "aura", "full-face-native", "full+aura"]
NATIVE_ENV = {"full-face-native": {"SKETCH_IP_BACKEND": "diffusers", "SKETCH_ID_ADAPTER": "none"},
              "full+aura": {"SKETCH_IP_BACKEND": "diffusers", "SKETCH_ID_ADAPTER": "aura"}}


# ════════════════════════════════════════════════════════════════════════════
# Triton backend in-process
# ════════════════════════════════════════════════════════════════════════════
def _pb_utils_stub():
    """The parts of triton_python_backend_utils that model.py touches."""
    pb = types.ModuleType("triton_python_backend_utils")

    class Tensor:
        def __init__(self, name, arr):
            self.name, self.arr = name, arr

        def as_numpy(self):
            return self.arr

    class TritonError(Exception):
        pass

    class InferenceResponse:
        def __init__(self, output_tensors=None, error=None):
            self.output_tensors, self.error = output_tensors or [], error

    pb.Tensor, pb.TritonError, pb.InferenceResponse = Tensor, TritonError, InferenceResponse
    # Triton returns None for an optional input the request did not send
    pb.get_input_tensor_by_name = lambda request, name: Tensor(name, request[name]) if name in request else None
    return pb


def load_generator(place, models_root):
    """Import triton-server's <place>_generator/1/model.py with triton-server's
    own config / pipeline / ip_adapter packages. dev/ has modules with the same
    names (config, pipeline), so they are swapped out for the import and put
    back afterwards; the generator keeps the references it bound."""
    clash = [k for k in sys.modules if k in ("config", "pipeline") or k.startswith("pipeline.")]
    saved = {k: sys.modules.pop(k) for k in clash}
    sys.path.insert(0, str(TRITON))
    sys.modules["triton_python_backend_utils"] = _pb_utils_stub()
    try:
        path = TRITON / "model_repository" / f"{place}_generator" / "1" / "model.py"
        spec = importlib.util.spec_from_file_location(f"triton_{place}_generator", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tcfg = sys.modules["config"]
        old_root = str(tcfg.MODELS_ROOT)
        for k, v in tcfg.MODEL_PATHS.items():          # same dict object model.py imported
            if isinstance(v, str) and v.startswith(old_root):
                tcfg.MODEL_PATHS[k] = str(Path(models_root)) + v[len(old_root):]
        ipa = sys.modules["ip_adapter.ip_adapter"]
    finally:
        for k in [k for k in sys.modules if k in ("config", "pipeline") or k.startswith("pipeline.")]:
            sys.modules.pop(k)
        sys.modules.update(saved)
        sys.path.remove(str(TRITON))
    return mod, tcfg, ipa


def aura_adapter_class(ipa):
    """FaceID-format adapter (h94 layout: image_proj.proj.0/proj.2/norm +
    ip_adapter.{k}.to_k_ip/to_v_ip) on top of production's own IPAdapter class,
    so the UNet / ControlNet attention patching is exactly production's."""
    import torch

    class AuraProj(torch.nn.Module):
        def __init__(self, cross_attention_dim=768, id_dim=512, num_tokens=4, mult=2):
            super().__init__()
            self.num_tokens, self.dim = num_tokens, cross_attention_dim
            self.proj = torch.nn.Sequential(torch.nn.Linear(id_dim, id_dim * mult), torch.nn.GELU(),
                                            torch.nn.Linear(id_dim * mult, cross_attention_dim * num_tokens))
            self.norm = torch.nn.LayerNorm(cross_attention_dim)

        def forward(self, id_embeds):
            return self.norm(self.proj(id_embeds).reshape(-1, self.num_tokens, self.dim))

    class AuraIPAdapter(ipa.IPAdapter):
        def __init__(self, sd_pipe, ip_ckpt, device, aura):
            # IPAdapter.__init__ minus the CLIP image encoder. Token count from
            # the weights: 4 (research adapter) or 16 (commercial recipe).
            from safetensors import safe_open
            with safe_open(ip_ckpt, framework="pt") as f:
                out_dim = f.get_slice("image_proj.proj.2.weight").get_shape()[0]
            tokens = out_dim // sd_pipe.unet.config.cross_attention_dim
            self.device, self.ip_ckpt, self.num_tokens, self.aura = device, ip_ckpt, tokens, aura
            self.image_encoder_path = None
            self.pipe = sd_pipe.to(device)
            self.set_ip_adapter()
            self.image_proj_model = self.init_proj()
            self.load_ip_adapter()

        def init_proj(self):
            return AuraProj(self.pipe.unet.config.cross_attention_dim,
                            num_tokens=self.num_tokens).to(self.device, dtype=torch.float16)

        @torch.inference_mode()
        def get_image_embeds(self, pil_image=None, clip_image_embeds=None, need_uncond=True):
            if clip_image_embeds is None:
                img = pil_image[0] if isinstance(pil_image, list) else pil_image
                emb, _ = self.aura.embed(np.asarray(img.convert("RGB"))[:, :, ::-1].copy())
                # no face (e.g. production's neutral grey stand-in, run at scale 0)
                clip_image_embeds = torch.from_numpy(emb if emb is not None else np.zeros(512, np.float32))[None]
            e = clip_image_embeds.to(self.device, dtype=torch.float16)
            cond = self.image_proj_model(e)
            return cond, (self.image_proj_model(torch.zeros_like(e)) if need_uncond else None)

    return AuraIPAdapter


def build_backend(place, adapter, args, aura):
    # config.py reads these at import, and load_generator re-imports it per backend
    for k in ("SKETCH_IP_BACKEND", "SKETCH_ID_ADAPTER", "SKETCH_AURA_ADAPTER", "SKETCH_AURA_STRENGTH"):
        os.environ.pop(k, None)
    os.environ.update(NATIVE_ENV.get(adapter, {}))
    if adapter == "full+aura":
        os.environ["SKETCH_AURA_ADAPTER"] = str(Path(args.aura_file).resolve())
        os.environ["SKETCH_AURA_STRENGTH"] = str(args.aura_scale)
    mod, tcfg, ipa = load_generator(place, args.models_root)
    if adapter in NATIVE_ENV:
        # the production class choice, untouched; only the env differs
        backend = mod.TritonPythonModel()
        backend.initialize({})
        if backend._ip_adapter is None:
            raise RuntimeError(f"{adapter}: the IP-Adapter did not load (see the log above)")
        return backend
    if adapter == "plus-face":
        plus = str(Path(args.models_root) / "ip_adapter" / "ip-adapter-plus-face_sd15.safetensors")
        mod.IPAdapterFull = lambda pipe, enc, _ckpt, device, num_tokens=None: ipa.IPAdapterPlus(
            pipe, enc, plus, device, num_tokens=16)
    elif adapter == "aura":
        cls = aura_adapter_class(ipa)
        mod.IPAdapterFull = lambda pipe, _enc, _ckpt, device, num_tokens=None: cls(
            pipe, args.aura_file, device, aura)
    mod.IP_ADAPTER_AVAILABLE = True
    backend = mod.TritonPythonModel()
    backend.initialize({})
    if backend._ip_adapter is None:
        raise RuntimeError(f"{adapter}: the IP-Adapter did not load (see the log above)")
    return backend


def infer(backend, request, outputs):
    resp = backend.execute([request])[0]
    if resp.error is not None:
        raise RuntimeError(str(resp.error))
    got = {t.name: t.arr for t in resp.output_tensors}
    return [got[o] for o in outputs]


# ════════════════════════════════════════════════════════════════════════════
# Requests: same tensors sketch-service sends (sketch_service/triton.py)
# ════════════════════════════════════════════════════════════════════════════
def cap_size(size, max_side):
    """(w, h) scaled so the longer side is <= max_side, aspect kept, both sides multiples of 8
    (diffusers needs /8; the rounding moves the aspect by < 1%). Never upscales."""
    w, h = size
    k = min(1.0, max_side / max(w, h))
    return max(8, int(w * k) // 8 * 8), max(8, int(h * k) // 8 * 8)


def cap_pre(pre, max_side):
    """Downscale the generator inputs (face crop + its edge map) to cap_size. Masks and
    face_box_in_sketch stay in face_img_size coordinates: postprocess rescales them onto
    whatever size the sketch comes back at."""
    for name in ("face_img", "face_edges"):
        im = getattr(pre, name, None)
        if im is not None and max(im.size) > max_side:
            setattr(pre, name, im.resize(cap_size(im.size, max_side), Image.LANCZOS))
    return pre


def _np(img):
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def fru_request(pre, scale, seed, defaults, face_id=None, id_strength=None):
    req = {
        "face_image": _np(pre.face_img)[None],
        "face_edges": _np(pre.face_edges)[None],
        "params": np.array([[float(defaults["use_lora"]), float(defaults["lora_scale"]), float(scale),
                             float(defaults["use_lcm"]), float(seed)]], np.float32),
        "gender": np.array([[pre.gender.encode("utf-8")]], dtype=object),
    }
    if face_id is not None:
        req["face_id"] = np.asarray(face_id, np.float32).reshape(1, 512)
        req["id_strength"] = np.array([[float(id_strength)]], np.float32)
    return req


def sau_request(pre, scale, seed, defaults):
    has_face = pre.face_img is not None and pre.face_edges is not None
    blank = np.zeros((8, 8, 3), np.uint8)
    return {
        "body_edges": _np(pre.body_edges)[None],
        "face_image": (_np(pre.face_img) if has_face else blank)[None],
        "face_edges": (_np(pre.face_edges) if has_face else blank)[None],
        "params": np.array([[float(defaults["use_lora"]), float(defaults["lora_scale"]), float(scale),
                             float(defaults["use_lcm"]), float(has_face), float(seed),
                             float(pre.original_size[0]), float(pre.original_size[1])]], np.float32),
        "gender": np.array([[pre.gender.encode("utf-8")]], dtype=object),
    }


# ════════════════════════════════════════════════════════════════════════════
def _image_paths(items):
    paths = []
    for it in map(Path, items):
        paths += sorted(p for p in it.iterdir() if p.suffix.lower() in _IMG_EXT) if it.is_dir() else [it]
    return paths


def _sheet(rows, cols, out_path, tile=256):
    """rows: list of (label, [PIL or None per column])."""
    sheet = Image.new("RGB", (tile * len(cols), tile * len(rows)), "white")
    for r, (_, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            if img is not None:
                im = ImageOps.contain(img.convert("RGB"), (tile, tile))
                sheet.paste(im, (c * tile + (tile - im.width) // 2, r * tile + (tile - im.height) // 2))
    sheet.save(out_path, quality=90)


def parity(items, raws, aura):
    """Production IPAdapterFull vs the diffusers-native port, same seed: per-image
    pixel difference of the raw face sketches and AuraFace cosine between them.
    Pass: PSNR >= 35 dB and cosine >= 0.98 on every image (fp16 + LCM; bit-equality
    is not expected once kernels differ, but anything larger is a real change)."""
    from faceid.aura import cosine
    per = []
    for it in items:
        a, b = raws.get(("full-face", it["path"].stem)), raws.get(("full-face-native", it["path"].stem))
        if a is None or b is None or a.size != b.size:
            per.append({"image": it["path"].name, "error": "missing or size mismatch"})
            continue
        x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
        mse = float(((x - y) ** 2).mean())
        ea, _ = aura.embed(np.asarray(a)[:, :, ::-1].copy())
        eb, _ = aura.embed(np.asarray(b)[:, :, ::-1].copy())
        per.append({"image": it["path"].name, "max_abs": float(np.abs(x - y).max()),
                    "mean_abs": round(float(np.abs(x - y).mean()), 3),
                    "psnr_db": round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else 99.0,
                    "aura_cos": None if ea is None or eb is None else round(cosine(ea, eb), 4)})
    ok = [p for p in per if "error" not in p]
    passed = bool(ok) and len(ok) == len(per) and all(
        p["psnr_db"] >= 35 and (p["aura_cos"] is None or p["aura_cos"] >= 0.98) for p in ok)
    return {"pass": passed, "min_psnr_db": min((p["psnr_db"] for p in ok), default=None),
            "min_aura_cos": min((p["aura_cos"] for p in ok if p["aura_cos"] is not None), default=None),
            "images": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", choices=["fru", "sau"], default="fru")
    ap.add_argument("--images", nargs="+", required=True, help="visitor photos or folders")
    ap.add_argument("--adapters", nargs="+", choices=ADAPTERS, default=ADAPTERS)
    ap.add_argument("--aura-file", help="our adapter .safetensors (needed for 'aura' and 'full+aura')")
    ap.add_argument("--aura-scale", type=float, default=0.6, help="AuraFace adapter scale in 'full+aura'")
    ap.add_argument("--scale", type=float, help="IP-Adapter scale for every adapter "
                                                "(default: what sketch-service sends)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gender", choices=["male", "female"], help="override detected gender")
    ap.add_argument("--models-root", default=os.environ.get("SKETCH_MODELS_ROOT", str(DEV / "models")))
    ap.add_argument("--faceid-repo", default=os.environ.get("FACEID_REPO", str(FACEID_REPO)),
                    help="AM/faceid-adapter checkout (for its AuraFace helper)")
    ap.add_argument("--max-side", type=int, default=0,
                    help="cap the generation size: longer side <= this, aspect kept (0 = production, no cap)")
    ap.add_argument("--compile", action="store_true", help="keep production's torch.compile + warm-up")
    ap.add_argument("--out", help="output dir (default: output/triton_<place>_<timestamp>)")
    args = ap.parse_args()
    if {"aura", "full+aura"} & set(args.adapters) and not args.aura_file:
        sys.exit("--aura-file is required for adapters 'aura' and 'full+aura'")
    if args.place == "sau" and {"full-face-native", "full+aura"} & set(args.adapters):
        sys.exit("the native backend is FRU-only for now (SAU keeps IPAdapterFull)")

    if not args.compile:
        # read by triton-server/config.py at import; the UNet compile has no switch
        os.environ.setdefault("SKETCH_WARMUP", "0")
        os.environ.setdefault("SKETCH_COMPILE_CONTROLNET", "0")
        import torch
        torch.compile = lambda model, *a, **k: model

    out = Path(args.out) if args.out else _HERE / "output" / (
        f"triton_{args.place}_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. preprocess (dev/pipeline == sketch-service/pipeline) ─────────────
    sys.path.insert(0, str(DEV))
    from config import GENERATION_DEFAULTS                               # dev config == sketch-service
    from pipeline.inference_types import FRURawResult, SAURawResult
    if args.place == "fru":
        from pipeline.fru_postprocess import postprocess_fru as post
        from pipeline.fru_preprocess import preprocess_fru_image as pre_fn, warm_up
    else:
        from pipeline.sau_postprocess import postprocess_sau as post
        from pipeline.sau_preprocess import preprocess_sau_image as pre_fn, warm_up
    scale = args.scale if args.scale is not None else GENERATION_DEFAULTS["faceid_strength"]

    sys.path.insert(0, args.faceid_repo)
    from faceid.aura import Aura, cosine
    aura = Aura(Path(args.models_root) / "AuraFace-v1")

    warm_up()
    items = []
    for p in _image_paths(args.images):
        photo = ImageOps.exif_transpose(Image.open(p))
        pre = pre_fn(photo.copy(), gender_override=args.gender)
        if pre.face_img is None:
            print(f"{p.name}: no face crop from preprocessing, skipped")
            continue
        size0 = pre.face_img.size
        if args.max_side:
            cap_pre(pre, args.max_side)
        print(f"{p.name}: face crop {size0[0]}x{size0[1]} -> generate at {pre.face_img.width}x{pre.face_img.height}")
        ref, _ = aura.embed(np.asarray(photo.convert("RGB"))[:, :, ::-1].copy())
        items.append({"path": p, "photo": photo.convert("RGB"), "pre": pre, "ref": ref})
    if not items:
        sys.exit("no usable images")
    print(f"{len(items)} images preprocessed; adapters {args.adapters}; scale {scale}; seed {args.seed}")

    # ── 2 + 3. generate per adapter, postprocess, score ─────────────────────
    rows, raws, finals = [], {}, {}
    for adapter in args.adapters:
        t = time.perf_counter()
        backend = build_backend(args.place, adapter, args, aura)
        print(f"[{adapter}] backend ready in {time.perf_counter() - t:.1f}s")
        (out / adapter).mkdir(exist_ok=True)
        for it in items:
            stem, pre = it["path"].stem, it["pre"]
            t = time.perf_counter()
            if args.place == "fru":
                fid = it["ref"] if adapter == "full+aura" else None
                (face,) = infer(backend, fru_request(pre, scale, args.seed, GENERATION_DEFAULTS, fid,
                                                     args.aura_scale), ["raw_face_sketch"])
                raw_face = Image.fromarray(face, "RGB")
                result = post(FRURawResult(raw_face_sketch=raw_face, seed=args.seed), pre,
                              compose_scenes=False)
                final = result.final_sketch
            else:
                body, face = infer(backend, sau_request(pre, scale, args.seed, GENERATION_DEFAULTS),
                                   ["raw_body_sketch", "raw_face_sketch"])
                raw_face = None if face.shape == (8, 8, 3) and not face.any() else Image.fromarray(face, "RGB")
                result = post(SAURawResult(raw_body_sketch=Image.fromarray(body, "RGB"),
                                           raw_face_sketch=raw_face), pre, compose_scenes=False)
                final = result.final_sketch
            gen_s = time.perf_counter() - t
            if raw_face is not None:
                raw_face.save(out / adapter / f"{stem}_raw.jpg", quality=95)
            final.convert("RGB").save(out / adapter / f"{stem}_final.jpg", quality=95)
            score = None
            if it["ref"] is not None and raw_face is not None:
                e, _ = aura.embed(np.asarray(raw_face)[:, :, ::-1].copy())
                score = None if e is None else round(cosine(e, it["ref"]), 4)
            rows.append({"adapter": adapter, "image": it["path"].name, "gender": pre.gender,
                         "scale": scale, "seed": args.seed, "identity": score, "seconds": round(gen_s, 2)})
            raws[(adapter, stem)], finals[(adapter, stem)] = raw_face, final
            print(f"[{adapter}] {it['path'].name}: identity {score}  ({gen_s:.1f}s)")
        backend.finalize()
        del backend
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for a in args.adapters:
        s = [r["identity"] for r in rows if r["adapter"] == a and r["identity"] is not None]
        summary[a] = {"mean_identity": round(float(np.mean(s)), 4) if s else None,
                      "face_found": f"{len(s)}/{sum(r['adapter'] == a for r in rows)}"}
    report = {"place": args.place, "max_side": args.max_side or None, "scale": scale, "seed": args.seed, "aura_file": args.aura_file,
              "aura_scale": args.aura_scale if "full+aura" in args.adapters else None, "adapters": summary}
    if {"full-face", "full-face-native"} <= set(args.adapters):
        report["parity"] = parity(items, raws, aura)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    for name, imgs in (("raw", raws), ("final", finals)):
        _sheet([(it["path"].stem, [it["pre"].face_img] + [imgs[(a, it["path"].stem)] for a in args.adapters])
                for it in items], ["input"] + args.adapters, out / f"sheet_{name}.jpg")
    print(json.dumps(report, indent=2))
    print(f"-> {out}  (sheet columns: input | {' | '.join(args.adapters)})")


if __name__ == "__main__":
    main()
