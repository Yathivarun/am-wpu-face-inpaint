"""
Target FRU test: visitor identity drawn in a premade character's head pose.

    python gen_characters.py --assets ../../../posed-characters/assets \
        --visitors mul_face/p1 mul_face/p2 mul_face/p3 mul_face/p4 mul_face/p5 \
        --aura-file <step-0155000/faceid_aura_sd15.safetensors> --faceid-repo ~/WPU/testings/faceid-adapter

Per visitor (a folder of their registration photos, e.g. FV / LV / RV):
  identity  AuraFace vector = mean over the photos (our adapter), and a hair-inclusive
            crop of the photo whose head yaw is closest to the character's (full-face).
            Nothing else is computed on the visitor: no lineart, no parsing.
Per character: the premade assets from build_assets.py (control maps, head mask, crop box).

Generation is the production FRU generator itself (triton-server fru_generator, loaded
in-process as in triton_local.py, full-face + AuraFace adapter, diffusers-native): LCM
4 steps, CFG off, sktchdrw style LoRA, TAESD, at the asset's 512 px crop. Only the
ControlNet and its control map change per variant:

  variant       ControlNet          control map          mode
  line_noface   lineart             lineart_noface.png   character style: hair/headwear lines, face free
  line_contour  lineart             lineart_contour.png  + the character's face-edge lines (size, forehead)
  <variant>+kps + MediaPipe-face    + mesh.png           second ControlNet at --kps-scale: the character's
                                                         face keypoints place eyes, nose and mouth
  depth_smooth  depth               depth_smooth.png     character style: head volume, face smoothed
  line_full     lineart             lineart_full.png     reference: the character's own face lines
  mesh          MediaPipe-face      mesh.png             user hair (dropped 2026-10-21; --variants mesh)
  none          lineart @ 0         (blank)              no character control (--variants none)

Each variant runs with identity "ff" (full-face only) and "ff+ours" (+ AuraFace adapter).
Besides the full sheets, sheet_<visitor>_prod_{raw,comp}.jpg hold only PROD_VARIANTS
(line_noface, line_noface+kps): the production assessment, without the research variants.
Identity knobs (one setting per run, recorded in summary.json):
  --guidance G      CFG scale for the LCM pass (production 1.0 = off; >1 runs a batch of 2)
  --id-mask face    both identity adapters act only inside the character's face_mask.png
                    (diffusers ip_adapter_masks), so their strength cannot spill into the style
  --kps-map F       the +kps map: mesh.png (full mesh) or a sparser kps_*.png from build_assets
  --steps N         LCM steps (production 4)
  --id-vector view  our adapter gets the AuraFace vector of the view nearest the character's
                    yaw instead of the mean over all views
The drawing is pasted back into the scene through the character's head silhouette
(head_mask.png), feathered, with its paper tone matched to the scene's.

Scores (per generation): AuraFace cosine (circular for ours), FaceNet cosine (independent),
head yaw of the drawing vs the character's (pose error, degrees). Faces the detector
misses (extreme profiles) score None: the visual sheets decide there.

Output <out>/:
  <visitor>/<char>/<variant>_<id>.png        raw 512 drawing
  <visitor>/<char>/<variant>_<id>_comp.jpg   pasted into the scene, zoomed on the head
  <visitor>/scene_<char>_<variant>_<id>.jpg  full scene (only for --full-scenes combos)
  sheet_<visitor>_raw.jpg / sheet_<visitor>_comp.jpg   rows = characters,
        columns = visitor photo used | character | each variant x identity
  scores.csv, summary.json
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import triton_local as TL                      # noqa: E402  (production generator loader)
from faces import face_app, largest_face, reference_crops   # noqa: E402
from stage0 import _crown                      # noqa: E402

VARIANTS = {  # name: (controlnet key, control file or None, default ControlNet scale)
    "line_noface":  ("lineart", "lineart_noface.png", 0.6),
    "line_contour": ("lineart", "lineart_contour.png", 0.6),
    "depth_smooth": ("depth", "depth_smooth.png", 0.6),
    "line_full":    ("lineart", "lineart_full.png", 0.5),
    "mesh":         ("mesh", "mesh.png", 0.9),
    "none":         ("lineart", None, 0.0),
}
DEFAULT_VARIANTS = ["line_noface", "line_contour", "line_noface+kps", "line_contour+kps"]
PROD_VARIANTS = ["line_noface", "line_noface+kps"]   # the production candidates: their own sheet
VARIANT_CHOICES = list(VARIANTS) + [v + "+kps" for v in VARIANTS if v not in ("mesh", "none")]
IDS = ["ff", "ff+ours"]
CN_DIRS = {"mesh": "control_v2p_sd15_mediapipe_face", "depth": "control_v11f1p_sd15_depth"}


class FaceNet:
    """Independent identity scorer (as faceid-adapter eval2.py): InceptionResnetV1 VGGFace2."""

    def __init__(self, device):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch, self.device = torch, device
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(self, rgb, box):
        x0, y0, x1, y1 = box
        m = 0.1 * max(x1 - x0, y1 - y0)
        h, w = rgb.shape[:2]
        crop = rgb[int(max(0, y0 - m)):int(min(h, y1 + m)), int(max(0, x0 - m)):int(min(w, x1 + m))]
        if crop.size == 0:
            return None
        x = cv2.resize(crop, (160, 160)).astype(np.float32)
        x = self.torch.from_numpy((x - 127.5) / 128.0).permute(2, 0, 1)[None].to(self.device)
        with self.torch.no_grad():
            v = self.net(x)[0].float().cpu().numpy()
        return v / np.linalg.norm(v)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-8)


def analyse_visitor(folder, aura, facenet):
    """Per photo: yaw, hair-inclusive crop (full-face input), AuraFace and FaceNet vectors."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in TL._IMG_EXT)
    views, sexes = [], []
    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        f3d = largest_face(face_app().get(bgr))
        emb, fa = aura.embed(bgr)
        if f3d is None or emb is None:
            print(f"  {p.name}: no face, skipped")
            continue
        sexes.append(getattr(fa, "sex", None))
        views.append({"name": p.name, "img": img, "yaw": float(f3d.pose[1]),
                      "crop": reference_crops([img])[0], "aura": emb,
                      "facenet": facenet.embed(np.array(img), fa.bbox) if facenet else None})
    if not views:
        return None
    sex = max(set(sexes), key=sexes.count) if any(sexes) else None
    fn = [v["facenet"] for v in views if v["facenet"] is not None]
    return {"name": Path(folder).name, "views": views,
            "gender": {"M": "male", "F": "female"}.get(sex),
            "aura": unit(np.mean([v["aura"] for v in views], axis=0)),
            "facenet": unit(np.mean(fn, axis=0)) if fn else None}


def load_assets(root, only=None):
    chars = []
    for d in sorted(Path(root).iterdir()):
        if (d / "meta.json").is_file() and (not only or d.name in only):
            m = json.loads((d / "meta.json").read_text())
            m["dir"] = d
            chars.append(m)
    return chars


def control_map(char, fname, size):
    if fname is None:
        w, h = size if isinstance(size, (list, tuple)) else (size, size)
        return np.zeros((h, w, 3), np.uint8)
    p = char["dir"] / fname
    return np.array(Image.open(p).convert("RGB")) if p.is_file() else None


def gen_head_mask(drawing):
    """Head of the generated drawing: AuraFace 68-point outline + crown (or None)."""
    from build_assets import face68_outline
    f = largest_face(face_app().get(cv2.cvtColor(np.array(drawing), cv2.COLOR_RGB2BGR)))
    if f is None or getattr(f, "landmark_3d_68", None) is None:
        return None, None
    oval = np.zeros(drawing.size[::-1], np.uint8)
    cv2.fillPoly(oval, [cv2.convexHull(face68_outline(np.asarray(f.landmark_3d_68)[:, :2]).astype(np.int32))], 255)
    return _crown(oval), f


def paste(scene, drawing, char):
    """Drawing -> scene at the crop box, through the character's head silhouette, feathered.
    Per-channel gain so the drawing's paper (bright end inside the mask) matches the scene's."""
    x0, y0, x1, y1 = char["crop_box"]
    bw, bh = x1 - x0, y1 - y0
    m = np.array(Image.open(char["dir"] / "head_mask.png").convert("L"))
    k = max(3, int(0.02 * min(m.shape))) | 1
    m = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255
    m = cv2.resize(m, (bw, bh))[..., None]
    d = np.array(drawing.convert("RGB").resize((bw, bh), Image.LANCZOS), np.float32)
    out = np.array(scene, np.float32)
    region = out[y0:y1, x0:x1]
    inside = m[..., 0] > 0.5
    if inside.any():
        gain = np.percentile(region[inside], 95, axis=0) / np.maximum(np.percentile(d[inside], 95, axis=0), 1)
        d = np.clip(d * np.clip(gain, 0.7, 1.3), 0, 255)
    out[y0:y1, x0:x1] = region * (1 - m) + d * m
    return Image.fromarray(out.astype(np.uint8))


def zoom(img, box, size=512, ctx=1.25):
    x0, y0, x1, y1 = box
    cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * ctx / 2
    W, H = img.size
    b = (int(max(0, cx - h)), int(max(0, cy - h)), int(min(W, cx + h)), int(min(H, cy + h)))
    return img.crop(b).resize((size, size), Image.LANCZOS)


def hook_pipe(pipe):
    """Let a per-variant function rewrite the pipeline call's kwargs. The production
    generator builds the call (prompt, LoRA, LCM, IP embeds, control image); the hook
    adds what it has no input for: a second control image (+kps) and identity masks."""
    base = type(pipe)

    class Hooked(base):
        def __call__(self, *a, **kw):
            fn = getattr(self, "_kw_hook", None)
            return base.__call__(self, *a, **(fn(kw) if fn else kw))

    Hooked.__name__ = base.__name__
    pipe.__class__ = Hooked
    pipe._kw_hook = None


def make_hook(kps_tensor, kps_scale, masks):
    def hook(kw):
        if kps_tensor is not None:
            kw["image"] = [kw["image"], kps_tensor]
            kw["controlnet_conditioning_scale"] = [float(kw.get("controlnet_conditioning_scale", 1.0)), kps_scale]
        if masks is not None:
            kw["cross_attention_kwargs"] = {**(kw.get("cross_attention_kwargs") or {}), "ip_adapter_masks": masks}
        return kw
    return hook


def sheet(rows, header, path, tile=256):
    hh = 28
    out = Image.new("RGB", (tile * len(header), hh + tile * len(rows)), "white")
    dr = ImageDraw.Draw(out)
    for c, t in enumerate(header):
        dr.text((c * tile + 6, 8), t, fill="black")
    for r, (label, ims) in enumerate(rows):
        for c, im in enumerate(ims):
            if im is None:
                continue
            t = im.convert("RGB").copy()
            t.thumbnail((tile, tile))
            out.paste(t, (c * tile + (tile - t.width) // 2, hh + r * tile + (tile - t.height) // 2))
        dr.text((6, hh + r * tile + 6), label, fill="red")
    out.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", required=True, help="build_assets.py output folder")
    ap.add_argument("--visitors", nargs="+", required=True, help="one folder of registration photos per visitor")
    ap.add_argument("--chars", nargs="*", help="only these character ids")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS, choices=VARIANT_CHOICES)
    ap.add_argument("--kps-scale", type=float, default=0.4, help="MediaPipe-face ControlNet scale in +kps variants")
    ap.add_argument("--kps-map", default="mesh.png",
                    help="+kps control map: mesh.png (full), kps_feat.png (no outline), kps_eyelip.png, kps_eyes.png")
    ap.add_argument("--steps", type=int, help="LCM steps (default: production LCM_SETTINGS, 4)")
    ap.add_argument("--guidance", type=float, help="CFG scale (default: production LCM_SETTINGS, 1.0)")
    ap.add_argument("--id-mask", choices=["none", "face"], default="none",
                    help="face: identity adapters act only inside the character's face mask")
    ap.add_argument("--id-vector", choices=["mean", "view", "front"], default="mean",
                    help="our adapter's vector: mean of all views, the view nearest the character's yaw, "
                         "or the front photo alone (*FV*, else the most frontal)")
    ap.add_argument("--cn-scale", nargs="*", default=[], metavar="VARIANT=S", help="override ControlNet scales")
    ap.add_argument("--ids", nargs="+", default=IDS, choices=IDS)
    ap.add_argument("--aura-file", required=True)
    ap.add_argument("--ff-scale", type=float, default=0.5, help="full-face IP-Adapter scale")
    ap.add_argument("--aura-scale", type=float, default=0.8, help="AuraFace adapter scale in ff+ours")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--all-genders", action="store_true", help="pair every visitor with every character")
    ap.add_argument("--gender", nargs="*", default=[], metavar="VISITOR=male|female",
                    help="override a visitor's detected gender, e.g. p5=female")
    ap.add_argument("--full-scenes", nargs="*", default=[],
                    metavar="VARIANT:ID", help="combos saved as full scene composites")
    ap.add_argument("--models-root", default=os.environ.get("SKETCH_MODELS_ROOT", str(TL.DEV / "models")))
    ap.add_argument("--faceid-repo", default=os.environ.get("FACEID_REPO", str(TL.FACEID_REPO)))
    ap.add_argument("--no-facenet", action="store_true")
    ap.add_argument("--out", default=str(_HERE / "output" / ("chars_" + time.strftime("%Y%m%d-%H%M%S"))))
    args = ap.parse_args()
    cn_scale = {v: s for v, (_, _, s) in VARIANTS.items()}
    for kv in args.cn_scale:
        k, v = kv.split("=")
        cn_scale[k] = float(v)

    # production speed knobs off for a test run (as triton_local without --compile)
    os.environ.setdefault("SKETCH_WARMUP", "0")
    os.environ.setdefault("SKETCH_COMPILE_CONTROLNET", "0")
    import torch
    torch.compile = lambda model, *a, **k: model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.faceid_repo)
    from faceid.aura import Aura
    aura = Aura(Path(args.models_root) / "AuraFace-v1")
    facenet = None if args.no_facenet else FaceNet("cuda" if torch.cuda.is_available() else "cpu")

    chars = load_assets(args.assets, args.chars)
    visitors = [v for v in (analyse_visitor(f, aura, facenet) for f in args.visitors) if v]
    for kv in args.gender:
        name, g = kv.split("=")
        for v in visitors:
            if v["name"] == name:
                v["gender"] = g
    print(f"{len(chars)} characters, {len(visitors)} visitors: "
          + ", ".join(f"{v['name']} ({v['gender']}, yaws {[round(x['yaw']) for x in v['views']]})" for v in visitors))

    # ── production FRU generator, full-face + AuraFace (diffusers-native) ─────
    ns = argparse.Namespace(models_root=args.models_root, aura_file=args.aura_file, aura_scale=args.aura_scale)
    backend = TL.build_backend("fru", "full+aura", ns, aura)
    pipe = backend.pipe
    G = backend._handle_request.__globals__          # the generator module's globals (LCM_SETTINGS)
    if args.guidance is not None:
        G["LCM_SETTINGS"]["guidance_scale"] = args.guidance
    guidance = G["LCM_SETTINGS"]["guidance_scale"]
    if args.steps is not None:
        G["LCM_SETTINGS"]["num_inference_steps"] = args.steps
    steps = G["LCM_SETTINGS"]["num_inference_steps"]
    hook_pipe(pipe)
    from diffusers import ControlNetModel
    from diffusers.image_processor import IPAdapterMaskProcessor
    from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
    mask_proc = IPAdapterMaskProcessor()
    cns = {"lineart": pipe.controlnet}
    keys = {VARIANTS[v.replace("+kps", "")][0] for v in args.variants}
    if any(v.endswith("+kps") for v in args.variants):
        keys.add("mesh")
    for key in keys - {"lineart"}:
        path = Path(args.models_root) / "controlnet" / CN_DIRS[key]
        if not path.is_dir():
            sys.exit(f"missing ControlNet for '{key}': {path}")
        fp16 = (path / "diffusion_pytorch_model.fp16.safetensors").is_file()
        cn = ControlNetModel.from_pretrained(path, torch_dtype=pipe.unet.dtype, variant="fp16" if fp16 else None)
        cns[key] = cn.to(pipe.device, memory_format=torch.channels_last)
    print(f"ControlNets: {sorted(cns)}")

    rows, full = [], {tuple(c.split(":")) for c in args.full_scenes}
    for vis in visitors:
        mine = [c for c in chars if args.all_genders or vis["gender"] is None or c["gender"] == vis["gender"]]
        raw_rows, comp_rows = [], []
        for ch in mine:
            yaw = (ch.get("pose") or {}).get("yaw", 0.0)
            view = min(vis["views"], key=lambda v: abs(v["yaw"] - yaw))
            scene = Image.open(ch["dir"] / "scene.png").convert("RGB")
            d = out / vis["name"] / ch["id"]
            d.mkdir(parents=True, exist_ok=True)
            raw_row = [view["crop"], Image.open(ch["dir"] / "crop.png")]
            comp_row = [view["crop"], zoom(scene, ch["crop_box"])]
            w_, h_ = ch["size"] if isinstance(ch["size"], list) else (ch["size"], ch["size"])
            masks = None
            if args.id_mask == "face":
                fm = mask_proc.preprocess([Image.open(ch["dir"] / "face_mask.png").convert("L")], height=h_, width=w_)
                masks = [fm.reshape(1, 1, h_, w_)] * 2           # one per adapter: full-face, ours
            front = next((v for v in vis["views"] if "FV" in v["name"]), min(vis["views"], key=lambda v: abs(v["yaw"])))
            idvec = {"view": view["aura"], "front": front["aura"]}.get(args.id_vector, vis["aura"])
            for var in args.variants:
                base_var = var.replace("+kps", "")
                key, fname, _ = VARIANTS[base_var]
                ctrl = control_map(ch, fname, ch["size"])
                kps = control_map(ch, args.kps_map, ch["size"]) if var.endswith("+kps") else None
                for idn in args.ids:
                    if ctrl is None or (var.endswith("+kps") and kps is None):   # e.g. no mesh
                        raw_row.append(None)
                        comp_row.append(None)
                        continue
                    kps_t = None
                    if kps is not None:
                        pipe.controlnet = MultiControlNetModel([cns[key], cns["mesh"]])
                        kps_t = torch.from_numpy(kps).permute(2, 0, 1)[None].to(pipe.device, pipe.unet.dtype) / 255
                    else:
                        pipe.controlnet = cns[key]
                    pipe._kw_hook = make_hook(kps_t, args.kps_scale, masks)
                    G["LCM_SETTINGS"]["controlnet_scale_fru_face"] = cn_scale[base_var]
                    req = {"face_image": np.asarray(view["crop"], np.uint8)[None],
                           "face_edges": ctrl[None],
                           "params": np.array([[1.0, 1.0, args.ff_scale, 1.0, float(args.seed)]], np.float32),
                           "gender": np.array([[ch["gender"].encode()]], dtype=object)}
                    if idn == "ff+ours":
                        req["face_id"] = idvec.reshape(1, 512).astype(np.float32)
                        req["id_strength"] = np.array([[args.aura_scale]], np.float32)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    (arr,) = TL.infer(backend, req, ["raw_face_sketch"])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    sec = time.perf_counter() - t
                    drawing = Image.fromarray(arr, "RGB")
                    _, gface = gen_head_mask(drawing)          # detection for scoring / pose
                    comp = paste(scene, drawing, ch)
                    tag = f"{var}_{idn.replace('+', '_')}"
                    drawing.save(d / f"{tag}.png")
                    zc = zoom(comp, ch["crop_box"])
                    zc.save(d / f"{tag}_comp.jpg", quality=92)
                    if (var, idn) in full:
                        c2 = comp.copy()
                        c2.thumbnail((1600, 1600))
                        c2.save(out / vis["name"] / f"scene_{ch['id']}_{tag}.jpg", quality=90)
                    a_cos = fn_cos = gyaw = None
                    if gface is not None:
                        rgb = np.array(drawing)
                        e, fa = aura.embed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        a_cos = None if e is None else round(float(e @ vis["aura"]), 4)
                        if facenet and vis["facenet"] is not None and fa is not None:
                            fe = facenet.embed(rgb, fa.bbox)
                            fn_cos = None if fe is None else round(float(fe @ vis["facenet"]), 4)
                        gyaw = round(float(gface.pose[1]), 1)
                    rows.append({"visitor": vis["name"], "char": ch["id"], "char_yaw": round(yaw, 1),
                                 "view": view["name"], "variant": var, "id": idn, "cn_scale": cn_scale[base_var],
                                 "aura": a_cos, "facenet": fn_cos, "gen_yaw": gyaw,
                                 "pose_err": None if gyaw is None else round(abs(gyaw - yaw), 1),
                                 "seconds": round(sec, 3)})
                    raw_row.append(drawing)
                    comp_row.append(zc)
                    print(f"[{vis['name']}] {ch['id']:6s} {var:12s} {idn:8s} aura {a_cos} facenet {fn_cos} "
                          f"yaw {gyaw} vs {yaw:+.0f}  {sec:.2f}s")
            raw_rows.append((ch["id"], raw_row))
            comp_rows.append((ch["id"], comp_row))
        header = ["visitor", "character"] + [f"{v} {i}" for v in args.variants for i in args.ids]
        sheet(raw_rows, header, out / f"sheet_{vis['name']}_raw.jpg")
        sheet(comp_rows, header, out / f"sheet_{vis['name']}_comp.jpg")
        cols = [(v, i) for v in args.variants for i in args.ids]
        keep = [2 + k for k, (v, _) in enumerate(cols) if v in PROD_VARIANTS]
        if keep and len(keep) < len(cols):      # production sheet: the research variants left out
            keep = [0, 1] + keep
            for rs, kind in ((raw_rows, "raw"), (comp_rows, "comp")):
                sheet([(lab, [ims[k] for k in keep]) for lab, ims in rs], [header[k] for k in keep],
                      out / f"sheet_{vis['name']}_prod_{kind}.jpg")

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return (round(float(np.mean(xs)), 4) if xs else None), len(xs)

    summary = {}
    for var in args.variants:
        for idn in args.ids:
            r = [x for x in rows if x["variant"] == var and x["id"] == idn]
            if not r:
                continue
            (a, n), (fn, _), (pe, _), (sec, _) = (mean([x[k] for x in r]) for k in ("aura", "facenet", "pose_err", "seconds"))
            summary[f"{var} {idn}"] = {"aura": a, "facenet": fn, "pose_err_deg": pe, "face_found": f"{n}/{len(r)}",
                                       "seconds": sec, "cn_scale": cn_scale[var.replace("+kps", "")]}
    report = {"assets": str(args.assets), "ff_scale": args.ff_scale, "aura_scale": args.aura_scale, "seed": args.seed,
              "guidance": guidance, "steps": steps, "kps_map": args.kps_map, "id_mask": args.id_mask, "id_vector": args.id_vector, "kps_scale": args.kps_scale,
              "aura_file": args.aura_file, "visitors": [v["name"] for v in visitors],
              "characters": [c["id"] for c in chars], "results": summary}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
