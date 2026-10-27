"""
Face inpaint lab — run the visitor through the scenes.

    # 3 visitor photos into every male scene
    python run.py --faces a.jpg b.jpg c.jpg --gender male

    # a folder of photos, chosen scenes, save every intermediate
    python run.py --faces ../inputs/fru/siri --scenes 1 5 --debug

    # CPU only: build and save the inpaint inputs (no diffusion)
    python run.py --faces a.jpg --scenes 1 --stage0-only

    # small GPU (<8 GB) / fast few-step mode
    python run.py --faces a.jpg --gender female --offload --lcm

    # modes: pose (default; character pose + visitor identity), photo, character
    python run.py --faces mul_face/p1 --scenes 1 9 --mode pose --hair visitor --score

    # also paste onto the character's head scene (not pixel-identical to the headless one)
    python run.py --faces mul_face/p1 --scenes 1 --compose-on both

Output: output/<run>/scene_<id>.png (scene_<id>_onhead.png with --compose-on head/both) (+ debug/ with --debug or --stage0-only,
and run.json with the settings and per-scene timings).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from faces import face_embedding, identity_score, reference_crops, visitor_views  # noqa: E402
from scene_meta import all_scene_ids, ensure_scene_meta  # noqa: E402
from stage0 import build_inputs  # noqa: E402

G = C.GENERATION
_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _face_paths(items):
    paths = []
    for it in map(Path, items):
        if it.is_dir():
            paths += sorted(p for p in it.iterdir() if p.suffix.lower() in _IMG_EXT)
        else:
            paths.append(it)
    if not paths:
        sys.exit("no visitor images found")
    return paths


def _crops():
    path = C.FACE_DIR / "crops.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _scenes_for(gender):
    ids = sorted(int(k) for k, v in _crops().items()
                 if k.isdigit() and gender in (v.get("gender") or []))
    return ids or C.GENDER_SCENES[gender]


def _scene_gender(sid):
    genders = (_crops().get(str(sid)) or {}).get("gender") or []
    if not genders:
        genders = [g for g, ids in C.GENDER_SCENES.items() if sid in ids]
    return genders[0] if len(genders) == 1 else None


def main():
    ap = argparse.ArgumentParser(description="SD1.5 face inpaint into character scenes.")
    ap.add_argument("--faces", nargs="+", required=True, help="visitor photos or a folder (1-3 images)")
    who = ap.add_mutually_exclusive_group()
    who.add_argument("--gender", choices=["male", "female"], help="every scene for this gender")
    who.add_argument("--scenes", nargs="+", type=int, help="scene ids (default: all)")
    ap.add_argument("--ip-adapter", choices=list(C.MODEL_PATHS["ip_adapter_weights"]),
                    default=G["ip_adapter"],
                    help="aura = our AuraFace identity adapter (AM/faceid-adapter)")
    ap.add_argument("--adapter-file", help="IP-Adapter weight file to load instead of the "
                                           "configured one (e.g. a faceid-adapter checkpoint)")
    ap.add_argument("--lcm", action="store_true", help="LCM-LoRA few-step mode")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--ip-scale", type=float,
                    help=f"identity strength (default {G['pose_ip_scale']} pose, {G['ip_scale']} others)")
    ap.add_argument("--cn-scale", type=float, help="lineart ControlNet scale (overrides mode defaults)")
    ap.add_argument("--face-cn-scale", type=float, default=G["face_cn_scale"],
                    help="pose mode: face-mesh ControlNet scale")
    ap.add_argument("--lora", help=f"style LoRA (default {Path(C.MODEL_PATHS['lora']).name})")
    ap.add_argument("--lora-scale", type=float, default=G["lora_scale"])
    ap.add_argument("--crop-size", type=int, default=G["crop_size"], choices=[512, 640, 768])
    ap.add_argument("--mode", choices=["pose", "photo", "character"], default=C.MODE,
                    help="pose: character pose + visitor identity; photo: visitor photo lines; "
                         "character: character hair, features erased")
    ap.add_argument("--hair", choices=["character", "visitor"], default=C.HAIR,
                    help="pose mode: keep the character's hair lines, or let the identity drive the hair")
    ap.add_argument("--compose-on", choices=["headless", "head", "both"], default="headless",
                    help="scene the result is pasted onto: the headless scene (default), the "
                         "character's head scene (scene_<id>_onhead.png), or both")
    ap.add_argument("--score", action="store_true",
                    help="AuraFace identity similarity: visitor photo vs generated face")
    ap.add_argument("--seed", type=int, default=G["seed"])
    ap.add_argument("--prompt", help="prompt override ({trigger}/{person} are filled in)")
    ap.add_argument("--no-pencil", action="store_true", help="skip the graphite post-process")
    ap.add_argument("--offload", action="store_true", help="CPU offload for small GPUs")
    ap.add_argument("--debug", action="store_true", help="save every intermediate image")
    ap.add_argument("--stage0-only", action="store_true", help="CPU inputs only, no diffusion")
    ap.add_argument("--out", help="output dir (default: output/<timestamp>)")
    args = ap.parse_args()

    steps = args.steps or (G["lcm_steps"] if args.lcm else G["steps"])
    guidance = args.guidance if args.guidance is not None else (
        G["lcm_guidance"] if args.lcm else G["guidance"])
    scenes = (_scenes_for(args.gender) if args.gender else args.scenes or all_scene_ids())
    people = {"male": "man", "female": "woman"}

    out = Path(args.out) if args.out else C.OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    dbg = out / "debug"
    (dbg if (args.debug or args.stage0_only) else out).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    photos = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in _face_paths(args.faces)]
    refs = reference_crops(photos)
    views = visitor_views(photos) if args.mode == "photo" else None
    ip_scale = args.ip_scale if args.ip_scale is not None else (
        G["pose_ip_scale"] if args.mode == "pose" else G["ip_scale"])
    ref_emb = face_embedding(photos[0]) if args.score else None
    print(f"{len(refs)}/{len(photos)} visitor photos have a face; scenes {scenes}")
    if views:
        print("views (photo, yaw): " + ", ".join(f"{v['index']}:{v['yaw']:+.0f}" for v in views))
    if args.debug or args.stage0_only:
        for i, r in enumerate(refs):
            r.save(dbg / f"ref_{i}.png")

    pipe = None
    if not args.stage0_only:
        t = time.perf_counter()
        from inpaint import build_pipeline
        pipe = build_pipeline(ip_adapter=args.ip_adapter, lcm=args.lcm, lora_path=args.lora,
                              lora_scale=args.lora_scale, offload=args.offload,
                              face_cn=args.mode == "pose", adapter_file=args.adapter_file)
        print(f"pipeline ready in {time.perf_counter() - t:.1f}s")

    report = {"settings": {**vars(args), "steps": steps, "guidance": guidance, "ip_scale": ip_scale},
              "scenes": {}}
    ip_embeds = None
    for sid in scenes:
        t = time.perf_counter()
        inp = build_inputs(ensure_scene_meta(sid), photos, crop_size=args.crop_size,
                           face_refs=refs, views=views, mode=args.mode, hair=args.hair)
        t_pre = time.perf_counter() - t
        if args.debug or args.stage0_only:
            for k in ("init_img", "mask_img", "lineart_img", "head_img", "mesh_img"):
                if inp[k] is not None:
                    inp[k].save(dbg / f"{sid}_{k.removesuffix('_img')}.png")
        person = people.get(args.gender or _scene_gender(sid), "person")
        prompt = (args.prompt or C.PROMPT).format(trigger=C.LORA_TRIGGER, person=person)
        entry = {"frontal": inp["frontal"], "crop_box": inp["crop_box"], "person": person,
                 "mode": inp["mode"], "hair": inp["hair"], "view": inp["view"],
                 "mesh_eye_error": inp["mesh_eye_error"],
                 "stage0_s": round(t_pre, 2)}
        if pipe is not None:
            from inpaint import aura_ip_embeds, averaged_ip_embeds, run_inpaint
            if ip_embeds is None:                     # identity is the same for every scene
                if args.ip_adapter in C.MODEL_PATHS["faceid_adapters"]:
                    ip_embeds = aura_ip_embeds(pipe, photos, refs, do_cfg=guidance > 1.0)
                else:
                    ip_embeds = averaged_ip_embeds(pipe, refs, do_cfg=guidance > 1.0)
            if inp["mode"] == "pose":
                cn = [args.face_cn_scale,
                      args.cn_scale if args.cn_scale is not None else G["pose_lineart_scale"]]
            else:
                cn = args.cn_scale if args.cn_scale is not None else (
                    G["cn_scale"] if inp["frontal"] else G["cn_scale_nonfrontal"])
                if args.mode == "pose":
                    # scene fell back to character mode on a two-ControlNet pipeline:
                    # blank mesh at scale 0
                    blank = Image.new("RGB", inp["lineart_img"].size)
                    inp["control_images"] = [blank, inp["lineart_img"]]
                    cn = [0.0, cn]
            t = time.perf_counter()
            composed, raw, final, paste_mask = run_inpaint(
                pipe, inp, prompt=prompt, negative=C.NEGATIVE, ip_scale=ip_scale,
                cn_scale=cn, steps=steps, guidance=guidance, seed=args.seed,
                pencil=not args.no_pencil, ip_embeds=ip_embeds)
            entry.update(cn_scale=cn, inpaint_s=round(time.perf_counter() - t, 2))
            if args.score:
                entry["identity"] = identity_score(ref_emb, raw)
            if args.compose_on in ("headless", "both"):
                cv2.imwrite(str(out / f"scene_{sid}.png"), composed)
            if args.compose_on in ("head", "both"):
                from inpaint import compose_on
                cv2.imwrite(str(out / f"scene_{sid}_onhead.png"),
                            compose_on(inp["head_bgr"], inp, final, paste_mask))
            if args.debug:
                raw.save(dbg / f"{sid}_raw.png")
                final.save(dbg / f"{sid}_final.png")
                paste_mask.save(dbg / f"{sid}_paste_mask.png")
        report["scenes"][sid] = entry
        print(f"scene {sid}: {entry}")

    report["total_s"] = round(time.perf_counter() - t0, 2)
    (out / "run.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"done in {report['total_s']}s -> {out}")


if __name__ == "__main__":
    main()
