"""
Per-scene metadata for the face inpaint, derived once and cached.

Each scene is a pair: <id>-head.png (character with head) and <id>.png (the
same scene with that head removed). Faces are detected in both; the head-scene
face with no match in the headless scene is the character. From it we cache:

    <FACE_DIR>/_meta/<id>/
        scene.png         copy of <id>-head.png (geometry source)
        lineart_map.png   full-scene lineart of the head version
        meta.json         face_bbox, face_kps, face_angle, ...

The cache format matches sdxl_ok's, so its _meta/ folder is reused as-is.

    python scene_meta.py                        # all scenes, generate if missing
    python scene_meta.py --scene 3 --force      # re-derive one scene
    python scene_meta.py --scene 3 --force --manual-xy 963,994   # pick the face
"""

import argparse
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from config import BBOX_IOU_MATCH, FACE_DIR, META_DIRNAME
from faces import face_app
from lineart import make_lineart


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(f):
    return (f.bbox[0] + f.bbox[2]) / 2.0, (f.bbox[1] + f.bbox[3]) / 2.0


def select_character_face(head_faces, headless_faces, W, H, manual_xy=None):
    """Manual point -> face removed in the headless render -> centre-most."""
    if not head_faces:
        return None
    if manual_xy is not None:
        mx, my = manual_xy
        inside = [f for f in head_faces
                  if f.bbox[0] <= mx <= f.bbox[2] and f.bbox[1] <= my <= f.bbox[3]]
        pool = inside or head_faces
        return min(pool, key=lambda f: (_center(f)[0] - mx) ** 2 + (_center(f)[1] - my) ** 2)
    removed = [hf for hf in head_faces
               if all(_iou(hf.bbox, gf.bbox) < BBOX_IOU_MATCH for gf in headless_faces)]
    pool = removed or list(head_faces)
    cx, cy = W / 2.0, H / 2.0
    return min(pool, key=lambda f: ((_center(f)[0] - cx) ** 2 + (_center(f)[1] - cy) ** 2,
                                    -float(getattr(f, "det_score", 0.0))))


def meta_dir(scene_id):
    return FACE_DIR / META_DIRNAME / str(scene_id)


def generate_scene_meta(scene_id, force=False, manual_xy=None):
    scene_id = str(scene_id)
    head_path = FACE_DIR / f"{scene_id}-head.png"
    headless_path = FACE_DIR / f"{scene_id}.png"
    for p in (head_path, headless_path):
        if not p.exists():
            raise FileNotFoundError(f"missing scene image: {p}")

    out = meta_dir(scene_id)
    if (out / "meta.json").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)

    head_bgr = cv2.imread(str(head_path))
    headless_bgr = cv2.imread(str(headless_path))
    H, W = head_bgr.shape[:2]
    if headless_bgr.shape[:2] != (H, W):
        headless_bgr = cv2.resize(headless_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    app = face_app()
    head_faces = app.get(head_bgr) or []
    headless_faces = app.get(headless_bgr) or []
    chosen = select_character_face(head_faces, headless_faces, W, H, manual_xy)
    if chosen is None:
        raise RuntimeError(f"no character face in {head_path.name}; pass --manual-xy")

    x1, y1, x2, y2 = (int(v) for v in chosen.bbox)
    face_angle = None
    if getattr(chosen, "pose", None) is not None:
        pitch, yaw, roll = (float(v) for v in chosen.pose)
        face_angle = {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)}
    kps = getattr(chosen, "kps", None)
    removed = all(_iou(chosen.bbox, gf.bbox) < BBOX_IOU_MATCH for gf in headless_faces)

    cv2.imwrite(str(out / "scene.png"), head_bgr)
    edge = make_lineart(Image.fromarray(cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)),
                        resolution=max(H, W))
    edge = cv2.resize(np.array(edge), (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out / "lineart_map.png"), edge)

    meta = {
        "scene_id": scene_id,
        "scene_file": "scene.png",
        "headless_file": headless_path.name,
        "resolution": {"w": W, "h": H},
        "face_bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
        "face_kps": ([[round(float(x), 2), round(float(y), 2)] for x, y in kps]
                     if kps is not None else None),
        "face_angle": face_angle,
        "lineart_map_file": "lineart_map.png",
        "face_selection": ("manual" if manual_xy is not None
                           else "removed-in-headless" if removed else "center-most"),
        "n_head_faces": len(head_faces),
        "n_headless_faces": len(headless_faces),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def ensure_scene_meta(scene_id):
    out = meta_dir(scene_id)
    return out if (out / "meta.json").exists() else generate_scene_meta(scene_id)


def all_scene_ids():
    return sorted(int(p.stem) for p in FACE_DIR.glob("*.png") if p.stem.isdigit())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate/cache face-scene metadata.")
    ap.add_argument("--scene", type=int, help="scene id (default: all)")
    ap.add_argument("--force", action="store_true", help="re-derive even if cached")
    ap.add_argument("--manual-xy", help="'x,y' on the character's face (with --scene)")
    args = ap.parse_args()
    mxy = tuple(int(v) for v in args.manual_xy.split(",")) if args.manual_xy else None
    for sid in ([args.scene] if args.scene else all_scene_ids()):
        d = generate_scene_meta(sid, force=args.force or mxy is not None, manual_xy=mxy)
        m = json.loads((d / "meta.json").read_text())
        print(f"scene {sid}: bbox={m['face_bbox']} angle={m['face_angle']} "
              f"selection={m['face_selection']}")
