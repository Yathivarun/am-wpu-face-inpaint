"""
Premade character assets for the target FRU path (offline, CPU, once per character).

    python build_assets.py --src ../../../posed-characters --out ../../../posed-characters/assets
    python build_assets.py ... --aspect 0.8      # 4:5 head crop (crown to neck), 512 x 640

Input: drawn characters, one per scene image (`M1.png`, `F3.png`, ...), or sheets of
several heads in a row (`M.png`, `F.png`: every face found becomes one character,
numbered left to right). Gender comes from the first letter of the file name.

Per character, in <out>/<id>/ (everything in the generation crop unless noted; --size is its
width, its height is width / --aspect):

    scene.png           the full scene (paste-back target)
    crop.png            the generation crop around the whole head (face box + room for hair,
                        bun, crown, headwear). --aspect 1: a square with side room;
                        --aspect 0.8: 4:5, crown to neck, so the head gets more of the pixels
    lineart_full.png    character lineart (white on black, the lineart ControlNet format)
    lineart_noface.png  same, with the face (mesh oval, dilated) erased: hair, ears,
                        headwear, neck and clothes stay, the face is left to identity
    lineart_contour.png the character's own face-edge lines kept (a CONTOUR_BAND strip just
                        inside the face oval: jaw, cheeks, hairline, profile nose and lips),
                        the interior erased: fixes the face's size and shape without its
                        eyes, nose or mouth (a drawn mesh outline cuts across turned faces).
                        Only up to CONTOUR_MAX_YAW: beyond it the mesh oval under-covers the
                        face, the strip would keep the character's eyes and nose, so the
                        file is lineart_noface (meta "contour": "band" / "noface")
    mesh.png            the character's 478-point face mesh in the MediaPipe-face
                        ControlNet format (pose and expression, no face shape detail).
                        Where MediaPipe fails (extreme profiles), AuraFace's 68 3D
                        landmarks drawn in the same colours (meta "mesh_source": "68")
    kps_feat.png        same map without the face outline (eyes, brows, lips): places the
                        features but leaves the face shape to identity
    kps_eyelip.png      eyes + lips only (the ControlNet has no nose group)
    kps_eyes.png        eyes (with pupils) only: position, scale and roll of the face
    depth.png           DPT-Hybrid depth of the crop (depth ControlNet format)
    depth_smooth.png    same, heavily blurred inside the face oval (head volume, not features)
    face_mask.png       the face oval, dilated (FACE_MASK_DILATE): where identity attention may act
    head_mask.png       the character's head for paste-back: the depth silhouette inside the
                        crown region (+ HEAD_ROOM for buns and crowns); face oval + crown without depth
    preview.jpg         crop | lineart_noface | mesh over crop | head mask over crop
    meta.json           crop box, face box, 5 keypoints, head pose (pitch, yaw, roll),
                        mesh eye error, gender, file list

Nothing here runs per visitor: at generation time only the visitor's AuraFace vector
and a face crop are computed.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config as C
from faces import draw_face_mesh, face_app, head_mask, oval_from_points
from lineart import make_lineart
from stage0 import _crown, character_mesh

# Generation crop around the AuraFace face box (× box height / width). Generous on
# purpose: buns, crowns and long hair must stay inside the crop.
TOP, BOTTOM, SIDE = 0.85, 0.40, 0.55
SIDE_PER_YAW = 0.50       # extra side room at 90° yaw (the back of a turned head)
MARGIN = 1.06             # final square × this
DEPTH_FACE_BLUR = 0.08    # depth_smooth: Gaussian sigma inside the face (× crop size)
# sparser keypoint maps for the +kps ControlNet (mesh.png = everything): which face-mesh
# groups each keeps. The MediaPipe-face ControlNet has no nose group, so "5 points"
# becomes eyes + lips.
_EYES = ["FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE"]
_BROWS = ["FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW"]
KPS_MAPS = {"kps_feat.png": _EYES + _BROWS + ["FACE_LANDMARKS_LIPS"],    # no face outline
            "kps_eyelip.png": _EYES + ["FACE_LANDMARKS_LIPS"],
            "kps_eyes.png": _EYES}
HEAD_ROOM = 0.10          # head mask: depth silhouette allowed this far past the crown (× crop size)
NECK = 0.35               # 4:5 head crop: below the chin (× face box height)
CONTOUR_BAND = 0.05       # lineart_contour: face-edge strip kept (× crop size)
CONTOUR_MAX_YAW = 30.0    # lineart_contour: beyond this |yaw| it is lineart_noface
FACE_MASK_DILATE = 0.04   # face_mask.png: face oval grown by this (× crop size)


def load_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def detect(rgb, sheet, at=None):
    """Characters in a scene (largest face, or the face nearest `at` = (x, y) in a crowd)
    or a sheet (every face, left to right)."""
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    faces = sorted(face_app().get(bgr), key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if not faces:
        return bgr, []
    if at is not None:
        return bgr, [min(faces, key=lambda f: np.hypot((f.bbox[0] + f.bbox[2]) / 2 - at[0], (f.bbox[1] + f.bbox[3]) / 2 - at[1]))]
    if not sheet:
        return bgr, faces[:1]
    big = (faces[0].bbox[3] - faces[0].bbox[1])
    faces = [f for f in faces if (f.bbox[3] - f.bbox[1]) > 0.5 * big]
    return bgr, sorted(faces, key=lambda f: f.bbox[0])


def isolate(rgb, faces, i):
    """Sheets: everything outside this face's panel (halfway to the neighbouring
    faces) painted with the paper colour, so crops and the mesh see one head."""
    a = np.array(rgb).copy()
    cx = [(f.bbox[0] + f.bbox[2]) / 2 for f in faces]
    lo = 0 if i == 0 else int((cx[i - 1] + cx[i]) / 2)
    hi = a.shape[1] if i == len(faces) - 1 else int((cx[i] + cx[i + 1]) / 2)
    paper = np.median(a[:, lo:hi].reshape(-1, 3)[::97], axis=0).astype(np.uint8)
    a[:, :lo] = paper
    a[:, hi:] = paper
    return Image.fromarray(a)


# 68-point (iBUG) groups -> MediaPipe-face ControlNet colours (BGR, subject's left/right)
_G68 = [("FACE_LANDMARKS_LEFT_EYE", range(42, 48), True), ("FACE_LANDMARKS_RIGHT_EYE", range(36, 42), True),
        ("FACE_LANDMARKS_LEFT_EYEBROW", range(22, 27), False), ("FACE_LANDMARKS_RIGHT_EYEBROW", range(17, 22), False),
        ("FACE_LANDMARKS_LIPS", range(48, 60), True), ("FACE_LANDMARKS_LIPS", range(60, 68), True)]


def face68_outline(p68):
    """Closed face outline from 68 points: jaw, then the brows raised to the hairline."""
    lift = 0.30 * (p68[8, 1] - p68[17:27, 1].mean())
    brows = p68[17:27][::-1] - np.array([0, lift], np.float32)
    return np.vstack([p68[0:17], brows])


def draw_face68(p68, size, groups=None):
    """68 landmarks drawn like faces.draw_face_mesh (same colours, line width, pupils;
    groups: only these _MESH_BGR groups, None = all)."""
    from faces import _MESH_BGR, _PUPILS_BGR
    img = np.zeros((size, size, 3), np.uint8)
    t = max(2, round(2 * size / 512))
    poly = lambda pts, closed, col: cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], closed, col, t)
    keep = lambda name: groups is None or name in groups
    if keep("FACE_LANDMARKS_FACE_OVAL"):
        poly(face68_outline(p68), True, _MESH_BGR["FACE_LANDMARKS_FACE_OVAL"])
    for name, idx, closed in _G68:
        if keep(name):
            poly(p68[list(idx)], closed, _MESH_BGR[name])
    if groups is not None and not any("_EYE" in g and "BROW" not in g for g in groups):
        return Image.fromarray(img[:, :, ::-1].copy())
    for idx, (a, b) in ((468, (36, 42)), (473, (42, 48))):
        x, y = p68[a:b].mean(axis=0).astype(int)
        img[max(0, y - t):y + t, max(0, x - t):x + t] = _PUPILS_BGR[idx]
    return Image.fromarray(img[:, :, ::-1].copy())


def crop_box(face, head_s, W, H, aspect=1.0):
    """Crop of width/height `aspect` around the head: the face box grown for hair,
    unioned with the mesh head region. 1.0 (square) also keeps the side room; below
    1.0 the box runs crown to neck (NECK below the chin) and is only as wide as the
    aspect allows (never narrower than the head)."""
    x1, y1, x2, y2 = face.bbox
    w, h = x2 - x1, y2 - y1
    yaw = abs(float(face.pose[1])) if getattr(face, "pose", None) is not None else 0.0
    side = SIDE + SIDE_PER_YAW * min(yaw, 90.0) / 90.0
    if aspect < 1.0:
        side *= 0.5      # 4:5: the head's own width, not the square's side room
    bx = [x1 - side * w, y1 - TOP * h, x2 + side * w, y2 + (BOTTOM if aspect >= 1.0 else NECK) * h]
    if head_s is not None and head_s.any():
        ys, xs = np.nonzero(head_s)
        bx = [min(bx[0], xs.min()), min(bx[1], ys.min()), max(bx[2], xs.max()), max(bx[3], ys.max())]
    cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    bh = max(bx[3] - bx[1], (bx[2] - bx[0]) / aspect) * MARGIN
    bh = min(bh, H, W / aspect)
    bw = bh * aspect
    cx, cy = min(max(cx, bw / 2), W - bw / 2), min(max(cy, bh / 2), H - bh / 2)
    x0, y0 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
    return [x0, y0, x0 + int(round(bw)), y0 + int(round(bh))]


def _draw(fn, pts, cw, ch, groups=None):
    """A square face-map drawer on a (cw, ch) crop: draw on the larger square, cut."""
    n = max(cw, ch)
    return Image.fromarray(np.array(fn(pts, n, groups))[:ch, :cw].copy())


_DEPTH = {}


def depth_map(rgb, cw, ch):
    """DPT-Hybrid (MiDaS) relative depth, normalised 0-255, as the depth ControlNet expects."""
    if "pipe" not in _DEPTH:
        from transformers import pipeline
        _DEPTH["pipe"] = pipeline("depth-estimation", model=str(C.MODELS_ROOT / "depth_estimator" / "dpt-hybrid-midas"))
    d = np.array(_DEPTH["pipe"](rgb)["depth"], np.float32)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-6) * 255
    return cv2.resize(d, (cw, ch), interpolation=cv2.INTER_CUBIC).astype(np.uint8)


def head_silhouette(dep, crown, oval, size, crop=None):
    """The character's real head outline: depth foreground (Otsu) inside the crown
    region grown by HEAD_ROOM (buns, crowns), the blob touching the face, holes
    filled. None if it looks wrong (smaller than the face oval)."""
    _, fg = cv2.threshold(dep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if crop is not None:   # sheets: the flat paint isolate() put outside the panel is never head
        g = np.array(crop.convert("L"), np.float32)
        var = cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2
        fg[cv2.dilate((var < 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
    r = max(1, int(HEAD_ROOM * size))
    region = cv2.dilate(crown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    sil = (fg & region) | oval
    n, lbl = cv2.connectedComponents((sil > 0).astype(np.uint8))
    hits = np.bincount(lbl[oval > 0].ravel(), minlength=n)
    hits[0] = 0
    sil = np.where(lbl == int(np.argmax(hits)), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(sil)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled if (filled > 0).sum() > 1.2 * (oval > 0).sum() else None


def overlay(base, mask_or_img, color=(255, 0, 0), alpha=0.45):
    a = np.array(base.convert("RGB")).astype(np.float32)
    if isinstance(mask_or_img, Image.Image):
        m = np.array(mask_or_img.convert("RGB")).astype(np.float32)
        on = m.max(axis=2, keepdims=True) > 0
        return Image.fromarray(np.where(on, m, a).astype(np.uint8))
    m = (mask_or_img > 0)[..., None]
    return Image.fromarray(np.where(m, a * (1 - alpha) + np.array(color) * alpha, a).astype(np.uint8))


def build_one(cid, gender, src, rgb, face, out, size, with_depth, aspect=1.0):
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
    bbox = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    kps = np.asarray(face.kps, np.float32).tolist()
    pts_s, eye_err = character_mesh(bgr, bbox, kps)
    mesh_ok = pts_s is not None and (eye_err is None or eye_err <= C.MESH_MAX_EYE_ERROR)
    p68_s = None
    if not mesh_ok:   # MediaPipe missed or fit another head: AuraFace's 68 landmarks instead
        pts_s = None
        if getattr(face, "landmark_3d_68", None) is not None:
            p68_s = np.asarray(face.landmark_3d_68, np.float32)[:, :2]
    if pts_s is not None:
        oval_s = oval_from_points(pts_s, H, W)
    elif p68_s is not None:
        oval_s = np.zeros((H, W), np.uint8)
        cv2.fillPoly(oval_s, [cv2.convexHull(face68_outline(p68_s).astype(np.int32))], 255)
    else:
        oval_s = None
    head_s = _crown(oval_s) if oval_s is not None else None

    box = crop_box(face, head_s, W, H, aspect)
    cw, ch = size, int(round(size / aspect / 8)) * 8
    k = cw / float(box[2] - box[0])
    crop = rgb.crop(tuple(box)).resize((cw, ch), Image.LANCZOS)
    size = min(cw, ch)                   # the reference length for every fraction below

    def to_crop(mask_s):
        return cv2.resize(mask_s[box[1]:box[3], box[0]:box[2]], (cw, ch), interpolation=cv2.INTER_NEAREST)

    if oval_s is not None:
        oval = to_crop(oval_s)
    else:   # no mesh: an ellipse on the face box
        oval = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(oval, (int(((x1 + x2) / 2 - box[0]) * k), int(((y1 + y2) / 2 - box[1]) * k)),
                    (int(bbox["w"] * k * 0.55), int(bbox["h"] * k * 0.6)), 0, 0, 360, 255, -1)
    pts_f = None if pts_s is None else (pts_s - np.array(box[:2], np.float32)) * k
    p68_f = None if p68_s is None else (p68_s - np.array(box[:2], np.float32)) * k

    d = out / cid
    d.mkdir(parents=True, exist_ok=True)
    files = {}

    def save(name, img):
        (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(d / name)
        files[name.split(".")[0]] = name

    scene = load_rgb(src)                 # the untouched scene (sheets: not the isolated panel)
    scene.save(d / "scene.png")
    files["scene"] = "scene.png"
    save("crop.png", crop)

    la = np.array(make_lineart(crop, size))
    save("lineart_full.png", Image.fromarray(la).convert("RGB"))
    de = max(1, int(C.FACE_ERASE_DILATE_FRAC * size))
    face_er = cv2.dilate(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * de + 1, 2 * de + 1)))
    la_nf = la.copy()
    la_nf[face_er > 0] = 0
    save("lineart_noface.png", Image.fromarray(la_nf).convert("RGB"))
    yaw = abs(float(face.pose[1])) if getattr(face, "pose", None) is not None else 90.0
    contour = "band" if yaw <= CONTOUR_MAX_YAW else "noface"
    la_ct = la_nf
    if contour == "band":
        cb = max(1, int(CONTOUR_BAND * size))
        la_ct = la.copy()
        la_ct[cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cb + 1, 2 * cb + 1))) > 0] = 0
    save("lineart_contour.png", Image.fromarray(la_ct).convert("RGB"))
    dm = max(1, int(FACE_MASK_DILATE * size))
    save("face_mask.png", cv2.dilate(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dm + 1, 2 * dm + 1))))

    mesh_img = None
    if pts_f is not None:
        mesh_img = _draw(draw_face_mesh, pts_f, cw, ch)
    elif p68_f is not None:
        mesh_img = _draw(draw_face68, p68_f, cw, ch)
    if mesh_img is not None:
        save("mesh.png", mesh_img)
        for name, groups in KPS_MAPS.items():
            if pts_f is not None:
                save(name, _draw(draw_face_mesh, pts_f, cw, ch, groups))
            else:
                save(name, _draw(draw_face68, p68_f, cw, ch, groups))

    if with_depth:
        dep = depth_map(crop, cw, ch)
        save("depth.png", Image.fromarray(dep).convert("RGB"))
        sig = DEPTH_FACE_BLUR * size
        blur = cv2.GaussianBlur(dep, (0, 0), sig)
        w8 = cv2.GaussianBlur((face_er > 0).astype(np.float32), (0, 0), sig / 3)
        save("depth_smooth.png", Image.fromarray((dep * (1 - w8) + blur * w8).astype(np.uint8)).convert("RGB"))

    hm = _crown(oval) if oval is not None else oval
    try:
        hm = hm | head_mask(crop)        # segmenter hair/face: catches buns and headwear past the crown
    except Exception as e:                # noqa: BLE001 — the crown mask alone still works
        print(f"    [warn] {cid}: head segmenter failed ({e})")
    if with_depth and oval is not None:
        sil = head_silhouette(dep, hm, oval, size, crop)
        if sil is not None:
            hm = sil
    dl = max(1, int(C.MASK_DILATE_FRAC * size))
    hm = cv2.dilate(hm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dl + 1, 2 * dl + 1)))
    save("head_mask.png", hm)

    tiles = [crop, Image.fromarray(la_nf).convert("RGB"),
             Image.fromarray(la_ct).convert("RGB"),
             overlay(crop, mesh_img) if mesh_img is not None else crop, overlay(crop, hm)]
    prev = Image.new("RGB", (cw * len(tiles), ch), "white")
    for i, t in enumerate(tiles):
        prev.paste(t, (i * cw, 0))
    prev.resize((prev.width // 2, prev.height // 2)).save(d / "preview.jpg", quality=90)

    pose = [float(v) for v in face.pose] if getattr(face, "pose", None) is not None else None
    meta = {
        "id": cid, "gender": gender, "source": src.name, "scene_size": [W, H], "size": [cw, ch],
        "crop_box": box, "face_bbox": bbox, "kps": kps,
        "pose": None if pose is None else {"pitch": pose[0], "yaw": pose[1], "roll": pose[2]},
        "mesh": mesh_img is not None, "mesh_ok": bool(mesh_ok), "contour": contour,
        "mesh_source": "mediapipe" if pts_s is not None else ("68" if p68_s is not None else None),
        "mesh_eye_error": None if eye_err is None else round(float(eye_err), 3),
        "files": files,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder of character images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512, help="generation crop width")
    ap.add_argument("--aspect", type=float, default=1.0,
                    help="crop width / height: 1 = square with side room, 0.8 = 4:5 crown to neck")
    ap.add_argument("--sheets", nargs="*", default=["M", "F"], help="file stems that are multi-head sheets")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--face-at", nargs="*", default=[], metavar="STEM=X,Y",
                    help="crowded scenes: use the face nearest this scene point, e.g. M5=1128,609")
    args = ap.parse_args()
    face_at = {k: tuple(float(c) for c in v.split(",")) for k, v in (a.split("=") for a in args.face_at)}

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        gender = {"m": "male", "f": "female"}.get(p.stem[0].lower())
        if gender is None:
            print(f"{p.name}: name must start with M or F, skipped")
            continue
        sheet = p.stem in args.sheets
        rgb = load_rgb(p)
        _, faces = detect(rgb, sheet, face_at.get(p.stem))
        if not faces:
            print(f"{p.name}: no face found, skipped")
            continue
        for i, f in enumerate(faces, 1):
            cid = f"{p.stem}_s{i}" if sheet else p.stem
            m = build_one(cid, gender, p, isolate(rgb, faces, i - 1) if sheet else rgb, f, out,
                          args.size, not args.no_depth, args.aspect)
            yaw = m["pose"]["yaw"] if m["pose"] else float("nan")
            fh = m["face_bbox"]["h"] * m["size"][0] / (m["crop_box"][2] - m["crop_box"][0])
            print(f"{cid}: yaw {yaw:+.0f}  crop {m['size'][0]}x{m['size'][1]}  face {fh:.0f} px tall  "
                  f"mesh {m['mesh_source']} (MediaPipe eye err {m['mesh_eye_error']})")
            index.append({k: m[k] for k in ("id", "gender", "source", "pose", "mesh_source")})
    (out / "index.json").write_text(json.dumps(index, indent=2))

    # one contact sheet of every preview, for review
    prevs = [Image.open(out / c["id"] / "preview.jpg") for c in index]
    if prevs:
        w, h = prevs[0].size
        sheet = Image.new("RGB", (w, h * len(prevs)), "white")
        for i, im in enumerate(prevs):
            sheet.paste(im, (0, i * h))
        sheet.thumbnail((w, 6000))
        sheet.save(out / "assets_overview.jpg", quality=88)
    print(f"{len(index)} characters -> {out}")


if __name__ == "__main__":
    main()
