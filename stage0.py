"""
Stage 0 (CPU): scene metadata + visitor photos -> inpaint inputs, in memory.

Two passes so the generation budget goes to the head, not the background:

    pass 1  wide crop (5× the face box) of the head scene → the character's
            MediaPipe face mesh, mapped to scene coordinates. Detected once and
            reused for the oval, the mask and the pose mesh; a wide crop fits
            turned and bowed drawn faces far better than a tight one. The mesh
            is checked against AuraFace's eye keypoints.
            Head region: oval + crown ellipse (pose, character) or the visitor's
            hair+face mask aligned onto the character's face (photo).
    pass 2  square crop fitted to that region (+ HEAD_MARGIN_FRAC), generated
            at crop_size. Lineart is extracted at this resolution from the
            crops themselves, so lines stay crisp instead of being upscaled.

Modes (config.MODE):

    pose       the character's face mesh (pose, expression) drives a
               MediaPipe-face ControlNet; the character's lineart has the face
               erased. The visitor supplies identity only (IP-Adapter), so they
               are drawn *as* the character. `hair`: "character" keeps the
               character's hair lines, "visitor" erases them inside the head.
    photo      inside the head region the character's lines are replaced by
               the visitor's own lineart (face + hair), taken from the photo
               whose yaw best matches the character and aligned on the five
               face keypoints. Keeps the photo's pose.
    character  sdxl_ok behaviour: the character's hair/jaw lines stay, the
               eyes/nose/mouth lines are erased on frontal characters.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config as C
from faces import draw_face_mesh, face_landmarks, landmarks_px, oval_from_points, reference_crops
from lineart import make_lineart

_IRIS = [468, 473]


def is_frontal(face_angle):
    if not face_angle:
        return True
    return (abs(face_angle.get("yaw", 0.0)) <= C.FRONTAL_MAX_YAW
            and abs(face_angle.get("pitch", 0.0)) <= C.FRONTAL_MAX_PITCH
            and abs(face_angle.get("roll", 0.0)) <= C.FRONTAL_MAX_ROLL)


def compute_crop_box(bbox, W, H, context_frac):
    """Square crop centred on the face bbox, padded for context, clipped to the image."""
    return _square_box(bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"],
                       1 + 2 * context_frac, W, H)


def _square_box(x1, y1, x2, y2, factor, W, H):
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = int(min(max(max(x2 - x1, y2 - y1) * factor, C.MIN_CROP_SIDE), W, H))
    half = side / 2.0
    cx = min(max(cx, half), W - half)
    cy = min(max(cy, half), H - half)
    bx, by = int(round(cx - half)), int(round(cy - half))
    return [bx, by, bx + side, by + side], side


def _crop(img, box, size, interp=cv2.INTER_LANCZOS4):
    return cv2.resize(img[box[1]:box[3], box[0]:box[2]], (size, size), interpolation=interp)


def _to_scene(mask, box, W, H):
    """A mask drawn in a crop -> the same mask in scene coordinates."""
    side = box[2] - box[0]
    out = np.zeros((H, W), np.uint8)
    out[box[1]:box[3], box[0]:box[2]] = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)
    return out


def _crown(oval):
    """Character head = face oval + crown ellipse (hair headroom), same as sdxl_ok."""
    ox, oy, ow, oh = cv2.boundingRect((oval > 0).astype(np.uint8))
    head = oval.copy()
    cv2.ellipse(head, (ox + ow // 2, oy + int(oh * C.MASK_CROWN_CENTER_FRAC)),
                (int(ow * C.MASK_CROWN_W_FRAC), int(oh * C.MASK_CROWN_H_FRAC)), 0, 0, 360, 255, -1)
    return head


def pick_view(views, face_angle):
    """The visitor photo whose head yaw is closest to the character's."""
    yaw = (face_angle or {}).get("yaw", 0.0)
    return min(views, key=lambda v: abs(v["yaw"] - yaw))


def _similarity(src_kps, dst_kps):
    """Photo keypoints -> scene keypoints (rotation, uniform scale, translation)."""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(src_kps, np.float32),
                                       np.asarray(dst_kps, np.float32), method=cv2.LMEDS)
    if M is None:   # fall back to the two eyes
        (a, b), (c, d) = np.asarray(src_kps[:2], np.float32), np.asarray(dst_kps[:2], np.float32)
        s = np.linalg.norm(d - c) / max(np.linalg.norm(b - a), 1e-3)
        ang = np.arctan2(*(d - c)[::-1]) - np.arctan2(*(b - a)[::-1])
        R = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        M = np.hstack([R, (c - R @ a)[:, None]]).astype(np.float32)
    return M


def character_mesh(head_bgr, bbox, kps):
    """The character's 478 face landmarks in scene pixels, and their eye error
    against AuraFace's eye keypoints as a fraction of the eye distance.

    MediaPipe's fit on drawn, turned or bowed faces depends on framing, so the
    mesh is detected on a few crops (MESH_TRIALS: context, size) and the one
    that agrees best with AuraFace is kept. Without keypoints the first
    detection wins and the error is None. Returns (None, None) if no mesh."""
    H, W = head_bgr.shape[:2]
    ref = np.asarray(kps[:2], np.float32) if kps else None
    best = (None, None)
    for ctx, n in C.MESH_TRIALS:
        box, side = compute_crop_box(bbox, W, H, ctx)
        lm = face_landmarks(Image.fromarray(cv2.cvtColor(_crop(head_bgr, box, n), cv2.COLOR_BGR2RGB)))
        if lm is None:
            continue
        pts = landmarks_px(lm, n, n) * (side / n) + np.array([box[0], box[1]], np.float32)
        if ref is None:
            return pts, None
        eyes = pts[_IRIS]
        d = min(np.linalg.norm(eyes - ref, axis=1).sum(), np.linalg.norm(eyes[::-1] - ref, axis=1).sum()) / 2
        err = float(d / max(np.linalg.norm(ref[0] - ref[1]), 1e-3))
        if best[1] is None or err < best[1]:
            best = (pts, err)
    return best


def build_inputs(meta_dir, face_images=None, crop_size=C.GENERATION["crop_size"],
                 face_refs=None, views=None, mode=C.MODE, hair=C.HAIR):
    """Returns a dict: init_img, mask_img, lineart_img, head_img, mesh_img
    (pose mode), control_images (ordered as the pipeline's ControlNets),
    face_refs, crop_box, scene_bgr (headless, full), head_bgr (with head, full),
    feather_px, crop_size,
    frontal, mode, hair, view (index of the photo used, photo mode).

    Pass `face_refs` / `views` to reuse the visitor analysis across scenes."""
    meta_dir = Path(meta_dir)
    meta = json.loads((meta_dir / "meta.json").read_text())
    head_bgr = cv2.imread(str(meta_dir / meta["scene_file"]))
    if head_bgr is None:
        raise RuntimeError(f"cannot read {meta_dir / meta['scene_file']}")
    H, W = head_bgr.shape[:2]
    # sdxl_ok's cached meta holds an absolute server path; the scene id is enough.
    headless_path = C.FACE_DIR / f"{meta['scene_id']}.png"
    headless_bgr = cv2.imread(str(headless_path))
    if headless_bgr is None:
        raise FileNotFoundError(f"headless scene not found: {headless_path}")
    if headless_bgr.shape[:2] != (H, W):
        headless_bgr = cv2.resize(headless_bgr, (W, H), interpolation=cv2.INTER_LANCZOS4)

    if face_refs is None:
        face_refs = reference_crops(face_images)
    if mode == "photo" and views is None:
        from faces import visitor_views
        views = visitor_views(face_images)
    kps = meta.get("face_kps")
    if mode == "photo" and (not views or not kps):
        print("    [warn] no visitor view or character keypoints; using mode=character")
        mode = "character"

    # ── pass 1: character face mesh on a wide crop → scene coordinates ───────
    bbox = meta["face_bbox"]
    pts_s, eye_err = character_mesh(head_bgr, bbox, kps)
    mesh_ok = pts_s is not None and (eye_err is None or eye_err <= C.MESH_MAX_EYE_ERROR)
    if pts_s is None:
        print("    [warn] no face mesh on the character; using a padded rectangle")
        oval_s = np.zeros((H, W), np.uint8)
        px, py = int(bbox["w"] * C.MASK_PAD_FRAC), int(bbox["h"] * C.MASK_PAD_FRAC)
        cv2.rectangle(oval_s, (bbox["x"] - px, bbox["y"] - py),
                      (bbox["x"] + bbox["w"] + px, bbox["y"] + bbox["h"] + py), 255, -1)
        char_head_s = oval_s
    else:
        oval_s = oval_from_points(pts_s, H, W)
        char_head_s = _crown(oval_s)
    if mode == "pose" and not mesh_ok:
        why = "no mesh" if pts_s is None else f"mesh off AuraFace eyes by {eye_err:.2f}× eye distance"
        print(f"    [warn] {why}; using mode=character")
        mode = "character"

    view, M = None, None
    if mode == "photo":
        view = pick_view(views, meta.get("face_angle"))
        M = _similarity(view["kps"], kps)                       # photo crop -> scene
        vis_s = cv2.warpAffine(view["mask"], M, (W, H), flags=cv2.INTER_NEAREST)
        region_s = oval_s | vis_s
    else:
        region_s = char_head_s

    # ── pass 2: square crop fitted to the head region ────────────────────────
    ys, xs = np.nonzero(region_s)
    box, side = _square_box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1,
                            1 + 2 * C.HEAD_MARGIN_FRAC, W, H)
    scale = crop_size / float(side)
    head_img = Image.fromarray(cv2.cvtColor(_crop(head_bgr, box, crop_size), cv2.COLOR_BGR2RGB))
    init_img = Image.fromarray(cv2.cvtColor(_crop(headless_bgr, box, crop_size), cv2.COLOR_BGR2RGB))
    region = _crop(region_s, box, crop_size, cv2.INTER_NEAREST)

    la = np.array(make_lineart(head_img, crop_size))            # character lineart, crisp
    frontal = is_frontal(meta.get("face_angle"))
    mesh_img = None
    pts_f = None if pts_s is None else (pts_s - np.array([box[0], box[1]], np.float32)) * scale
    oval = None if pts_f is None else oval_from_points(pts_f, crop_size, crop_size)
    if mode == "pose":
        mesh_img = draw_face_mesh(pts_f, crop_size)
        mask = _crown(oval)
        if hair == "visitor":
            la[mask > 0] = 0                                   # every character line in the head
        else:
            de = max(1, int(C.FACE_ERASE_DILATE_FRAC * crop_size))
            face = cv2.dilate(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * de + 1, 2 * de + 1)))
            la[face > 0] = 0                                   # the mesh supplies the face
        dl = max(1, int(C.MASK_DILATE_FRAC * crop_size))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dl + 1, 2 * dl + 1)))
    elif mode == "photo":
        # scene -> crop, composed with photo -> scene
        S = np.array([[scale, 0, -box[0] * scale], [0, scale, -box[1] * scale], [0, 0, 1]], np.float32)
        Mf = (S @ np.vstack([M, [0, 0, 1]]))[:2]
        warped = cv2.warpAffine(np.array(view["rgb"]), Mf, (crop_size, crop_size),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        vmask = cv2.warpAffine(view["mask"], Mf, (crop_size, crop_size), flags=cv2.INTER_NEAREST)
        vla = np.array(make_lineart(Image.fromarray(warped), crop_size))
        la[region > 0] = 0                                     # drop the character's head lines
        keep = cv2.dilate(vmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        la = np.maximum(la, np.where(keep > 0, vla, 0).astype(np.uint8))
        dl = max(1, int(C.VISITOR_MASK_DILATE_FRAC * crop_size))
        mask = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dl + 1, 2 * dl + 1)))
    else:
        mask = _crown(oval) if oval is not None else region
        if frontal and oval is not None:
            ox, oy, ow, oh = cv2.boundingRect((oval > 0).astype(np.uint8))
            inner = np.zeros_like(la)
            cv2.ellipse(inner, (ox + ow // 2, oy + int(oh * C.FACE_INTERIOR_CY_FRAC)),
                        (int(ow * C.FACE_INTERIOR_W_FRAC), int(oh * C.FACE_INTERIOR_H_FRAC)),
                        0, 0, 360, 255, -1)
            la[inner > 0] = 0
        dl = max(1, int(C.MASK_DILATE_FRAC * crop_size))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dl + 1, 2 * dl + 1)))

    fk = max(1, int(C.MASK_FEATHER_FRAC * crop_size))
    fk += (fk + 1) % 2
    mask = (cv2.GaussianBlur(mask, (fk, fk), 0) > 127).astype(np.uint8) * 255

    lineart_img = Image.fromarray(la).convert("RGB")
    return {
        "init_img": init_img,
        "mask_img": Image.fromarray(mask).convert("L"),
        "lineart_img": lineart_img,
        "mesh_img": mesh_img,
        "control_images": [mesh_img, lineart_img] if mode == "pose" else [lineart_img],
        "head_img": head_img,
        "face_refs": face_refs,
        "crop_box": box,
        "scene_bgr": headless_bgr,
        "head_bgr": head_bgr,            # same size as scene_bgr; paste target for --compose-on head
        "feather_px": C.PASTE_FEATHER_PX,
        "crop_size": crop_size,
        "frontal": frontal,
        "mode": mode,
        "hair": hair if mode == "pose" else None,
        "view": view["index"] if view else None,
        "mesh_eye_error": None if eye_err is None else round(eye_err, 3),
    }
