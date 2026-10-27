"""
Face detection and face masks (CPU only, no VRAM).

  • AuraFace (insightface pack, Apache-2.0) — detects faces in visitor photos
    and scenes, and gives the scene character's head pose. Only the detector
    and the 3D-landmark model are loaded; identity comes from the IP-Adapter,
    not from an ArcFace embedding.
  • MediaPipe FaceLandmarker — the face-oval mask, which works on drawn faces.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import (ID_CROP_BOT_FRAC, ID_CROP_SIDE_FRAC, ID_CROP_TOP_FRAC,
                    MODEL_PATHS, VIEW_CROP_FRAC, VIEW_FACE_PX)

_CACHE = {"app": None, "landmarker": None, "segmenter": None}

# MediaPipe face-mesh landmark indices (same as sdxl_ok / dev face_parser).
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
_INNER_FEATURES = [
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
     181, 91, 146],
]


def face_app():
    """CPU AuraFace: detection + 3D landmarks (head pose)."""
    if _CACHE["app"] is None:
        from insightface.app import FaceAnalysis
        pack = Path(MODEL_PATHS["auraface"])
        app = FaceAnalysis(name=pack.name, root=str(pack.parent.parent),
                           allowed_modules=["detection", "landmark_3d_68"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _CACHE["app"] = app
    return _CACHE["app"]


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def reference_crops(images):
    """Hair-inclusive face crop of each visitor photo (PIL RGB) -> list of PIL RGB.

    Padded upward so the hairstyle is inside the IP-Adapter reference. Photos
    without a detectable face are skipped; raises if none has one."""
    app = face_app()
    crops = []
    for img in images:
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        face = largest_face(app.get(bgr))
        if face is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        s = max(x2 - x1, y2 - y1)
        h, w = bgr.shape[:2]
        cx1 = max(0, int(x1 - ID_CROP_SIDE_FRAC * s))
        cx2 = min(w, int(x2 + ID_CROP_SIDE_FRAC * s))
        cy1 = max(0, int(y1 - ID_CROP_TOP_FRAC * s))
        cy2 = min(h, int(y2 + ID_CROP_BOT_FRAC * s))
        crops.append(Image.fromarray(cv2.cvtColor(bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)))
    if not crops:
        raise RuntimeError("no face detected in any visitor image")
    return crops


def _segmenter():
    if _CACHE["segmenter"] is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATHS["mediapipe_parser"]),
            output_category_mask=True)
        _CACHE["segmenter"] = vision.ImageSegmenter.create_from_options(opts)
    return _CACHE["segmenter"]


def head_mask(img):
    """Hair + face mask (uint8, 255) of a photo: multiclass segmenter hair (1)
    and face skin (3), plus the face-mesh oval, holes filled, largest blob."""
    import mediapipe as mp
    rgb = np.ascontiguousarray(np.array(img.convert("RGB")))
    cat = np.squeeze(_segmenter().segment(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).category_mask.numpy_view())
    mask = np.where((cat == 1) | (cat == 3), 255, 0).astype(np.uint8)
    oval = face_oval_mask(img)
    if oval is not None:
        mask |= oval
    k = max(3, int(0.02 * max(mask.shape)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n > 1:
        mask = np.where(lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    return filled


def visitor_views(images):
    """Per visitor photo: a head crop (face scaled to ~VIEW_FACE_PX), its 5
    keypoints and hair+face mask in crop coordinates, and the head yaw. Used to
    draw the visitor's own face and hair lines into the character's head;
    the view whose yaw best matches each character is chosen per scene."""
    app = face_app()
    views = []
    for idx, img in enumerate(images):
        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        face = largest_face(app.get(bgr))
        if face is None or getattr(face, "kps", None) is None:
            continue
        x1, y1, x2, y2 = face.bbox
        fh = max(y2 - y1, x2 - x1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - 0.15 * fh     # centre a little up: hair
        half = VIEW_CROP_FRAC * fh / 2
        box = [int(cx - half), int(cy - half), int(cx + half), int(cy + half)]
        # pad instead of clipping so the keypoints keep their geometry
        h, w = bgr.shape[:2]
        pad = [max(0, -box[0]), max(0, -box[1]), max(0, box[2] - w), max(0, box[3] - h)]
        padded = cv2.copyMakeBorder(bgr, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_REPLICATE)
        crop = padded[box[1] + pad[1]:box[3] + pad[1], box[0] + pad[0]:box[2] + pad[0]]
        scale = VIEW_FACE_PX / fh
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        kps = (np.asarray(face.kps, np.float32) - [box[0], box[1]]) * scale
        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pose = getattr(face, "pose", None)
        views.append({"index": idx, "rgb": rgb, "kps": kps, "mask": head_mask(rgb),
                      "yaw": float(pose[1]) if pose is not None else 0.0})
    return views


def _landmarker():
    if _CACHE["landmarker"] is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=MODEL_PATHS["mediapipe_face_landmarker"]),
            num_faces=1)
        _CACHE["landmarker"] = vision.FaceLandmarker.create_from_options(opts)
    return _CACHE["landmarker"]


def face_landmarks(img):
    """The 478 MediaPipe face-mesh landmarks (normalised x, y) of the largest
    face in a PIL image, or None. Works on drawn (sketch) faces."""
    import mediapipe as mp
    rgb = np.ascontiguousarray(np.array(img.convert("RGB")))
    try:
        res = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception as e:  # noqa: BLE001 — callers fall back when there is no mesh
        print(f"    [warn] face landmarker failed ({e})")
        return None
    return res.face_landmarks[0] if res.face_landmarks else None


def landmarks_px(landmarks, w, h):
    """Normalised landmarks -> (N, 2) float pixel coordinates."""
    return np.array([[p.x * w, p.y * h] for p in landmarks], np.float32)


def face_oval_mask(img):
    """Face oval + inner features mask (uint8, 255 = face) for a PIL image, or
    None when no face mesh is found."""
    lm = face_landmarks(img)
    if lm is None:
        return None
    return oval_from_points(landmarks_px(lm, *img.size), img.size[1], img.size[0])


def oval_from_points(pts, h, w):
    """Face oval + inner features mask (uint8, 255) from pixel landmarks."""
    mask = np.zeros((h, w), np.uint8)
    for idx in [_FACE_OVAL, *_INNER_FEATURES]:
        cv2.fillPoly(mask, [pts[idx].astype(np.int32).reshape(-1, 1, 2)], 255)
    # seal holes: fill the outer contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sealed = np.zeros_like(mask)
    cv2.drawContours(sealed, cnts, -1, 255, cv2.FILLED)
    return sealed if sealed.max() > 0 else None


# CrucibleAI ControlNetMediaPipeFace annotator spec (controlnet_aux
# mediapipe_face_common.py): colours are BGR, drawn on black, then the image is
# channel-reversed to RGB — reproduced exactly so the ControlNet sees what it
# was trained on.
_MESH_BGR = {
    "FACE_LANDMARKS_FACE_OVAL":      (10, 200, 10),
    "FACE_LANDMARKS_LEFT_EYE":       (180, 200, 10),
    "FACE_LANDMARKS_LEFT_EYEBROW":   (180, 220, 10),
    "FACE_LANDMARKS_RIGHT_EYE":      (10, 200, 180),
    "FACE_LANDMARKS_RIGHT_EYEBROW":  (10, 220, 180),
    "FACE_LANDMARKS_LIPS":           (10, 180, 10),
}
_PUPILS_BGR = {468: (10, 200, 250), 473: (250, 200, 10)}   # right iris, left iris


def draw_face_mesh(pts, size, groups=None):
    """Face-mesh control image (RGB, size×size) in the ControlNetMediaPipeFace
    format, from (N, 2) pixel landmarks. Line width scales with size (2 px at
    the 512 px training size). groups: only these _MESH_BGR groups (pupils are
    drawn with the eyes); None = all."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC
    img = np.zeros((size, size, 3), np.uint8)
    thick = max(2, round(2 * size / 512))
    half = max(2, round(2 * size / 512))
    pt = [(int(x), int(y)) for x, y in pts]
    for name, color in _MESH_BGR.items():
        if groups is not None and name not in groups:
            continue
        for c in getattr(FLC, name):
            cv2.line(img, pt[c.start], pt[c.end], color, thick)
    for idx, color in _PUPILS_BGR.items():
        if idx < len(pt) and (groups is None or any("_EYE" in g and "BROW" not in g for g in groups)):
            x, y = pt[idx]
            img[max(0, y - half):y + half, max(0, x - half):x + half] = color
    return Image.fromarray(img[:, :, ::-1].copy())


def identity_app():
    """AuraFace detection + recognition (glintr100, Apache-2.0), for --score."""
    if _CACHE.get("id_app") is None:
        from insightface.app import FaceAnalysis
        pack = Path(MODEL_PATHS["auraface"])
        app = FaceAnalysis(name=pack.name, root=str(pack.parent.parent),
                           allowed_modules=["detection", "recognition"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _CACHE["id_app"] = app
    return _CACHE["id_app"]


def face_embedding(img):
    """Normalised AuraFace embedding of the largest face in a PIL image, or None."""
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    face = largest_face(identity_app().get(bgr))
    return None if face is None else face.normed_embedding


def identity_score(ref_embedding, img):
    """Cosine similarity between a visitor embedding and the face in `img`."""
    emb = face_embedding(img)
    if ref_embedding is None or emb is None:
        return None
    return round(float(np.dot(ref_embedding, emb)), 3)
