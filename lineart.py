"""White-on-black lineart (controlnet_aux LineartDetector) — the annotator the
SD1.5 lineart ControlNet was trained on."""

from PIL import Image, ImageEnhance

from config import MODEL_PATHS

_CACHE = {"detector": None}


def make_lineart(img, resolution):
    """Lineart of `img` (PIL), returned as an "L" image of the same size."""
    if _CACHE["detector"] is None:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            # controlnet_aux imports mp.solutions at import time; the Tasks-only
            # mediapipe build lacks it and we never use that detector.
            from unittest.mock import MagicMock
            mp.solutions = MagicMock()
        from controlnet_aux import LineartDetector
        _CACHE["detector"] = LineartDetector.from_pretrained(MODEL_PATHS["annotators_lineart"])
    edge = _CACHE["detector"](img.convert("RGB"), coarse=False, detect_resolution=resolution,
                              image_resolution=resolution, output_type="pil")
    edge = ImageEnhance.Contrast(edge).enhance(1.2).convert("L")
    return edge if edge.size == img.size else edge.resize(img.size, Image.BILINEAR)
