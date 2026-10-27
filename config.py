"""
Configuration for the SD1.5 face-inpaint lab.

SD1.5 port of sdxl_ok's `fri` place (inpaint face swap): the visitor's face is
inpainted into a pre-drawn character's head, then pasted back into the scene.
Standalone — nothing here imports dev/config.py — but model paths follow the
same layout as dev/, so it runs against dev/models/ as-is.

Nothing downloads at runtime (local_files_only everywhere). Provision the two
missing IP-Adapter weights with fetch_models.py.
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Same root as dev/ (and the same SKETCH_MODELS_ROOT override).
MODELS_ROOT = Path(os.environ.get("SKETCH_MODELS_ROOT", _HERE.parent / "models"))

# Scene pairs: <id>.png (headless, inpaint base) + <id>-head.png (character
# with head, geometry source); optional crops.json (gender per scene).
# inputs/ holds the current scene set; inputs/face/ is the older sdxl_ok set.
FACE_DIR = Path(os.environ.get("FACE_INPAINT_SCENES", _HERE / "inputs"))
META_DIRNAME = "_meta"
OUTPUT_DIR = _HERE / "output"

MODEL_PATHS = {
    "sd15":       str(MODELS_ROOT / "stable-diffusion-v1-5"),
    "controlnet": str(MODELS_ROOT / "controlnet" / "control_v11p_sd15_lineart"),
    # Pose mode: CrucibleAI ControlNetMediaPipeFace (OpenRAIL) — conditions on a
    # drawn MediaPipe face mesh (pose + expression), not on identity.
    "controlnet_face": str(MODELS_ROOT / "controlnet" / "control_v2p_sd15_mediapipe_face"),

    # Style LoRA — SD1.5 build of the sktchdrw LoRA sdxl_ok's FRI uses.
    "lora":       str(MODELS_ROOT / "loras" / "sktchdrw_sd15_v1-000009.safetensors"),
    "lcm_lora":   str(MODELS_ROOT / "loras" / "lcm-lora-sdv1-5"),
    "taesd":      str(MODELS_ROOT / "taesd"),

    # IP-Adapter (h94/IP-Adapter, Apache-2.0). plus-face is the SD1.5 twin of
    # the SDXL Plus-Face sdxl_ok uses; full-face is what dev/'s FRU uses.
    "ip_adapter_dir": str(MODELS_ROOT / "ip_adapter"),
    "ip_adapter_weights": {
        "plus-face": "ip-adapter-plus-face_sd15.safetensors",
        "full-face": "ip-adapter-full-face_sd15.safetensors",
        # Our AuraFace identity adapter (AM/faceid-adapter, h94 FaceID format):
        # copy a trained step-XXXXXXX/faceid_aura_sd15.safetensors here, or
        # point --adapter-file at it.
        "aura": "faceid_aura_sd15.safetensors",
    },
    # Adapters conditioned on an AuraFace embedding, not a CLIP image: no
    # image encoder is loaded for these.
    "faceid_adapters": {"aura"},
    # OpenCLIP ViT-H/14 (LAION, MIT) — the encoder both adapters were trained on.
    "image_encoder": str(MODELS_ROOT / "image_encoder"),

    # AuraFace-v1 (Apache-2.0): face detection + head pose. FaceAnalysis reads
    # <root>/models/<name>, so this directory's parent must be named "models".
    "auraface": str(MODELS_ROOT / "AuraFace-v1"),

    "annotators_lineart": str(MODELS_ROOT / "annotators" / "lineart"),

    # MediaPipe Tasks models (Apache-2.0). as_posix(): MediaPipe mangles
    # Windows backslash paths (see dev/config.py).
    "mediapipe_face_landmarker": (MODELS_ROOT / "mediapipe" / "face_landmarker.task").as_posix(),
    "mediapipe_parser": (MODELS_ROOT / "mediapipe" / "selfie_multiclass_256x256.tflite").as_posix(),
}

# Weights fetch_models.py downloads: (repo, file in repo) -> path under MODELS_ROOT.
DOWNLOADS = {
    ("h94/IP-Adapter", "models/ip-adapter-plus-face_sd15.safetensors"):
        "ip_adapter/ip-adapter-plus-face_sd15.safetensors",
    ("h94/IP-Adapter", "models/ip-adapter-full-face_sd15.safetensors"):
        "ip_adapter/ip-adapter-full-face_sd15.safetensors",
    ("h94/IP-Adapter", "models/image_encoder/config.json"):
        "image_encoder/config.json",
    ("h94/IP-Adapter", "models/image_encoder/model.safetensors"):
        "image_encoder/model.safetensors",
    ("CrucibleAI/ControlNetMediaPipeFace", "diffusion_sd15/config.json"):
        "controlnet/control_v2p_sd15_mediapipe_face/config.json",
    ("CrucibleAI/ControlNetMediaPipeFace", "diffusion_sd15/diffusion_pytorch_model.fp16.safetensors"):
        "controlnet/control_v2p_sd15_mediapipe_face/diffusion_pytorch_model.fp16.safetensors",
}

# Scene ids per gender when crops.json has no "gender" for a scene.
GENDER_SCENES = {"male": [1, 2, 3, 5, 9], "female": [6, 7, 8]}

LORA_TRIGGER = "sktchdrw style"

# ── Generation defaults ─────────────────────────────────────────────────────
# Carried over from sdxl_ok's FRI; retune for SD1.5 once tested on the GPU.
GENERATION = {
    "ip_adapter":   "plus-face",
    "crop_size":    768,      # head-fitted crop → face ≈ 300 px; sdxl_ok ran 1024
    "steps":        25,       # DPM++ 2M
    "guidance":     5.0,
    "ip_scale":     0.70,
    "cn_scale":     0.5,      # frontal character: interior facial lineart erased
    "cn_scale_nonfrontal": 0.6,  # turned/bowed character: lineart kept as pose guide
    "lora_scale":   1.0,
    "seed":         42,
    # LCM mode (--lcm): production FRU's speed stack — LCM-LoRA + LCMScheduler,
    # 4 steps, CFG off (guidance 1.0, so the negative prompt is inactive), TAESD.
    "lcm_steps":    4,
    "lcm_guidance": 1.0,
    # Pose mode: character face mesh + character lineart (face erased); identity
    # comes only from the IP-Adapter, so it can run stronger.
    "face_cn_scale":      0.9,
    "pose_lineart_scale": 0.4,
    "pose_ip_scale":      0.85,
}

PROMPT = (
    "{trigger}, a clean light graphite pencil line-sketch portrait of a {person}, "
    "the person's own natural hairstyle, soft even light, minimal shading, "
    "fine clean outlines, monochrome, matching the light pencil style of the "
    "surrounding scene"
)
NEGATIVE = (
    "color, photo, photorealistic, 3d render, blurry, lowres, deformed face, "
    "extra faces, bald, hairless, heavy shadows, dark shading, harsh contrast, "
    "sooty, muddy, dirty face, over-hatched, black blotches, watermark, text"
)

# ── Stage-0 geometry (sdxl_ok faceswap_stage0 kiosk values) ─────────────────
CONTEXT_FRAC = 2.0            # pass-1 crop side = max(face bbox) * (1 + 2*CONTEXT_FRAC):
                              # wide, so MediaPipe fits turned/bowed drawn faces
# Face-mesh framings tried (context frac, crop px); the one closest to AuraFace's
# eye keypoints wins. Drawn turned/bowed faces fit best with wide context.
MESH_TRIALS = [(2.0, 512), (1.5, 512), (3.0, 512), (2.0, 768), (1.0, 512)]
MESH_MAX_EYE_ERROR = 0.25     # mesh eyes vs AuraFace eyes (× eye distance); worse → no pose mode
MASK_PAD_FRAC = 0.35          # rectangle fallback when no face oval is found
MASK_DILATE_FRAC = 0.008
MASK_FEATHER_FRAC = 0.012
MASK_CROWN_CENTER_FRAC = 0.18  # crown ellipse over the face oval (hair headroom)
MASK_CROWN_W_FRAC = 0.85
MASK_CROWN_H_FRAC = 1.05
FACE_INTERIOR_W_FRAC = 0.46    # ellipse of character eyes/nose/mouth lineart to erase
FACE_INTERIOR_H_FRAC = 0.60
FACE_INTERIOR_CY_FRAC = 0.54
FRONTAL_MAX_YAW = 30.0
FRONTAL_MAX_PITCH = 14.0
FRONTAL_MAX_ROLL = 14.0
PASTE_FEATHER_PX = 6

# ── Zoom: the generation crop is fitted to the head, not to 3× the face box ──
# Pass 1 finds the head region on a coarse crop; pass 2 crops the head region
# plus this margin per side and generates it at crop_size, so the face lands
# at roughly crop_size / 2.5 px instead of crop_size / 3.
HEAD_MARGIN_FRAC = 0.12
MIN_CROP_SIDE = 64

# ── Modes ────────────────────────────────────────────────────────────────────
#   pose      : the character's face mesh (pose, expression) + the character's
#               lineart with the face erased; the visitor supplies identity only
#               (IP-Adapter). The visitor is drawn *as* the character.
#   photo     : the visitor's own photo lineart (face + hair from the photo whose
#               yaw best matches), aligned on 5 keypoints. Keeps the photo's pose.
#   character : sdxl_ok behaviour — character hair/jaw lines, features erased.
MODE = "pose"
# Pose mode hair: "character" keeps the character's hair/headwear lines;
# "visitor" erases every character line inside the head so the IP-Adapter
# identity (hair-inclusive reference crops) drives the hair.
HAIR = "character"
FACE_ERASE_DILATE_FRAC = 0.02   # lineart erased around the face oval (× crop size)
VIEW_FACE_PX = 360            # visitor face height in the analysed head crop
VIEW_CROP_FRAC = 3.0          # head crop side = this × face size (room for long hair)
VISITOR_MASK_DILATE_FRAC = 0.03   # inpaint room around the visitor's head (× crop size)

# Visitor reference crop: padded upward so the hair is in the IP-Adapter image.
ID_CROP_TOP_FRAC = 1.10
ID_CROP_SIDE_FRAC = 0.50
ID_CROP_BOT_FRAC = 0.45

# Scene metadata: a head-scene face overlapping a headless detection by more
# than this IoU is a bystander, not the removed character.
BBOX_IOU_MATCH = 0.30

# ── Post-processing: force the head to scene-matched graphite ───────────────
PENCIL_PARAMS = {
    "enabled":               True,
    "bilateral_d":           5,
    "bilateral_sigma_color": 35,
    "bilateral_sigma_space": 5,
    "clahe_clip":            1.7,
    "clahe_grid":            8,
    "unsharp_radius":        1.5,
    "unsharp_percent":       115,
    "unsharp_threshold":     3,
    "black_point":           0.02,
    "white_point":           1.0,
    "contrast":              1.12,
    "tone_match":            True,
    "tone_match_strength":   0.7,
}
