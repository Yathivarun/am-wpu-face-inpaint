"""
Stage 1 (GPU): inpaint the visitor's face into the character head, then paste
it back into the headless scene.

SD1.5 port of sdxl_ok faceswap_inpaint:

    SDXL union ControlNet inpaint (lineart mode)  ->  SD1.5 ControlNet inpaint
                                                      + control_v11p_sd15_lineart
    IP-Adapter Plus-Face SDXL                     ->  ip-adapter-plus-face_sd15
                                                      (or full-face_sd15)
    sktchdrw SDXL LoRA                            ->  sktchdrw_sd15 LoRA

Identity is the IP-Adapter embedding averaged over all visitor photos.
Post-processing (graphite treatment, tone match, drawn-head paste mask,
feathered paste-back) is plain OpenCV, unchanged from sdxl_ok.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

import config as C
from faces import face_oval_mask


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════
def build_pipeline(ip_adapter=C.GENERATION["ip_adapter"], lcm=False, lora_path=None,
                   lora_scale=C.GENERATION["lora_scale"], offload=False, face_cn=False,
                   adapter_file=None):
    """Build once, reuse for every scene. Returns the diffusers pipeline.

    face_cn=True (pose mode) adds the MediaPipe-face ControlNet in front of the
    lineart one; control images and scales are then passed as [mesh, lineart].
    adapter_file overrides the IP-Adapter weight file (e.g. a training checkpoint)."""
    import torch
    from diffusers import (ControlNetModel, DPMSolverMultistepScheduler, LCMScheduler,
                           StableDiffusionControlNetInpaintPipeline)
    from transformers import CLIPVisionModelWithProjection

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    sd15 = Path(C.MODEL_PATHS["sd15"])
    variant = "fp16" if (sd15 / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists() else None

    controlnet = ControlNetModel.from_pretrained(
        C.MODEL_PATHS["controlnet"], torch_dtype=dtype, local_files_only=True)
    if face_cn:
        face = ControlNetModel.from_pretrained(
            C.MODEL_PATHS["controlnet_face"], torch_dtype=dtype, variant="fp16",
            use_safetensors=True, local_files_only=True)
        controlnet = [face, controlnet]            # diffusers wraps this in MultiControlNetModel
    image_encoder = None if ip_adapter in C.MODEL_PATHS["faceid_adapters"] else (
        CLIPVisionModelWithProjection.from_pretrained(
            C.MODEL_PATHS["image_encoder"], torch_dtype=dtype, local_files_only=True))
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        str(sd15), controlnet=controlnet, image_encoder=image_encoder,
        torch_dtype=dtype, variant=variant, use_safetensors=True,
        safety_checker=None, requires_safety_checker=False, local_files_only=True)

    # LoRAs: style always; LCM-LoRA on top in --lcm mode. Fused for speed.
    lora = Path(lora_path or C.MODEL_PATHS["lora"])
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="style")
    names, weights = ["style"], [lora_scale]
    if lcm:
        pipe.load_lora_weights(C.MODEL_PATHS["lcm_lora"],
                               weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="lcm")
        names.append("lcm")
        weights.append(1.0)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lcm:
        # Same speed stack as production FRU: TAESD in place of the full VAE.
        from diffusers import AutoencoderTiny
        pipe.vae = AutoencoderTiny.from_pretrained(C.MODEL_PATHS["taesd"], torch_dtype=dtype,
                                                   local_files_only=True)
    pipe.set_adapters(names, adapter_weights=weights)
    pipe.fuse_lora(adapter_names=names)
    pipe.unload_lora_weights()          # drop the (now fused) adapter layers

    weight = Path(adapter_file) if adapter_file else (
        Path(C.MODEL_PATHS["ip_adapter_dir"]) / C.MODEL_PATHS["ip_adapter_weights"][ip_adapter])
    if not weight.is_file():
        raise FileNotFoundError(f"IP-Adapter weights not found: {weight}")
    pipe.load_ip_adapter(str(weight.parent), subfolder="", weight_name=weight.name,
                         image_encoder_folder=None, local_files_only=True)

    if offload and cuda:
        pipe.enable_model_cpu_offload()   # for small (<8 GB) GPUs
    else:
        pipe.to("cuda" if cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def averaged_ip_embeds(pipe, face_refs, do_cfg):
    """IP-Adapter embeddings averaged over every visitor reference crop."""
    import torch
    per = [pipe.prepare_ip_adapter_image_embeds(
               ip_adapter_image=[img], ip_adapter_image_embeds=None,
               device=pipe._execution_device, num_images_per_prompt=1,
               do_classifier_free_guidance=do_cfg)[0]
           for img in face_refs]
    return [torch.stack(per, dim=0).mean(dim=0)]


def aura_ip_embeds(pipe, photos, face_refs, do_cfg):
    """FaceID-adapter embeddings: the AuraFace identity vector averaged over the
    visitor photos (re-normalised), shaped as the adapter was trained on it.
    A photo without a detectable face falls back to its reference crop."""
    import torch
    from faces import face_embedding
    vecs = [e for e in (face_embedding(p) for p in photos) if e is not None]
    if not vecs:
        vecs = [e for e in (face_embedding(r) for r in face_refs) if e is not None]
    if not vecs:
        raise RuntimeError("AuraFace found no face in the visitor photos")
    v = np.mean(np.stack(vecs), axis=0)
    v = v / (np.linalg.norm(v) + 1e-8)
    pos = torch.from_numpy(v.astype(np.float32)).to(pipe._execution_device, pipe.unet.dtype)[None, None]
    return [torch.cat([torch.zeros_like(pos), pos]) if do_cfg else pos]   # [negative, positive]


def run_inpaint(pipe, inp, *, prompt, negative, ip_scale, cn_scale, steps, guidance,
                seed, pencil=True, ip_embeds=None):
    """One scene: inpaint the crop, post-process, paste back.

    `inp` is stage0.build_inputs(...). `cn_scale` is a float, or a list matching
    inp["control_images"]. Pass `ip_embeds` to reuse the averaged identity
    across scenes. Returns (composed_bgr, raw_crop, final_crop, paste_mask)."""
    import torch
    pipe.set_ip_adapter_scale(ip_scale)
    if ip_embeds is None:
        ip_embeds = averaged_ip_embeds(pipe, inp["face_refs"], do_cfg=guidance > 1.0)
    size = inp["crop_size"]
    raw = pipe(
        prompt=prompt, negative_prompt=negative,
        image=inp["init_img"], mask_image=inp["mask_img"],
        control_image=inp["control_images"] if len(inp["control_images"]) > 1 else inp["control_images"][0],
        controlnet_conditioning_scale=cn_scale, ip_adapter_image_embeds=ip_embeds,
        strength=1.0, num_inference_steps=steps, guidance_scale=guidance,
        height=size, width=size, generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    result = raw
    if pencil and C.PENCIL_PARAMS["enabled"]:
        result = pencilify_crop(result, inp["mask_img"], C.PENCIL_PARAMS)
        if C.PENCIL_PARAMS["tone_match"]:
            result = tone_match_to_scene(result, inp["mask_img"], inp["scene_bgr"],
                                         inp["crop_box"], C.PENCIL_PARAMS["tone_match_strength"])
    paste_mask = head_mask_from_generated(result, inp["mask_img"])
    composed = paste_back(inp["scene_bgr"], result, inp["crop_box"], inp["feather_px"], paste_mask)
    return composed, raw, result, paste_mask


def compose_on(scene_bgr, inp, result, paste_mask):
    """The same finished crop pasted onto another full-size scene (e.g. the
    head scene, which is not pixel-identical to the headless one)."""
    return paste_back(scene_bgr, result, inp["crop_box"], inp["feather_px"], paste_mask)


# ════════════════════════════════════════════════════════════════════════════
# Post-processing (unchanged from sdxl_ok faceswap_inpaint)
# ════════════════════════════════════════════════════════════════════════════
def pencilify_crop(result_rgb, mask_pil, p):
    """Monochrome graphite inside the head mask (bilateral -> CLAHE -> unsharp ->
    levels/contrast); plain desaturate outside, blended over a feathered mask."""
    L = np.asarray(result_rgb.convert("L"), dtype=np.uint8)
    work = L
    if p.get("bilateral_d", 0) > 0:
        work = cv2.bilateralFilter(L, p["bilateral_d"], p["bilateral_sigma_color"],
                                   p["bilateral_sigma_space"])
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"],
                            tileGridSize=(p["clahe_grid"], p["clahe_grid"]))
    out = Image.fromarray(clahe.apply(work)).convert("RGB").filter(ImageFilter.UnsharpMask(
        radius=p["unsharp_radius"], percent=p["unsharp_percent"], threshold=p["unsharp_threshold"]))
    a = np.asarray(out.convert("L"), dtype=np.float32) / 255.0
    span = max(p["white_point"] - p["black_point"], 1e-3)
    a = np.clip((a - p["black_point"]) / span, 0.0, 1.0)
    a = np.clip((a - 0.5) * p["contrast"] + 0.5, 0.0, 1.0)
    pencil = np.repeat((a * 255.0).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    gray = np.repeat(L[..., None], 3, axis=2).astype(np.float32)
    m = np.asarray(mask_pil.convert("L").resize(result_rgb.size), dtype=np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), 6 * result_rgb.size[0] / 1024)[..., None]  # 6 px at 1024
    return Image.fromarray((m * pencil + (1.0 - m) * gray).astype(np.uint8))


def tone_match_to_scene(result_rgb, mask_pil, scene_bgr, crop_box, strength=0.7):
    """Shift the head's median luminance toward the scene in a ring around it."""
    if strength <= 0:
        return result_rgb
    w, h = result_rgb.size
    x1, y1, x2, y2 = crop_box
    scene_gray = cv2.cvtColor(cv2.resize(scene_bgr[y1:y2, x1:x2], (w, h)),
                              cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = np.asarray(mask_pil.convert("L").resize((w, h)), np.uint8)
    inside = m > 127
    if inside.sum() < 50:
        return result_rgb
    ring = max(3, int(61 * w / 1024)) | 1            # sdxl_ok's 61 px ring at 1024
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring, ring)))
    ring = (dil > 127) & ~inside
    if ring.sum() < 50:
        ring = ~inside
    target = float(np.median(scene_gray[ring]))
    res = np.asarray(result_rgb.convert("RGB"), np.float32)
    cur = float(np.median(cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_RGB2GRAY)[inside]))
    soft = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 8 * w / 1024) / 255.0)[..., None]
    return Image.fromarray(np.clip(res + (target - cur) * strength * soft, 0, 255).astype(np.uint8))


def head_mask_from_generated(result_rgb, inpaint_mask_pil):
    """Paste mask = the head the model actually drew (strokes darker than the
    local paper tone + the drawn face oval), largest blob, filled. Avoids a white
    halo and does not clip hair that grew past the inpaint mask."""
    w, h = result_rgb.size
    gray = cv2.cvtColor(np.asarray(result_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    inp = np.asarray(inpaint_mask_pil.convert("L").resize((w, h)), np.uint8) > 127
    if inp.sum() < 50:
        return Image.fromarray(inp.astype(np.uint8) * 255)
    bg = float(np.percentile(gray[inp], 85))
    ink = inp & (gray < bg - 18)
    oval = face_oval_mask(result_rgb)
    if oval is not None:
        ink |= (cv2.resize(oval, (w, h)) > 0) & inp
    drawn = ink.astype(np.uint8) * 255
    k = max(3, int(0.02 * max(h, w)))
    drawn = cv2.morphologyEx(drawn, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((drawn > 0).astype(np.uint8), 8)
    if n > 1:
        drawn = np.where(lbl == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(drawn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(drawn)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return Image.fromarray(filled)


def paste_back(scene_bgr, result_rgb, crop_box, feather_px, paste_mask=None):
    """Composite the crop back into the scene; only `paste_mask` (crop space) changes."""
    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    res = cv2.resize(np.array(result_rgb)[:, :, ::-1], (side, side), interpolation=cv2.INTER_AREA
                     if result_rgb.size[0] > side else cv2.INTER_LANCZOS4).astype(np.float32)
    out = scene_bgr.copy()
    region = out[y1:y2, x1:x2].astype(np.float32)
    if paste_mask is None:
        out[y1:y2, x1:x2] = res.astype(np.uint8)
        return out
    m = cv2.resize(np.asarray(paste_mask.convert("L")), (side, side), interpolation=cv2.INTER_LINEAR)
    fp = max(3, feather_px or 0, int(0.008 * side))
    m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), fp) / 255.0)[..., None]
    out[y1:y2, x1:x2] = (m * res + (1.0 - m) * region).astype(np.uint8)
    return out
