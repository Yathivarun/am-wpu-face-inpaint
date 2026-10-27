# am-wpu-face-inpaint: FRU face lab (SD1.5)

This lab draws a visitor's face into a premade sketch character, in the character's pose and the scene's pencil style. It is where the FRU identity work is generated and tested:
- the AuraFace identity adapter from [am-faceid-adapter](https://github.com/Yathivarun/am-faceid-adapter);
- the h94 full-face IP-Adapter;
- the ControlNet maps and crop recipes.

Results, charts and the research log are in [wpu-face-inpaint-research](https://github.com/Yathivarun/wpu-face-inpaint-research).

**Lab only.** Nothing here ships. Production FRU is `triton-server` / `sketch-service` in `am-ml-fullbody-sketch`, and this lab runs that production generator in-process.

## Where it lives

The lab is a folder **inside** `am-ml-fullbody-sketch`, at `dev/face_inpaint/`. It imports the parent repo's code: the production generator, its config and the pre/post-processing. It needs:

| What | Where | Why |
|---|---|---|
| `am-ml-fullbody-sketch`, branch **`aura-adapter`** (commit `8a89808`, on top of `v2`) | parent of this folder | The FRU generator with the diffusers-native IP-Adapter backend and the optional AuraFace adapter (`pipeline/ip_native.py`, `SKETCH_ID_ADAPTER=aura`) |
| [am-faceid-adapter](https://github.com/Yathivarun/am-faceid-adapter) | sibling folder, or `--faceid-repo` / `FACEID_REPO` | AuraFace helpers (`faceid/aura.py`) and the adapter checkpoint |
| Models | `dev/models` or `SKETCH_MODELS_ROOT` | See [Models](#models) |

The parent repo ignores the folder through its `.git/info/exclude`. This repo tracks it on its own.

```bash
git clone <am-ml-fullbody-sketch> fullbody-lab && cd fullbody-lab && git checkout aura-adapter
git clone https://github.com/Yathivarun/am-wpu-face-inpaint.git dev/face_inpaint
echo "/dev/face_inpaint/" >> .git/info/exclude
git clone https://github.com/Yathivarun/am-faceid-adapter.git ../faceid-adapter
```

On the GPU box the layout is `~/WPU/testings/fullbody-lab/dev/face_inpaint` and `~/WPU/testings/faceid-adapter`. The `queue_*.sh` scripts assume this layout.

## Current recipe (after run s6)

| Part | Setting |
|---|---|
| Generator | Production FRU: SD1.5 + LCM-LoRA, 4 steps, CFG 1.0, sktchdrw style LoRA, TAESD |
| Crop | 4:5 head crop (crown to neck), 512 × 640; 768 × 960 for the best quality |
| ControlNets | Lineart **with the face erased** (`lineart_noface.png`) at 0.6, plus MediaPipe-face at 0.4 on **eyes + lips only** (`kps_eyelip.png`) |
| Identity | h94 full-face IP-Adapter at 0.5 on the nearest-view crop, plus our AuraFace adapter at 1.0 on the mean vector of the visitor's photos |
| Adapter | `real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors` (research-only weights) |
| Best setting | 768 × 960 + guidance 1.5 (about 2.8× the time of 512) |
| Paste-back | Through the character's head mask into the scene |

Results: FaceNet identity at 512 is **0.578** for ff+ours vs **0.418** for ff alone. The best setting gives 0.616 vs 0.426. Full numbers are in the research repo.

## Pipeline

| Step | Script | Runs on | What |
|---|---|---|---|
| Character assets (once per character) | `build_assets.py` | CPU | From a drawn character it makes the crop, `lineart_full` / `lineart_noface` / `lineart_contour`, `mesh.png` and the sparse `kps_*.png` maps, depth, `face_mask`, `head_mask` and `meta.json`. The depth model is used for the head mask. |
| Generate + score | `gen_characters.py` | GPU | For each visitor × character × variant × identity (`ff`, `ff+ours`): the production generator, then the composite, sheets, `scores.csv` (FaceNet + AuraFace + pose error) and `summary.json` |
| Identity evaluation | `eval_ident.py` | GPU / CPU | Wrong-person scores, rank-1, TAR at 1% / 5% FAR, EER, photo-to-photo reference, 68-landmark face-shape ratio |
| Resolution test | `rescore_ref.py` | CPU | Rescores runs against one full-resolution reference |
| Production path A/B | `triton_local.py` | GPU | The production direct-sketch FRU/SAU path with a choice of adapter (full-face, plus-face, aura, full+aura); a parity check of the native port |

### Run it

```bash
# character assets: 4:5, 512 x 640 (--face-at picks one face from a crowd scene)
python build_assets.py --src <chars_src> --out <assets_512x640> --size 512 --aspect 0.8 --face-at M5=1130,659

# current recipe on a set of visitors
python gen_characters.py --assets <assets_512x640> --visitors mul_face/p1 mul_face/p2 \
    --gender p1=male p2=female --variants line_noface+kps --kps-map kps_eyelip.png \
    --aura-scale 1.0 --ids ff ff+ours \
    --aura-file <faceid-adapter>/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
    --faceid-repo <faceid-adapter> --out output/my_run
#   best quality: --assets <assets_768x960> --guidance 1.5

python eval_ident.py --runs output/my_run --gender p1=male p2=female --out output/my_run_eval.json
```

Other `gen_characters.py` knobs:
- `--steps`, `--guidance`, `--seed`;
- `--id-vector mean|view|front`, `--id-mask face`;
- `--ff-scale`, `--aura-scale`, `--kps-scale`, `--cn-scale VARIANT=S`;
- `--variants` (`line_noface`, `line_contour`, `+kps`, `depth_smooth`, `line_full`, `mesh`, `none`).

The docstring at the top of each script has the full details.

### Visitor input

There is one folder per visitor with their registration photos. A name containing `FV`, `LV` or `RV` marks the front, left or right view; otherwise the head yaw is measured. Visitors are `mul_face/p0` … `p15`, and they are **not in git**, because they are real people's photos. Keep them on the laptop and the GPU box only.

### Run queues

`queue_*.sh` are the unattended runs on the GPU box, kept as the exact record of each experiment. Each step writes `output/<run>_<step>/` and marks itself done in `.queue_<run>/`; a rerun skips finished steps.

| Script | Run | What it tested |
|---|---|---|
| `queue_chars3.sh` | s3 | Square vs 4:5, contour, +kps, identity knobs (7 visitors) |
| `queue_s4.sh` | s4 | The run-3 steps on 12 visitors, plus the production sheet |
| `queue_res.sh` | res | Input photo resolution (4032 → 384 px) |
| `queue_kps.sh` | s5 | Sparse kps maps (square / 4:5) and the capacity ceiling (768, 1024, steps, guidance) |
| `queue_s6.sh` | s6 | The s5 winner only: 512 baseline, front vector, seeds, 768, 768 + g1.5, then `eval_ident.py` |

```bash
cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_s6.sh > queue_s6.log 2>&1 & disown
```

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python fetch_models.py          # downloads the IP-Adapter files and checks the rest
```

On the offline GPU box, use the master venv from `AM/linux-wheelhouse` (`bash install.sh master`).

On the Windows laptop, Smart App Control (since 2026-10-26) blocks the `onnx` DLL that insightface needs, so AuraFace scoring can't run locally. Run scoring on the box, or use the `onnx_stub.py` shim from the research repo.

## Models

All the weights listed here allow commercial use. Our own adapter's *weights* are research-only until it is retrained, because its training data was FFHQ + CelebA-HQ.

| Model | Source | Licence |
|---|---|---|
| SD 1.5 | runwayml / stable-diffusion-v1-5 | CreativeML OpenRAIL-M |
| LCM-LoRA SD1.5 | latent-consistency | OpenRAIL++ |
| TAESD | madebyollin | MIT |
| sktchdrw style LoRA | (licence to confirm) | — |
| ControlNet lineart (`control_v11p_sd15_lineart`) | lllyasviel | OpenRAIL |
| ControlNet MediaPipe-face (`control_v2p_sd15_mediapipe_face`) | CrucibleAI | OpenRAIL |
| IP-Adapter full-face / plus-face SD1.5 + image encoder (OpenCLIP ViT-H/14) | h94/IP-Adapter | Apache-2.0 / MIT |
| AuraFace-v1 | fal | Apache-2.0 |
| MediaPipe face mesh | Google | Apache-2.0 |
| DPT-Hybrid MiDaS (head mask) | Intel | MIT |
| FaceNet (VGGFace2) | facenet-pytorch | **Scoring only**, never in the generator |

Do not use the h94 IP-Adapter **FaceID** weights or InsightFace's antelopev2 / buffalo models in the generator: they are non-commercial.

## Legacy: scene inpaint mode (`run.py`)

This is the first version of the lab: the SD1.5 port of sdxl_ok's `fri` place. It inpaints the face directly inside a full scene rather than a premade character crop. The files are `run.py`, `scene_meta.py`, `faces.py`, `stage0.py`, `inpaint.py` and `config.py`, with scenes in `inputs/` (`<id>.png` headless plus `<id>-head.png`).

| `--mode` | ControlNets | Result |
|---|---|---|
| `pose` (default) | MediaPipe-face on the character's mesh + character lineart with the face erased | The visitor's identity in the character's pose |
| `photo` | The visitor's own face and hair lines from the closest-yaw photo | The visitor's photo pose and hairstyle |
| `character` | Character hair and jaw lines | The visitor's face with the character's hair |

```bash
python scene_meta.py --scene 10                       # once per scene (--force --manual-xy X,Y to pick the character)
python run.py --faces mul_face/p1 --scenes 1 9 --mode pose --hair visitor --score
python run.py --faces a.jpg --gender female --lcm     # FRU speed: 4 steps, no CFG, TAESD
```

Output goes to `output/<timestamp>/`: `scene_<id>.png`, `debug/` (with `--debug`) and `run.json` (settings, timings, AuraFace `identity`). The tuning defaults are in `config.py`.

Known limits:
- Pose mode depends on MediaPipe fitting the drawn face.
- Scene heads are small (100–260 px), so crops are upscaled 2–3×.
- Scenes with several headless figures need `--manual-xy`.
