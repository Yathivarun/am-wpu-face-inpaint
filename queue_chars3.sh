#!/usr/bin/env bash
# Run 3 (2026-10-22): geometry (square vs 4:5, contour, +kps), then identity knobs on 4:5.
# Lab only. Same mechanics as faceid-adapter/queue_*.sh: output/chars3_<step>/, .queue3/<step>.done,
# a rerun skips finished steps.
#
#   cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_chars3.sh > queue3.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
mkdir -p .queue3 output

T=~/WPU/testings
GEN="python gen_characters.py --gender p5=female \
  --visitors mul_face/p1 mul_face/p2 mul_face/p3 mul_face/p4 mul_face/p5 mul_face/p6 mul_face/p7 \
  --aura-file $T/faceid-adapter/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
  --faceid-repo $T/faceid-adapter"
SQ="--assets $T/assets_512"
R45="--assets $T/assets_512x640"
PAIR="--variants line_noface line_contour+kps"

step() {
  local name=$1; shift
  if [ -f ".queue3/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" --out "output/chars3_$name" > "output/chars3_$name.log" 2>&1; then
    touch ".queue3/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (see output/chars3_$name.log)"
  fi
}

# A. geometry, production identity settings (ff 0.5, ours 0.8, guidance 1.0)
step sq   $GEN $SQ             # line_noface, line_contour, line_noface+kps, line_contour+kps
step r45  $GEN $R45

# B. identity knobs on 4:5, baseline line_noface vs line_contour+kps
step r45_g15      $GEN $R45 $PAIR --guidance 1.5
step r45_mask10   $GEN $R45 $PAIR --id-mask face --aura-scale 1.0
step r45_mask12   $GEN $R45 $PAIR --id-mask face --aura-scale 1.2
step r45_a10      $GEN $R45 $PAIR --aura-scale 1.0
step r45_view     $GEN $R45 $PAIR --id-vector view
echo "$(date '+%F %T')  queue finished"
