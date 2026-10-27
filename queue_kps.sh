#!/usr/bin/env bash
# Run s5 (2026-10-23): (A/B) which keypoint map keeps the character's pose without
# imposing its face shape, on the square and 4:5 crops; (C) does identity improve with
# more capacity (768 / 1024 px, 6 / 8 LCM steps, all + guidance 1.5)?
# Only the production candidate is generated: line_noface+kps, ff+ours with ours at 1.0
# (C also runs ff alone, the baseline the paper needs). Lab only.
# output/s5_<step>/, .queue_s5/<step>.done, a rerun skips finished steps.
#
#   cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_kps.sh > queue_s5.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
mkdir -p .queue_s5 output

T=~/WPU/testings
VIS="" GENDERS=""
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
  [ -d "mul_face/p$i" ] || continue
  VIS="$VIS mul_face/p$i"
  if [ "$i" -le 5 ] || [ "$i" -eq 12 ]; then GENDERS="$GENDERS p$i=male"; else GENDERS="$GENDERS p$i=female"; fi
done
echo "$(date '+%F %T')  visitors:$GENDERS"

GEN="python gen_characters.py --gender $GENDERS --visitors $VIS --variants line_noface+kps --aura-scale 1.0 \
  --aura-file $T/faceid-adapter/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
  --faceid-repo $T/faceid-adapter"
SQ="--assets $T/assets_512"
R45="--assets $T/assets_512x640"
OURS="--ids ff+ours"
BOTH="--ids ff ff+ours"

step() {
  local name=$1; shift
  if [ -f ".queue_s5/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" --out "output/s5_$name" > "output/s5_$name.log" 2>&1; then
    touch ".queue_s5/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (see output/s5_$name.log)"
  fi
}

# A. keypoint maps, square crop
step sq_full    $GEN $SQ $OURS --kps-map mesh.png                       # baseline (s4: full mesh 0.4)
step sq_full025 $GEN $SQ $OURS --kps-map mesh.png --kps-scale 0.25
step sq_feat    $GEN $SQ $OURS --kps-map kps_feat.png                   # no face outline
step sq_eyelip  $GEN $SQ $OURS --kps-map kps_eyelip.png
step sq_eyes    $GEN $SQ $OURS --kps-map kps_eyes.png

# B. the same maps on 4:5 (is the slim face at angles the outline's doing?)
step r45_full   $GEN $R45 $OURS --kps-map mesh.png
step r45_feat   $GEN $R45 $OURS --kps-map kps_feat.png
step r45_eyelip $GEN $R45 $OURS --kps-map kps_eyelip.png
step r45_eyes   $GEN $R45 $OURS --kps-map kps_eyes.png

# C. capacity, square crop, full mesh: resolution, steps, and everything at once
step cap_512    $GEN $SQ $BOTH                                          # the 512 / 4-step reference
step cap_768    $GEN --assets $T/assets_768sq $BOTH
step cap_1024   $GEN --assets $T/assets_1024sq $BOTH
step cap_s6     $GEN $SQ $BOTH --steps 6
step cap_s8     $GEN $SQ $BOTH --steps 8
step cap_max    $GEN --assets $T/assets_1024sq $BOTH --steps 8 --guidance 1.5
echo "$(date '+%F %T')  queue finished"
