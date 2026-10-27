#!/usr/bin/env bash
# Run s4 (2026-10-22): run 3's seven steps on the updated visitors (mul_face: 0-5 male,
# 6-11 female). Production candidates line_noface ± kps run in every step, with their own
# sheet (sheet_<visitor>_prod_*.jpg); the contour variants stay for research only.
# Lab only. output/s4_<step>/, .queue_s4/<step>.done, a rerun skips finished steps.
#
#   cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_s4.sh > queue_s4.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
mkdir -p .queue_s4 output

# visitors: every mul_face/<name> folder with a number in its name; gender from the number
VIS="" GENDERS=""
for d in mul_face/*/; do
  n=$(basename "$d"); i=${n//[!0-9]/}
  [ -z "$i" ] && continue
  VIS="$VIS mul_face/$n"
  if [ "$((10#$i))" -le 5 ]; then GENDERS="$GENDERS $n=male"; else GENDERS="$GENDERS $n=female"; fi
done
echo "$(date '+%F %T')  visitors:$GENDERS"
[ -z "$VIS" ] && { echo "no visitor folders in mul_face/"; exit 1; }

T=~/WPU/testings
GEN="python gen_characters.py --gender $GENDERS --visitors $VIS \
  --aura-file $T/faceid-adapter/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
  --faceid-repo $T/faceid-adapter"
SQ="--assets $T/assets_512"
R45="--assets $T/assets_512x640"
TRIO="--variants line_noface line_noface+kps line_contour+kps"

step() {
  local name=$1; shift
  if [ -f ".queue_s4/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" --out "output/s4_$name" > "output/s4_$name.log" 2>&1; then
    touch ".queue_s4/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (see output/s4_$name.log)"
  fi
}

# A. geometry, production identity settings (ff 0.5, ours 0.8, guidance 1.0)
step sq   $GEN $SQ             # line_noface, line_contour, line_noface+kps, line_contour+kps
step r45  $GEN $R45

# B. identity knobs on 4:5: the production pair line_noface ± kps, plus line_contour+kps (research)
step r45_g15      $GEN $R45 $TRIO --guidance 1.5
step r45_mask10   $GEN $R45 $TRIO --id-mask face --aura-scale 1.0
step r45_mask12   $GEN $R45 $TRIO --id-mask face --aura-scale 1.2
step r45_a10      $GEN $R45 $TRIO --aura-scale 1.0
step r45_view     $GEN $R45 $TRIO --id-vector view
echo "$(date '+%F %T')  queue finished"
