#!/usr/bin/env bash
# Input-resolution test (2026-10-23): p1's 3 photos at 5 sizes (long side 4032 / 2048 /
# 1024 / 512 / 384) through the production candidates (4:5, line_noface ± kps, ff / ff+ours),
# at production settings and at guidance 1.5; then every drawing rescored against the
# full-res photos (rescore_ref.py). Lab only. output/res_<step>/, .queue_res/<step>.done.
#
#   cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_res.sh > queue_res.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
mkdir -p .queue_res output

T=~/WPU/testings
R=res_test/mul_face
VIS="$R/p1_r4k $R/p1_r2k $R/p1_r1k $R/p1_r512 $R/p1_r384"
GEN="python gen_characters.py --gender p1_r4k=male p1_r2k=male p1_r1k=male p1_r512=male p1_r384=male \
  --visitors $VIS --assets $T/assets_512x640 --variants line_noface line_noface+kps \
  --aura-file $T/faceid-adapter/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
  --faceid-repo $T/faceid-adapter"

step() {
  local name=$1; shift
  if [ -f ".queue_res/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" > "output/res_$name.log" 2>&1; then
    touch ".queue_res/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (see output/res_$name.log)"
  fi
}

step prod     $GEN --out output/res_prod
step g15      $GEN --out output/res_g15 --guidance 1.5
step rescore  python rescore_ref.py --ref $R/p1_r4k --visitors $VIS --runs output/res_prod output/res_g15 \
  --faceid-repo $T/faceid-adapter
echo "$(date '+%F %T')  queue finished"
