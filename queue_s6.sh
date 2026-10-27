#!/usr/bin/env bash
# Run s6 (2026-10-26): the s5 winner (4:5, line_noface+kps, eyes+lips map, ours 1.0) only,
# on 14 visitors (p13 new) and 11 characters per gender (M5 = stock f1f rider, F5 = stock f8f).
#   base512   the recipe at 512, ff and ff+ours (baseline, adapter gap)
#   front512  ours fed the front photo's vector instead of the mean of 3
#   seed7 / seed42   the recipe at two more seeds: run-to-run variation
#   r768      the recipe at 768 x 960, ff and ff+ours
#   best768   768 + guidance 1.5: every gain that held up (4:5, eyes+lips, ours 1.0, 768, g1.5)
# Then eval_ident.py: wrong-person baseline, thresholds, rank-1, face shape. Lab only.
# Assets are rebuilt from $T/chars_src into new folders; nothing existing is touched.
#
#   cd ~/WPU/testings/fullbody-lab/dev/face_inpaint && nohup bash queue_s6.sh > queue_s6.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
mkdir -p .queue_s6 output

T=~/WPU/testings
VIS="" GENDERS=""
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
  [ -d "mul_face/p$i" ] || { echo "missing mul_face/p$i"; continue; }
  VIS="$VIS mul_face/p$i"
  if [ "$i" -le 5 ] || [ "$i" -ge 12 ]; then GENDERS="$GENDERS p$i=male"; else GENDERS="$GENDERS p$i=female"; fi
done
echo "$(date '+%F %T')  visitors:$GENDERS"

A512=$T/assets_s6_512x640
A768=$T/assets_s6_768x960
build() {     # out size
  if [ -f "$1/index.json" ]; then echo "$(date '+%F %T')  skip   assets $1"; return 0; fi
  echo "$(date '+%F %T')  build  assets $1"
  python build_assets.py --src $T/chars_src --out "$1" --size "$2" --aspect 0.8 --face-at M5=1130,659 \
    > "$1.log" 2>&1 || { echo "FAILED assets $1 (see $1.log)"; return 1; }
  grep -E "^(M|F)" "$1.log"
}
build $A512 512 || exit 1
build $A768 768 || exit 1

GEN="python gen_characters.py --gender $GENDERS --visitors $VIS --variants line_noface+kps \
  --kps-map kps_eyelip.png --aura-scale 1.0 \
  --aura-file $T/faceid-adapter/checkpoints/real_v3_cd150/step-0155000/faceid_aura_sd15.safetensors \
  --faceid-repo $T/faceid-adapter"
OURS="--ids ff+ours"
BOTH="--ids ff ff+ours"

step() {
  local name=$1; shift
  if [ -f ".queue_s6/$name.done" ]; then echo "$(date '+%F %T')  skip   $name (done)"; return; fi
  echo "$(date '+%F %T')  start  $name"
  if "$@" --out "output/s6_$name" > "output/s6_$name.log" 2>&1; then
    touch ".queue_s6/$name.done"; echo "$(date '+%F %T')  done   $name"
  else
    echo "$(date '+%F %T')  FAILED $name (see output/s6_$name.log)"
  fi
}

step base512  $GEN --assets $A512 $BOTH
step front512 $GEN --assets $A512 $OURS --id-vector front
step seed7    $GEN --assets $A512 $OURS --seed 7
step seed42   $GEN --assets $A512 $OURS --seed 42
step r768     $GEN --assets $A768 $BOTH
step best768  $GEN --assets $A768 $BOTH --guidance 1.5

EXTRA=""
[ -f $T/p12-gpt.png ] && EXTRA="--extra $T/p12-gpt.png=p12"
echo "$(date '+%F %T')  start  eval"
python eval_ident.py --runs output/s6_base512 output/s6_front512 output/s6_seed7 output/s6_seed42 \
  output/s6_r768 output/s6_best768 --gender $GENDERS $EXTRA --out output/s6_eval.json \
  > output/s6_eval.log 2>&1 && echo "$(date '+%F %T')  done   eval" || echo "FAILED eval (see output/s6_eval.log)"
echo "$(date '+%F %T')  queue finished"
