"""
Input-resolution test: score every drawing against ONE reference identity.

gen_characters.py scores each drawing against its own visitor's photos, so a 384 px
visitor is judged against a 384 px reference. Here every run folder is rescored against
the reference folder (the full-res photos), and the inputs themselves are compared:

    python rescore_ref.py --ref res_test/mul_face/p1_r4k \
        --visitors res_test/mul_face/p1_r* --runs output/res_prod output/res_g15

Writes <run>/scores_ref.csv and prints, per run: input similarity of each visitor folder
to the reference (AuraFace / FaceNet of the mean vectors), then the mean drawing scores
per visitor × variant × identity.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import gen_characters as GC
import triton_local as TL


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference visitor folder (full-res photos)")
    ap.add_argument("--visitors", nargs="+", required=True, help="the visitor folders used in the runs")
    ap.add_argument("--runs", nargs="+", required=True, help="gen_characters.py output folders")
    ap.add_argument("--faceid-repo", required=True)
    ap.add_argument("--models-root", default=os.environ.get("SKETCH_MODELS_ROOT", str(TL.DEV / "models")))
    args = ap.parse_args()

    sys.path.insert(0, args.faceid_repo)
    from faceid.aura import Aura
    aura = Aura(Path(args.models_root) / "AuraFace-v1")
    facenet = GC.FaceNet("cuda" if torch.cuda.is_available() else "cpu")
    ref = GC.analyse_visitor(args.ref, aura, facenet)
    if ref is None:
        sys.exit(f"no face in {args.ref}")

    print("input similarity to the reference (mean vectors):  aura  facenet  photo size")
    for f in args.visitors:
        v = GC.analyse_visitor(f, aura, facenet)
        if v is None:
            print(f"  {Path(f).name}: no face")
            continue
        size = Image.open(next(p for p in sorted(Path(f).iterdir()) if p.suffix.lower() in TL._IMG_EXT)).size
        print(f"  {v['name']:10s} {float(v['aura'] @ ref['aura']):.4f}  {float(v['facenet'] @ ref['facenet']):.4f}  {size}")

    for run in map(Path, args.runs):
        rows = []
        for png in sorted(run.glob("*/*/*.png")):
            vis, ch, tag = png.parent.parent.name, png.parent.name, png.stem
            var, idn = tag.rsplit("_ff", 1)
            idn = "ff" + idn.replace("_", "+")
            rgb = np.array(Image.open(png).convert("RGB"))
            e, fa = aura.embed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            fe = facenet.embed(rgb, fa.bbox) if fa is not None else None
            rows.append({"visitor": vis, "char": ch, "variant": var, "id": idn,
                         "aura_ref": None if e is None else round(float(e @ ref["aura"]), 4),
                         "facenet_ref": None if fe is None else round(float(fe @ ref["facenet"]), 4)})
        if not rows:
            print(f"{run}: no drawings")
            continue
        with (run / "scores_ref.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        agg = defaultdict(lambda: ([], []))
        for r in rows:
            a, b = agg[(r["visitor"], r["variant"], r["id"])]
            if r["aura_ref"] is not None:
                a.append(r["aura_ref"])
            if r["facenet_ref"] is not None:
                b.append(r["facenet_ref"])
        print(f"\n{run}  (vs {Path(args.ref).name})     facenet  aura   n")
        for (vis, var, idn), (a, b) in sorted(agg.items()):
            print(f"  {vis:10s} {var:16s} {idn:8s} {np.mean(b):.4f}  {np.mean(a):.4f}  {len(b)}")


if __name__ == "__main__":
    main()
