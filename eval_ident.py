"""
Identity threshold and face-shape evaluation of gen_characters.py outputs.

FaceNet cosine on a stylised drawing never reaches photo-to-photo levels, so a raw score
says little. This puts each score against wrong-person scores of the same drawing:

  genuine   drawing of visitor V vs V's mean FaceNet vector (the scores.csv number)
  impostor  the same drawing vs every other visitor of the same gender (they share the
            character, style and pose, so only identity differs)
  rank-1    V is the closest of all same-gender visitors to their own drawing
  TAR@tau   share of drawings whose genuine score passes tau, the score that only 1% / 5%
            of impostor comparisons (pooled over every step and identity setting) reach

The same numbers for the visitors' own photos (each photo vs the mean of that person's
other photos) give the general-recogniser reference. Face shape: 68-landmark jaw width /
brow-to-chin height of each ff+ours drawing vs the visitor's front photo.

    python eval_ident.py --runs output/s6_* --gender p0=male ... --out output/s6_eval.json
        [--extra s5/p12-gpt.png=p12]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from faces import face_app, largest_face   # noqa: E402
from gen_characters import FaceNet, unit   # noqa: E402

BINS = [(0, 15), (15, 35), (35, 55), (55, 91)]


def look(rgb, fn):
    """Largest face -> (FaceNet vector, face width/height ratio) or (None, None)."""
    f = largest_face(face_app().get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
    if f is None:
        return None, None
    p = f.landmark_3d_68[:, :2]
    ratio = float(np.linalg.norm(p[0] - p[16]) / np.linalg.norm((p[19] + p[24]) / 2 - p[8]))
    return fn.embed(rgb, f.bbox), ratio


def load(p):
    return np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))


def stats(gen, imp, tau):
    gen, imp = np.array(gen), np.array(imp)
    return {"n": int(len(gen)), "genuine": round(float(gen.mean()), 4), "impostor": round(float(imp.mean()), 4),
            "margin": round(float(gen.mean() - imp.mean()), 4),
            **{f"TAR@{k}": round(float(np.mean(gen >= t)), 3) for k, t in tau.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="gen_characters.py output folders")
    ap.add_argument("--visitors", default=str(_HERE / "mul_face"))
    ap.add_argument("--gender", nargs="+", required=True, metavar="VISITOR=male|female")
    ap.add_argument("--extra", nargs="*", default=[], metavar="IMAGE=VISITOR", help="score other images too")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fn = FaceNet("cuda" if torch.cuda.is_available() else "cpu")
    gender = dict(kv.split("=") for kv in a.gender)

    # visitors: per photo vectors, mean vector, front-photo shape
    photos, mean, shape = {}, {}, {}
    for v in sorted(gender, key=lambda s: int(s[1:])):
        d = Path(a.visitors) / v
        if not d.is_dir():
            continue
        vecs = {}
        for p in sorted(x for x in d.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")):
            e, r = look(load(p), fn)
            if e is not None:
                vecs[p.name] = e
                if "FV" in p.name:
                    shape[v] = r
        photos[v], mean[v] = vecs, unit(np.mean(list(vecs.values()), axis=0))
    same = {v: [o for o in mean if o != v and gender[o] == gender[v]] for v in mean}

    # photo reference: each photo vs the mean of the person's other photos / other people
    pg, pi, prank = [], [], []
    for v, vecs in photos.items():
        for name, e in vecs.items():
            rest = [x for n, x in vecs.items() if n != name]
            if not rest:
                continue
            g = float(e @ unit(np.mean(rest, axis=0)))
            imps = [float(e @ mean[o]) for o in same[v]]
            pg.append(g)
            pi += imps
            prank.append(g > max(imps))

    # drawings
    rows = []
    for run in a.runs:
        run = Path(run)
        if not (run / "scores.csv").is_file():
            continue
        for r in csv.DictReader(open(run / "scores.csv")):
            v, ch, idn = r["visitor"], r["char"], r["id"]
            if v not in mean:
                continue
            png = run / v / ch / f"{r['variant']}_{idn.replace('+', '_')}.png"
            e, ratio = look(load(png), fn)
            yaw = abs(float(r["char_yaw"]))
            if e is None:
                rows.append({"step": run.name, "id": idn, "v": v, "char": ch, "yaw": yaw, "found": False})
                continue
            rows.append({"step": run.name, "id": idn, "v": v, "char": ch, "yaw": yaw, "found": True,
                         "gen": float(e @ mean[v]), "imp": [float(e @ mean[o]) for o in same[v]],
                         "ratio": ratio, "front_shape": shape.get(v)})

    found = [x for x in rows if x["found"]]
    pooled = np.concatenate([x["imp"] for x in found])
    tau = {"FAR1": float(np.quantile(pooled, 0.99)), "FAR5": float(np.quantile(pooled, 0.95))}
    allg = np.array([x["gen"] for x in found])
    ts = np.linspace(-0.2, 1, 1201)
    eer_t = ts[np.argmin([abs(np.mean(pooled >= t) - np.mean(allg < t)) for t in ts])]
    tau["EER"] = float(eer_t)

    rep = {"threshold": {k: round(t, 4) for k, t in tau.items()},
           "photos": {**stats(pg, pi, tau), "rank1": round(float(np.mean(prank)), 3)},
           "steps": {}}
    groups = defaultdict(list)
    for x in rows:
        groups[(x["step"], x["id"])].append(x)
    for (st, idn), xs in sorted(groups.items()):
        ok = [x for x in xs if x["found"]]
        s = stats([x["gen"] for x in ok], np.concatenate([x["imp"] for x in ok]), tau)
        s["rank1"] = round(float(np.mean([x["gen"] > max(x["imp"]) for x in ok])), 3)
        s["missed"] = len(xs) - len(ok)
        s["by_yaw"] = {}
        for lo, hi in BINS:
            b = [x for x in ok if lo <= x["yaw"] < hi]
            if b:
                s["by_yaw"][f"{lo}-{hi}"] = {"n": len(b), "genuine": round(float(np.mean([x["gen"] for x in b])), 4),
                                             "rank1": round(float(np.mean([x["gen"] > max(x["imp"]) for x in b])), 3),
                                             "TAR@FAR1": round(float(np.mean([x["gen"] >= tau["FAR1"] for x in b])), 3),
                                             "shape_ratio": round(float(np.mean([x["ratio"] for x in b])), 3)}
        fr = [x for x in ok if x["yaw"] < 20 and x["front_shape"]]
        if fr:
            s["frontal_shape_abs_err"] = round(float(np.mean([abs(x["ratio"] - x["front_shape"]) for x in fr])), 4)
        rep["steps"][f"{st} {idn}"] = s

    rep["extra"] = {}
    for kv in a.extra:
        p, v = kv.rsplit("=", 1)
        e, _ = look(load(p), fn)
        if e is None:
            rep["extra"][p] = "no face"
            continue
        sc = {o: round(float(e @ mean[o]), 4) for o in [v] + same[v]}
        rep["extra"][p] = {"genuine": sc[v], "best_impostor": max(sc[o] for o in same[v]),
                           "rank1": sc[v] > max(sc[o] for o in same[v]), "passes_FAR1": sc[v] >= tau["FAR1"]}

    rep["visitor_front_shape"] = {k: round(v, 3) for k, v in shape.items() if v}
    Path(a.out).write_text(json.dumps(rep, indent=2))
    with open(Path(a.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "id", "visitor", "char", "yaw", "genuine", "best_impostor", "rank1", "shape_ratio"])
        for x in rows:
            w.writerow([x["step"], x["id"], x["v"], x["char"], x["yaw"]] +
                       ([round(x["gen"], 4), round(max(x["imp"]), 4), x["gen"] > max(x["imp"]), round(x["ratio"], 3)]
                        if x["found"] else ["", "", "", ""]))

    t = rep["threshold"]
    print(f"thresholds (pooled impostors): FAR1 {t['FAR1']:.3f}  FAR5 {t['FAR5']:.3f}  EER {t['EER']:.3f}")
    ph = rep["photos"]
    print(f"photos: genuine {ph['genuine']:.3f}  impostor {ph['impostor']:.3f}  rank1 {ph['rank1']:.2f}  "
          f"TAR@FAR1 {ph['TAR@FAR1']:.2f}")
    print(f"{'step / id':34s} {'n':>4} {'gen':>6} {'imp':>6} {'rank1':>6} {'TAR1':>5} {'TAR5':>5} {'shapeErr':>8}")
    for k, s in rep["steps"].items():
        print(f"{k:34s} {s['n']:4d} {s['genuine']:6.3f} {s['impostor']:6.3f} {s['rank1']:6.2f} "
              f"{s['TAR@FAR1']:5.2f} {s['TAR@FAR5']:5.2f} {s.get('frontal_shape_abs_err', float('nan')):8.3f}")
    for k, s in rep["steps"].items():
        print(f"  {k:32s} " + "  ".join(f"{b} g{y['genuine']:.3f} r{y['rank1']:.2f} sh{y['shape_ratio']:.2f}"
                                          for b, y in s["by_yaw"].items()))
    for p, s in rep["extra"].items():
        print(f"extra {p}: {s}")


if __name__ == "__main__":
    main()
