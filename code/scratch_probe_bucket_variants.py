"""Try remaining bucketing variants to reproduce E7 n_rough=77, cand_p90_rough=28.0/27.6 (global)."""
import sys, glob
import numpy as np
sys.path.insert(0, r"D:\dinghuibianyuan\code")
from qcert.scores import load_dir
from qcert.seastate import clutter_features, roughness_score, SeaStateBucketer
from PIL import Image

ROOT16 = r"D:\dinghuibianyuan\asc-sar\results\scores_ssdd_fp16"
ROOT8 = r"D:\dinghuibianyuan\asc-sar\results\scores_ssdd_int8"
IMGS = r"D:\dinghuibianyuan\asc-sar\datasets\ssdd\images\test"

d16 = load_dir(ROOT16)
d8 = load_dir(ROOT8)
stems = sorted(d16.keys())
assert stems == sorted(d8.keys())

# Precompute features per frame (clutter_features returns dict)
feats = {}
for s in stems:
    img = np.asarray(Image.open(f"{IMGS}\\{s}.jpg").convert("L"), dtype=np.float64) / 255.0
    fd = clutter_features(img)
    feats[s] = (fd["cv"], fd["weibull_c"], fd["contrast"])

def rough_subset_by_score(score_vec, n=77):
    order = np.argsort(-score_vec)
    return set(np.array(stems)[order[:n]])

def eval_subset(sub, root, tag):
    lam_global = {}
    for tag2, dd in (("fp16", d16), ("int8", d8)):
        pool = np.concatenate([f.z() for f in dd.values()])
        lam_global[tag2] = float(np.quantile(pool, 0.999))
    out = {}
    for tag2, dd in (("fp16", d16), ("int8", d8)):
        cands = [int((f.z() > lam_global[tag2]).sum()) for s, f in dd.items() if s in sub]
        out[tag2] = (len(cands), float(np.percentile(cands, 90)))
    print(f"{tag}: n={out['fp16'][0]} p90_fp16={out['fp16'][1]} p90_int8={out['int8'][1]}")
    return out

cv_v = np.array([feats[s][0] for s in stems])
c_v = np.array([feats[s][1] for s in stems])
lc_v = np.array([feats[s][2] for s in stems])
rs = roughness_score([{"cv": feats[s][0], "weibull_c": feats[s][1], "contrast": feats[s][2]} for s in stems])

print("target: n=77 p90_fp16=28.0 p90_int8=27.6")
print("--- single feature terciles (top 1/3 = rough) ---")
eval_subset(rough_subset_by_score(lc_v), None, "local_contrast top77")
eval_subset(rough_subset_by_score(cv_v), None, "cv top77")
eval_subset(rough_subset_by_score(c_v), None, "weibull_c top77")
eval_subset(rough_subset_by_score(-c_v), None, "1/weibull_c top77")

print("--- kmeans (fixed seed 0) ---")
feats_list = [{"cv": feats[s][0], "weibull_c": feats[s][1], "contrast": feats[s][2]} for s in stems]
b = SeaStateBucketer().fit(feats_list)
lab = b.predict(feats_list)
from collections import Counter
print("kmeans label counts:", Counter(lab))
rough = set(np.array(stems)[np.array(lab) == "rough"])
eval_subset(rough, None, "kmeans default")

print("--- contrast >= high quantile (like calib stratified q) ---")
for q in (0.6552, 0.9220, 2/3):
    thr = float(np.quantile(lc_v, q))
    sub = set(s for s in stems if feats[s][2] >= thr)
    eval_subset(sub, None, f"contrast>={q:.4f} thr={thr:.4f}")
