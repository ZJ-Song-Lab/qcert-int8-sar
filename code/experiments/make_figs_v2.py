# -*- coding: utf-8 -*-
"""Sample figures for the experiments chapter (approval round, not final numbering).

三类样例：
  sample_det     识别框图：SSDD test 原图 + GT 框 + 整数阈值告警点（E5S1 dump, α=1e-3）
  sample_heat    热力图：stress_v2 三海况原图 + head1 分数场叠加（cv3_stress dump）
  sample_scatter 散点图：E0 逐场景 P_fa / λ 的 FP16 vs INT8 对比

风格与 make_figs.py 一致（色盲安全五色、rcParams、png+pdf 双输出）。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from qcert.pfa_eval import boxes_to_letterbox, build_omega, calibrate_lambda, cell_centers  # noqa: E402

RES = CODE / "results"
ASC = CODE.parent / "asc-sar"
DUMPS = CODE / "dumps" / "e5s1" / "int8_calmA"
SSDD_IMG = ASC / "datasets" / "ssdd" / "images" / "test"
STRESS_IMG = ASC / "data" / "stress_v2"
CV3 = ASC / "results" / "cv3_stress"
FIGS = CODE.parent / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

C_FP16, C_INT8, C_ACC, C_RED, C_GRAY = "#0072B2", "#E69E00", "#009E73", "#D55E00", "#7F7F7F"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.bottom": False, "axes.spines.left": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

ALPHA = 1e-3
GUARD_PX = 8.0
BLOCK_PX = 64.0
KEEP = 0.5


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


def load_dump(npz_path):
    d = np.load(npz_path)
    meta = json.loads(str(d["meta"]))
    return d, meta


def grid_of(prob, head_sizes, k):
    """第 k 个 head 的稠密分数图 (h,w)，与 qcert.scores.ScoreField.grid 一致。"""
    hs = np.asarray(head_sizes)
    off = int(np.prod(hs[:k, :]).sum()) if k > 0 else 0
    h, w = int(hs[k, 0]), int(hs[k, 1])
    return prob[off:off + h * w].reshape(h, w)


def lb_to_orig(xy, meta):
    """letterbox 网格中心 -> 原图像素坐标。"""
    ratio = float(meta["ratio"])
    px, py = (float(v) for v in meta["pad"])
    x = (xy[:, 0] - px) / ratio
    y = (xy[:, 1] - py) / ratio
    return x, y


def near_boxes(centers, lb_boxes, guard_px):
    """每个网格中心是否落在任一 GT 框的 guard 邻域内（letterbox 坐标）。"""
    if lb_boxes is None or len(lb_boxes) == 0:
        return np.zeros(len(centers), dtype=bool)
    g = guard_px
    exp = lb_boxes.copy()
    exp[:, 0] -= g; exp[:, 1] -= g; exp[:, 2] += g; exp[:, 3] += g
    inside = ((centers[:, None, 0] >= exp[None, :, 0]) & (centers[:, None, 0] <= exp[None, :, 2]) &
              (centers[:, None, 1] >= exp[None, :, 1]) & (centers[:, None, 1] <= exp[None, :, 3]))
    return inside.any(axis=1)


# ---------------------------------------------------------------- 样例1: 识别框图
def sample_det():
    # 扫描全部 dump，挑 3 个代表场景：多舰 / 单舰 / 纯杂波
    files = sorted(DUMPS.glob("*.npz"))
    counts = {}
    for f in files:
        _, m = load_dump(f)
        counts[f.stem] = len(m.get("boxes", []))
    multi = max(counts, key=counts.get)
    single = next(s for s in sorted(counts) if counts[s] == 1)
    mid = min(sorted(counts), key=lambda s: abs(counts[s] - 5))
    stems = [multi, mid, single]
    print("det scenes:", [(s, counts[s]) for s in stems])

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    for ax, letter, stem in zip(axes, "abc", stems):
        d, meta = load_dump(DUMPS / f"{stem}.npz")
        z = d["prob"].astype(np.float64)
        imgsz = int(meta.get("imgsz", 640))
        centers = cell_centers(d["head_sizes"], imgsz)
        boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
        lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
        mc, mm, info = build_omega(centers, lb, guard_px=GUARD_PX,
                                   block_px=BLOCK_PX, keep=KEEP)
        lam = calibrate_lambda(z, mc, ALPHA)
        alarm = z > lam
        hit = near_boxes(centers, lb, GUARD_PX) & alarm

        iw, ih = int(meta["iw"]), int(meta["ih"])
        ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
        # letterbox 统一显示：所有 SAR 场景投影到同一 640x640 画布
        ax.imshow(plt.imread(SSDD_IMG / f"{stem}.jpg"), cmap="gray",
                  extent=(px, px + iw * ratio, py + ih * ratio, py))
        # GT 框（letterbox 坐标）
        for b in (lb if lb is not None else []):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, ec=C_ACC, lw=0.7))
        # 告警点：命中=绿，虚警=红（centers 本就是 letterbox 坐标）
        fp = alarm & ~hit
        ax.scatter(centers[fp, 0], centers[fp, 1], s=14, marker="x", c=C_RED,
                   lw=0.9, zorder=3)
        ax.scatter(centers[hit, 0], centers[hit, 1], s=10, c=C_ACC, zorder=4,
                   edgecolors="none")
        n_fp = int(fp.sum())
        ax.set_title(f"({letter}) {stem}: {len(boxes)} GT, "
                     f"{int(alarm.sum())} alarms ({n_fp} FP)", fontsize=8)
        ax.set_xlim(0, 640); ax.set_ylim(640, 0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].scatter([], [], s=10, c=C_ACC, label="alarm on ship")
    axes[0].scatter([], [], s=14, marker="x", c=C_RED, label="false alarm")
    axes[0].plot([], [], c=C_ACC, lw=0.7, label="GT box")
    axes[0].legend(loc="lower right", framealpha=0.85, fontsize=6.5)
    fig.suptitle(r"INT8 OM, per-scene $\lambda$ at $\alpha=10^{-3}$ "
                 f"(guard {GUARD_PX:.0f}px, letterbox)", fontsize=8.5, y=1.0)
    fig.tight_layout()
    save(fig, "sample_det")


# ---------------------------------------------------------------- 样例2: 热力图
def sample_heat():
    stems = [sorted(STRESS_IMG.glob(f"{c}_*.jpg"))[0].stem for c in ("c1", "c2", "c3")]
    print("heat scenes:", stems)
    vmax = 0.0
    grids, metas, imgs = [], [], []
    for stem in stems:
        d, meta = load_dump(CV3 / f"{stem}.npz")
        g = grid_of(d["prob"].astype(np.float64), d["head_sizes"], 1)  # head1: 40x40
        grids.append(g); metas.append(meta)
        imgs.append(plt.imread(STRESS_IMG / f"{stem}.jpg"))
        vmax = max(vmax, float(g.max()))

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.7))
    norm = PowerNorm(0.3, vmin=0.0, vmax=vmax)
    for j, (stem, img, g, meta) in enumerate(zip(stems, imgs, grids, metas)):
        iw, ih = int(meta["iw"]), int(meta["ih"])
        ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
        for i, ax in enumerate((axes[0, j], axes[1, j])):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 640); ax.set_ylim(640, 0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.imshow(img, cmap="gray", extent=(px, px + iw * ratio,
                                                     py + ih * ratio, py))
                ax.set_title(f"({'abc'[j]}) {stem} (sea {j + 1})", fontsize=8.5)
                if j == 0:
                    ax.set_ylabel("raw SAR input", fontsize=8)
            else:
                ax.imshow(img, cmap="gray", extent=(px, px + iw * ratio,
                                                     py + ih * ratio, py))
                im = ax.imshow(g, cmap="inferno", norm=norm, alpha=0.55,
                               extent=(0, 640, 640, 0), interpolation="bilinear")
                if j == 0:
                    ax.set_ylabel("score field $s(u)$, head 1", fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label(r"$s(u)$ (power norm, $\gamma$=0.3)", fontsize=8)
    save(fig, "sample_heat")


# ---------------------------------------------------------------- 样例3: 散点图
def sample_scatter():
    df = pd.read_csv(RES / "e0" / "e0_per_scene.csv")
    n = len(df)

    def clip0(v, fallback):
        v = np.asarray(v, dtype=float).copy()
        v[v <= 0] = fallback
        return v

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    panels = [
        ("pfa_ref", "pfa_test", r"per-scene $P_{fa}$",
         0.5 / df["n_meas"].max(), (a := r"$R=$median$=%.2f$" % df["R"].median())),
        ("lambda_ref", "lambda_test", r"per-scene $\lambda$",
         1e-9, r"$\lambda_{INT8}/\lambda_{FP16}$ median $=%.2f$"
         % float((df["lambda_test"] / df["lambda_ref"]).median())),
    ]
    for k, (ax, (xc, yc, lab, fb, note)) in enumerate(zip(axes, panels)):
        x = clip0(df[xc], fb); y = clip0(df[yc], fb)
        ax.scatter(x, y, s=9, alpha=0.5, c=C_INT8, edgecolors="none")
        lo = min(x.min(), y.min()) * 0.6
        hi = max(x.max(), y.max()) * 1.6
        ax.plot([lo, hi], [lo, hi], c=C_GRAY, ls="--", lw=0.9, zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{lab}, FP16 OM")
        ax.set_ylabel(f"{lab}, INT8 OM")
        ax.set_title(f"({'ab'[k]}) {lab}, $n={n}$", fontsize=9)
        ax.text(0.05, 0.93, note, transform=ax.transAxes, fontsize=7.5, va="top")
    fig.tight_layout()
    save(fig, "sample_scatter")


# ------------------------------------------------- 样例4: E5S1 阈值迁移识别框对比
def _mu_sigma_blocks(s, centers, block_px=64.0):
    """run_e5._mu_sigma 的复刻：块内 median / std 的逐位置 (mu, sigma) 代理。"""
    from qcert.pfa_eval import block_ids
    ids = block_ids(centers, block_px)
    mu = np.empty_like(s)
    sig = np.empty_like(s)
    for bid in np.unique(ids):
        m = ids == bid
        v = s[m]
        mu[m] = np.median(v)
        sig[m] = max(float(np.std(v)), 1e-3 * max(float(np.median(np.abs(v))), 1e-6))
    return mu, sig


def _logit_prob(prob):
    """OM 输出 prob 的反变换 ln(p/(1-p))——e5_s1 实跑的分数域（spec domain）。

    注意：npz 里的 "logit" 是 sigmoid 前的卷积原值，与 prob 不同源
    （prob 经逐头 Δ 量化后再 sigmoid），两者差异可达 O(10)，不可混用。
    """
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _e5s1_scene(stem, spec_old, spec_new):
    """复现 e5_s1 逐场景管线，返回绘图所需一切 + 与 CSV 对账用的中间量。

    分数域：logit(prob) 反变换（λ 与 hits 均已与 e5_per_scene.csv 精确对账）。
    """
    from qcert.transfer import (lambda_int_from_float, lambda_float_from_int,
                                transport_lambda_int)
    dA, mA = load_dump(DUMPS.parent / "int8_calmA" / f"{stem}.npz")
    dB, mB = load_dump(DUMPS.parent / "int8_roughB" / f"{stem}.npz")
    centers = cell_centers(dA["head_sizes"], int(mA.get("imgsz", 640)))
    boxes = np.asarray(mA.get("boxes", []), dtype=float).reshape(-1, 4)
    lb = boxes_to_letterbox(boxes, mA) if len(boxes) else None
    mc, mm, info = build_omega(centers, lb, guard_px=GUARD_PX,
                               block_px=BLOCK_PX, keep=KEEP)
    sA = _logit_prob(dA["prob"])
    sB = _logit_prob(dB["prob"])
    muA, sigA = _mu_sigma_blocks(sA, centers, BLOCK_PX)
    muB, sigB = _mu_sigma_blocks(sB, centers, BLOCK_PX)
    zA = (sA - muA) / sigA
    # 实跑口径（run_e5 run_dumps）：B 域的 σ 量化到 Δ_σ 网格后再进 z，
    # 否则个别场景 hits 会差 1（已验证：量化后与 CSV 逐场景一致）。
    sigB_q = np.maximum(np.round(sigB / spec_new.delta_sigma) * spec_new.delta_sigma,
                        1e-6)
    zB = (sB - muB) / sigB_q
    lam_z_old = calibrate_lambda(zA, mc, ALPHA)
    lam_int_old = lambda_int_from_float(lam_z_old, spec_old)
    lam_int_new = transport_lambda_int(lam_int_old, spec_old, spec_new)
    lams = {
        "transport": lambda_float_from_int(lam_int_new, spec_new),
        "no_convert": lambda_float_from_int(lam_int_old, spec_new),
        "recal": calibrate_lambda(zB, mc, ALPHA),
    }
    hits = {k: int((zB[mm] > v).sum()) for k, v in lams.items()}
    return dict(stem=stem, meta=mA, centers=centers, boxes=boxes, mm=mm,
                zB=zB, lams=lams, hits=hits, n_meas=int(mm.sum()),
                lam_z_old=float(lam_z_old), lam_int_old=lam_int_old,
                lam_int_new=lam_int_new)


def _ship_area(stem):
    """平均 GT 船面积占整幅比例——用于挑船体大的场景（避免框压住小船）。"""
    _, m = load_dump(DUMPS / f"{stem}.npz")
    b = np.asarray(m.get("boxes", []), dtype=float).reshape(-1, 4)
    if len(b) == 0:
        return 0.0
    iw, ih = float(m["iw"]), float(m["ih"])
    return float(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).mean() / (iw * ih))


def sample_transport():
    import json as _json
    from qcert.transfer import TransportSpec
    def _spec(path):
        d = _json.load(open(path))
        return TransportSpec(**{k: d[k] for k in
                                ("delta_s", "delta_sigma", "shift_bits") if k in d})
    spec_old = _spec(CODE / "specs" / "e5s1_calmA_spec.json")
    spec_new = _spec(CODE / "specs" / "e5s1_roughB_spec.json")
    csv = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                      dtype={"stem": str})
    csv["stem"] = csv["stem"].str.zfill(6)
    # 选 no_convert / transport 虚警倍数大、样本足够、且船体足够大的场景
    # （对比最刺眼 + 框不挡船：平均 GT 面积 >= 0.2% 图幅）
    ratio = csv["hits_no_convert"] / csv["hits_transport"].clip(lower=1)
    pick = csv[(csv["hits_no_convert"] >= 5)].assign(ratio=ratio)
    pick["area"] = pick["stem"].map(_ship_area)
    big = pick[pick["area"] >= 0.002]
    row_max = (big if len(big) else pick).sort_values(
        ["ratio", "area"], ascending=False).iloc[0]
    print(f"[pick] {row_max['stem']}: ratio={row_max['ratio']:.1f}, "
          f"mean ship area={row_max['area'] * 100:.2f}% of image")
    stems = [row_max["stem"]]
    scenes = []
    for stem in stems:
        sc = _e5s1_scene(stem, spec_old, spec_new)
        ref = csv[csv["stem"] == stem].iloc[0]
        print(f"[validate {stem}] lam_z_old {sc['lam_z_old']:.5f} vs csv "
              f"{ref['lambda_z_old']:.5f} | lam_int_old {sc['lam_int_old']} vs "
              f"{ref['lambda_int_old']} | lam_int_new {sc['lam_int_new']} vs "
              f"{ref['lambda_int_new']} | hits {sc['hits']} vs csv "
              f"t={ref['hits_transport']} n={ref['hits_no_convert']} "
              f"r={ref['hits_recal']}")
        scenes.append(sc)

    order = ["transport", "no_convert", "recal"]
    titles = {"transport": r"(a) transported $\lambda_{int}$",
              "no_convert": r"(b) $\lambda_{int}$ reused, no conversion",
              "recal": r"(c) recalibrated on B (oracle)"}
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1))
    for ax, key in zip(axes, order):
        sc = scenes[0]
        meta = sc["meta"]
        alarm = sc["zB"] > sc["lams"][key]
        lb = boxes_to_letterbox(sc["boxes"], meta) if len(sc["boxes"]) else None
        hit = near_boxes(sc["centers"], lb, GUARD_PX) & alarm
        iw, ih = int(meta["iw"]), int(meta["ih"])
        ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
        # letterbox 统一显示（640x640 画布）
        ax.imshow(plt.imread(SSDD_IMG / f"{sc['stem']}.jpg"), cmap="gray",
                  extent=(px, px + iw * ratio, py + ih * ratio, py))
        for b in (lb if lb is not None else []):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, ec=C_ACC, lw=0.5))
        fp = alarm & ~hit
        ax.scatter(sc["centers"][fp, 0], sc["centers"][fp, 1], s=14,
                   marker="x", c=C_RED, lw=0.9, zorder=3)
        ax.scatter(sc["centers"][hit, 0], sc["centers"][hit, 1], s=10, c=C_ACC,
                   zorder=4, edgecolors="none")
        h, n = sc["hits"][key], sc["n_meas"]
        ax.set_title(f"{titles[key]}\n"
                     r"$P_{fa}$=" + f"{h / n:.2e} ({h}/{n})", fontsize=8)
        ax.set_xlim(0, 640); ax.set_ylim(640, 0)
        ax.set_xticks([]); ax.set_yticks([])
    s = _json.load(open(RES / "e5_s1" / "summary.json", encoding="utf-8"))["rows"][0]
    fig.suptitle(f"E5 setting 1 (A$\\to$B, 232 scenes): pooled $P_{{fa}}$ "
                 f"transport={s['pfa_transport']:.2e}, no-convert="
                 f"{s['pfa_no_convert']:.2e}, recal={s['pfa_recal']:.2e} "
                 f"($\\alpha$={ALPHA:g})", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sample_transport")


# ------------------------------------------------- 样例5: Omega 划分示意
def sample_omega():
    stem = "000059"
    d, meta = load_dump(DUMPS / f"{stem}.npz")
    centers = cell_centers(d["head_sizes"], int(meta.get("imgsz", 640)))
    boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
    lb = boxes_to_letterbox(boxes, meta)
    mc, mm, info = build_omega(centers, lb, guard_px=GUARD_PX,
                               block_px=BLOCK_PX, keep=KEEP)
    iw, ih = int(meta["iw"]), int(meta["ih"])
    ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
    guard = near_boxes(centers, lb, GUARD_PX)

    fig, ax = plt.subplots(figsize=(4.4, 4.3))
    ax.imshow(plt.imread(SSDD_IMG / f"{stem}.jpg"), cmap="gray",
              extent=(px, px + iw * ratio, py + ih * ratio, py))
    ax.set_xlim(0, 640); ax.set_ylim(640, 0)
    # 块网格（64px letterbox）
    for t in np.arange(0, 641, BLOCK_PX):
        ax.axhline(t, c=C_GRAY, lw=0.35, alpha=0.5)
        ax.axvline(t, c=C_GRAY, lw=0.35, alpha=0.5)
    ax.scatter(*(centers[mc].T), s=2.2, c=C_GRAY, alpha=0.55, label=r"$\Omega_{cal}$"
               f" ({mc.sum()})", linewidths=0)
    ax.scatter(*(centers[mm].T), s=2.2, c=C_FP16, alpha=0.55, label=r"$\Omega_{meas}$"
               f" ({mm.sum()})", linewidths=0)
    ax.scatter(*(centers[guard].T), s=2.2, c=C_RED, alpha=0.8,
               label=f"GT guard ({guard.sum()})", linewidths=0)
    for b in lb:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, ec=C_ACC, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7, handletextpad=0.1)
    ax.set_title(f"Omega split, {stem} (letterbox 640, block "
                 f"{BLOCK_PX:.0f}px, keep={KEEP})", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sample_omega")


# ------------------------------------------------- 样例6: 杂波分数 CCDF 尾部
def sample_ccdf():
    states = ["c1", "c2", "c3"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for j, st in enumerate(states):
        vals = []
        for f in sorted(CV3.glob(f"{st}_*.npz")):
            d, meta = load_dump(f)
            s = d["prob"].astype(np.float64)
            centers = cell_centers(d["head_sizes"], int(meta.get("imgsz", 640)))
            boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
            lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
            mask = ~near_boxes(centers, lb, GUARD_PX) if lb is not None \
                else np.ones(len(s), bool)
            vals.append(s[mask])
        v = np.sort(np.concatenate(vals))
        ccdf = 1.0 - np.arange(1, len(v) + 1) / (len(v) + 1)
        lam = calibrate_lambda(v, np.ones(len(v), bool), ALPHA)
        ax.plot(v, ccdf, c=[C_FP16, C_INT8, C_RED][j], lw=1.2,
                label=f"sea {j + 1} ({st}, n={len(v)})")
        ax.axvline(lam, c=[C_FP16, C_INT8, C_RED][j], ls=":", lw=1.0)
        print(f"{st}: lam={lam:.3e}, P(s>lam)={float((v > lam).mean()):.3e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"clutter score $s(u)$")
    ax.set_ylabel(r"$\Pr[s > t]$")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"Pooled clutter tail per sea state, "
                 r"$\lambda$ at $\alpha=10^{-3}$ (dotted)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sample_ccdf")


# ------------------------------------------------- 样例7: 整数码域可视化
_HEAD_NAMES = ("cv3.0.2", "cv3.1.2", "cv3.2.2")


def _head_deltas():
    """从 int8 scale 表取三个输出头的逐头 Δ（npz 里的标量 delta 只是 head0 的）。"""
    d = json.load(open(ASC / "models" / "yolo11n_ssdd_int8_scale.json",
                       encoding="utf-8"))
    return {e["name"].split("/")[-2]: float(e["delta"]) for e in d["entries"]
            if e["name"].endswith("/Conv_quant_dequant")
            and e["name"].split("/")[-2] in _HEAD_NAMES}


def sample_intcode():
    hd = _head_deltas()

    def logit_of(d):
        """s_q -> logit 域（逐头 Δ），并做映射自检。"""
        hs = np.asarray(d["head_sizes"])
        out = np.empty(len(d["s_q"]), dtype=np.float64)
        off = 0
        for i, (h, w) in enumerate(hs):
            n = int(h) * int(w)
            out[off:off + n] = d["s_q"][off:off + n].astype(np.float64) * hd[_HEAD_NAMES[i]]
            off += n
        return out

    # 映射自检：prob ?= sigmoid(s_q * delta_head)（逐头）
    f0 = sorted(CV3.glob("c2_*.npz"))[0]
    d0, _ = load_dump(f0)
    err = float(np.abs(1.0 / (1.0 + np.exp(-logit_of(d0)))
                       - d0["prob"].astype(np.float64)).max())
    print(f"intcode mapping check (per-head delta): max err = {err:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {"c1": C_FP16, "c2": C_INT8, "c3": C_RED}
    for st in ("c1", "c2", "c3"):
        codes = []
        for f in sorted(CV3.glob(f"{st}_*.npz")):
            d, meta = load_dump(f)
            centers = cell_centers(d["head_sizes"], int(meta.get("imgsz", 640)))
            boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
            lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
            mask = ~near_boxes(centers, lb, GUARD_PX) if lb is not None \
                else np.ones(len(d["s_q"]), bool)
            codes.append(d["s_q"][mask].astype(np.int64))
        c = np.concatenate(codes)
        axes[0].hist(c, bins=120, histtype="step", lw=1.1, color=colors[st],
                     label=f"sea {st[1]} (n={len(c)})")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"integer score code $s_q$ (int32)")
    axes[0].set_ylabel("count")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].set_title("(a) code histogram, clutter cells", fontsize=8.5)

    # (b) 逐头 Δ 的确定性映射：x = s_q*Δ_head（logit 域），y = sigmoid(x)
    rng = np.random.default_rng(0)
    for st in ("c1", "c2", "c3"):
        xs, ys = [], []
        for f in sorted(CV3.glob(f"{st}_*.npz"))[:6]:
            d, meta = load_dump(f)
            centers = cell_centers(d["head_sizes"], int(meta.get("imgsz", 640)))
            boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
            lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
            mask = ~near_boxes(centers, lb, GUARD_PX) if lb is not None \
                else np.ones(len(d["s_q"]), bool)
            lg = logit_of(d)[mask]
            sub = lg[rng.choice(len(lg), size=min(4000, len(lg)), replace=False)]
            xs.append(sub)
            ys.append(1.0 / (1.0 + np.exp(-sub)))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        o = np.argsort(x)
        axes[1].plot(x[o], y[o], c=colors[st], lw=1.0, alpha=0.8,
                     label=f"sea {st[1]}")
    axes[1].set_xlabel(r"$s_q \cdot \Delta_S^{(head)}$ (logit domain)")
    axes[1].set_ylabel(r"$s(u)=\sigma(s_q \Delta_S^{(head)})$")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].set_title("(b) deterministic int$\\to$float mapping", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sample_intcode")


# ------------------------------------------------- 消融样例（E9 / E4）

def _e9tables():
    real = pd.read_csv(RES / "e9_real" / "table3_e9_real.csv")
    synth = pd.read_csv(RES / "e9" / "table3_e9.csv")
    return real, synth


def _e9val(df, item, setting, metric):
    r = df[(df["item"] == item) & (df["setting"] == setting)
           & (df["metric"] == metric)]
    return float(r["value"].iloc[0])


def sample_ablation_h0():
    """E9 item6：z=(s−μ)/σ 归一化（h0）vs 直接分数阈值——方法核心设计消融。"""
    _, synth = _e9tables()
    item = "6 有/无 h0"
    zs = "z = (s−μ)/σ"          # U+2212，与 CSV 落盘一致
    ss = "s（无 h0，直接取阈值）"
    zR, zPd = (_e9val(synth, item, zs, "R（INT8/FP32，汇总）"),
               _e9val(synth, item, zs, "P_d@1e-3（INT8，汇总）"))
    sR, sPd = (_e9val(synth, item, ss, "R（INT8/FP32，汇总）"),
               _e9val(synth, item, ss, "P_d@1e-3（INT8，汇总）"))

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.6))
    x = np.arange(2)
    w = 0.56
    for ax, vals, ylab, title in (
            (axes[0], [zR, sR], "detection ratio $R$ (INT8/FP32)",
             "(a) cross-config ratio"),
            (axes[1], [zPd, sPd], "$P_d$ @ $\\alpha=10^{-3}$",
             "(b) detection probability")):
        bars = ax.bar(x, vals, w, color=[C_ACC, C_GRAY])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["$z=(s-\\mu)/\\sigma$", "raw $s$"], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].annotate(f"{zPd / sPd:.1f}×", xy=(1, sPd + 0.06),
                     xytext=(0.42, zPd * 0.9), fontsize=9, color=C_RED,
                     arrowprops=dict(arrowstyle="->", color=C_RED, lw=1))
    axes[0].axhline(1.0, color=C_RED, lw=0.8, ls="--")
    fig.suptitle("Ablation: score normalization (synthetic, SSDD 232)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "sample_ablation_h0")


def sample_ablation_keep():
    """E9 item3：块抽样 keep 比扫描（真实 232 图 + 合成下界）。"""
    real, synth = _e9tables()
    item = "3 块抽样"
    keeps = [1.0, 0.25, 0.125, 0.0625]
    mP, mE, mN = ("P_fa", "α 误差 |log10(P_fa/α)|", "n_eff（测量块数，均值）")

    keeps_s = ["1", "0.25", "0.125", "0.0625"]
    keeps = [1.0, 0.25, 0.125, 0.0625]

    pr = [_e9val(real, item, f"keep={k}", mP) for k in keeps_s]
    er = [_e9val(real, item, f"keep={k}", mE) for k in keeps_s]
    nr = [_e9val(real, item, f"keep={k}", mN) for k in keeps_s]
    ps = [_e9val(synth, item, f"keep={k}", mP) for k in keeps_s]
    es = [_e9val(synth, item, f"keep={k}", mE) for k in keeps_s]
    ns = [_e9val(synth, item, f"keep={k}", mN) for k in keeps_s]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    ax.plot(keeps, ps, "s-", color=C_FP16, ms=4, label="synthetic")
    ax.plot(keeps, pr, "o-", color=C_INT8, ms=4, label="real (232 imgs)")
    ax.axhline(1e-3, color=C_RED, lw=0.8, ls="--", label=r"$\alpha=10^{-3}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block-sampling keep ratio")
    ax.set_ylabel(r"$P_{fa}$")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(a) false-alarm rate vs. keep", fontsize=8.5)

    ax = axes[1]
    ax.plot(ns, es, "s-", color=C_FP16, ms=4, label="synthetic")
    ax.plot(nr, er, "o-", color=C_INT8, ms=4, label="real (232 imgs)")
    ax.set_xscale("log")
    ax.set_xlabel(r"effective blocks per scene $n_{\mathrm{eff}}$")
    ax.set_ylabel(r"err $=|\log_{10}(P_{fa}/\alpha)|$ (dex)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(b) contract error vs. sample size", fontsize=8.5)
    fig.suptitle(r"Ablation: block sampling (reps=10, $\alpha=10^{-3}$)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "sample_ablation_keep")


def sample_ablation_ncal():
    """E9 item4：标定样本量扫描（真实 vs 合成，mean±std）。"""
    real, synth = _e9tables()
    item = "4 标定样本量"
    mMean, mStd = ("α 误差 |log10|（均值）", "α 误差 |log10|（std）")

    nr = [16, 64, 128, 200]
    mr = np.array([_e9val(real, item, f"{n} 场景", mMean) for n in nr])
    sr = np.array([_e9val(real, item, f"{n} 场景", mStd) for n in nr])
    ns = [16, 64, 256]
    ms = np.array([_e9val(synth, item, f"{n} 场景", mMean) for n in ns])
    ss = np.array([_e9val(synth, item, f"{n} 场景", mStd) for n in ns])

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.errorbar(ns, ms, yerr=ss, fmt="s-", color=C_FP16, ms=4, capsize=2.5,
                lw=1.1, label="synthetic")
    ax.errorbar(nr, mr, yerr=sr, fmt="o-", color=C_INT8, ms=4, capsize=2.5,
                lw=1.1, label="real (232 imgs)")
    ax.set_xlabel("calibration scenes $n$")
    ax.set_ylabel(r"err $=|\log_{10}(P_{fa}/\alpha)|$ (dex)")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Ablation: calibration set size (reps=10)", fontsize=8.5)
    ax.annotate("block-sampling\nboundary effect", xy=(200, 0.151),
                xytext=(80, 0.19), fontsize=7, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))
    fig.tight_layout()
    save(fig, "sample_ablation_ncal")


def sample_ablation_scale():
    """E4 标度律 + E9 item8 舍入方向：(a) |Δln P_fa| ~ ε_σ^p（5 配置）；
    (b) round（二阶）vs floor（一阶，Prop.2 预言）。"""
    e4 = pd.read_csv(RES / "e4" / "table3_e4.csv")
    _, synth = _e9tables()
    item8 = "8 floor vs round"
    eps8 = ["0.02", "0.05", "0.10", "0.20"]
    dl_round = np.array([_e9val(synth, item8, f"round @ ε_σ={e}",
                                "Δln P_fa（汇总）") for e in eps8])
    dl_floor = np.array([_e9val(synth, item8, f"floor @ ε_σ={e}",
                                "Δln P_fa（汇总）") for e in eps8])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    colors = {"linear_tensor": C_INT8, "log_tensor": C_ACC,
              "channel_linear": C_RED, "channel_log": "#CC79A7", "fp16": C_GRAY}
    noise = float(e4.loc[e4["config"] == "fp16", "dln_meas"].abs().mean())
    for cfg, g in e4.groupby("config"):
        g = g.sort_values("eps_sigma")
        x, y, s = (g["eps_sigma"].to_numpy(), np.abs(g["dln_meas"].to_numpy()),
                   g["dln_std"].to_numpy())
        # 信息点口径（与 Table 5 一致）：剔除 |Δln| ≤ fp16 噪声底的点再拟合
        info = y > noise if cfg != "fp16" else np.ones(len(y), bool)
        if info.sum() >= 2 and cfg != "fp16":
            p = np.polyfit(np.log10(x[info]), np.log10(y[info]), 1)[0]
            lab = f"{cfg} ($p$={p:.2f})"
        elif cfg == "fp16":
            lab = f"{cfg} (noise floor)"
        else:
            lab = cfg
        ax.errorbar(x, y, yerr=s, fmt="o-", ms=3.5, lw=1.0, capsize=2,
                    color=colors[cfg], label=lab)
        if (~info).any():  # 非信息点改空心
            ax.plot(x[~info], y[~info], "o", ms=4, mfc="white",
                    mec=colors[cfg], mew=1.0)
    xx = np.array([0.02, 0.1])
    ax.plot(xx, 0.06 * (xx / 0.1) ** 2, "k:", lw=0.9, label=r"$\propto\varepsilon^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_\sigma$")
    ax.set_ylabel(r"$|\Delta\ln P_{fa}|$")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.set_title(r"(a) $\sigma$-quantization scaling (E4, gauss)", fontsize=8.5)

    ax = axes[1]
    ax.plot(eps8, dl_round, "o-", color=C_ACC, ms=4, label="round (order 2)")
    ax.plot(eps8, dl_floor, "s-", color=C_RED, ms=4, label="floor (order 1)")
    xx = np.array([0.02, 0.2])
    ax.plot(xx, 0.15 * (xx / 0.05) ** 2, "k:", lw=0.9, label=r"$\propto\varepsilon^2$")
    ax.plot(xx, 3.6 * (xx / 0.2) ** 1, "k--", lw=0.9, label=r"$\propto\varepsilon^1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_\sigma$")
    ax.set_ylabel(r"$\Delta\ln P_{fa}$")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(b) rounding direction (E9 item8, synthetic)", fontsize=8.5)
    fig.tight_layout()
    save(fig, "sample_ablation_scale")


# ------------------------------------------------- 多典型场景大拼图

def _plot_det_panel(ax, letter, stem, tag):
    """单个识别框面板（与 sample_det 同口径：prob 域逐场景 λ）。"""
    d, meta = load_dump(DUMPS / f"{stem}.npz")
    z = d["prob"].astype(np.float64)
    imgsz = int(meta.get("imgsz", 640))
    centers = cell_centers(d["head_sizes"], imgsz)
    boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
    lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
    mc, mm, info = build_omega(centers, lb, guard_px=GUARD_PX,
                               block_px=BLOCK_PX, keep=KEEP)
    lam = calibrate_lambda(z, mc, ALPHA)
    alarm = z > lam
    hit = near_boxes(centers, lb, GUARD_PX) & alarm
    iw, ih = int(meta["iw"]), int(meta["ih"])
    ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
    # letterbox 统一显示（640x640 画布）
    ax.imshow(plt.imread(SSDD_IMG / f"{stem}.jpg"), cmap="gray",
              extent=(px, px + iw * ratio, py + ih * ratio, py))
    for b in (lb if lb is not None else []):
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, ec=C_ACC, lw=0.7))
    fp = alarm & ~hit
    ax.scatter(centers[fp, 0], centers[fp, 1], s=12, marker="x", c=C_RED,
               lw=0.8, zorder=3)
    ax.scatter(centers[hit, 0], centers[hit, 1], s=9, c=C_ACC, zorder=4,
               edgecolors="none")
    ax.set_title(f"({letter}) {stem} {tag}: {len(boxes)} GT, "
                 f"{int(alarm.sum())} alarms ({int(fp.sum())} FP)",
                 fontsize=8.5)
    ax.set_xlim(0, 640); ax.set_ylim(640, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def sample_det_mosaic():
    """2x3 六典型场景：密集编队 / 小目标群 / 多尺度混编 / 大型舰船 / 单舰 / 典型中量。"""
    files = sorted(DUMPS.glob("*.npz"))
    stats = []
    for f in files:
        _, m = load_dump(f)
        b = np.asarray(m.get("boxes", []), dtype=float).reshape(-1, 4)
        if len(b) == 0:
            continue
        iw, ih = float(m["iw"]), float(m["ih"])
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (iw * ih)
        stats.append(dict(stem=f.stem, n=len(b), area=float(areas.mean()),
                          cv=float(areas.std() / max(areas.mean(), 1e-9))))
    picked = []

    def take(pred, key):
        chosen = {p["stem"] for p in picked}
        cand = [s for s in stats if pred(s) and s["stem"] not in chosen]
        if cand:
            picked.append(max(cand, key=key))

    take(lambda s: True, lambda s: s["n"])              # 密集编队（GT 最多）
    take(lambda s: s["n"] >= 4, lambda s: -s["area"])   # 小目标群（均值面积最小）
    take(lambda s: s["n"] >= 3, lambda s: s["cv"])      # 多尺度混编（面积 CV 最大）
    take(lambda s: s["n"] >= 2, lambda s: s["area"])    # 大型舰船（均值面积最大）
    take(lambda s: s["n"] == 1, lambda s: s["stem"])    # 单舰
    take(lambda s: True, lambda s: -abs(s["n"] - 5))    # 典型 5 艘
    tags = ["dense fleet", "small ships", "multi-scale",
            "large ships", "single ship", "typical (5)"]
    print("det mosaic scenes:", [(p["stem"], p["n"], round(p["area"], 5))
                                 for p in picked])

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.5))
    for k, (ax, p, tag) in enumerate(zip(axes.ravel(), picked, tags)):
        _plot_det_panel(ax, "abcdef"[k], p["stem"], tag)
    axes[0, 0].scatter([], [], s=9, c=C_ACC, label="alarm on ship")
    axes[0, 0].scatter([], [], s=12, marker="x", c=C_RED, label="false alarm")
    axes[0, 0].plot([], [], c=C_ACC, lw=0.7, label="GT box")
    axes[0, 0].legend(loc="lower right", framealpha=0.85, fontsize=7.5)
    fig.suptitle(r"INT8 OM, per-scene $\lambda$ at $\alpha=10^{-3}$: "
                 f"six typical SSDD scenes (guard {GUARD_PX:.0f}px)",
                 fontsize=9, y=0.99)
    fig.tight_layout()
    save(fig, "sample_det_mosaic")


def sample_transport_mosaic():
    """2 场景 x 3 设置：阈值迁移三联对比，挑虚警倍数最大的两个场景。"""
    import json as _json
    from qcert.transfer import TransportSpec

    def _spec(path):
        d = _json.load(open(path))
        return TransportSpec(**{k: d[k] for k in
                                ("delta_s", "delta_sigma", "shift_bits") if k in d})
    spec_old = _spec(CODE / "specs" / "e5s1_calmA_spec.json")
    spec_new = _spec(CODE / "specs" / "e5s1_roughB_spec.json")
    csv = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                      dtype={"stem": str})
    csv["stem"] = csv["stem"].str.zfill(6)
    ratio = csv["hits_no_convert"] / csv["hits_transport"].clip(lower=1)
    pick = csv[csv["hits_no_convert"] >= 5].assign(ratio=ratio)
    # 船体足够大的场景优先（平均 GT 面积 >= 0.2% 图幅，框不压船）
    pick["area"] = pick["stem"].map(_ship_area)
    big = pick[pick["area"] >= 0.002]
    picks = (big if len(big) else pick).sort_values(
        ["ratio", "area"], ascending=False).head(2)
    scenes = []
    for _, row in picks.iterrows():
        sc = _e5s1_scene(row["stem"], spec_old, spec_new)
        print(f"[validate {row['stem']}] lam_int_old {sc['lam_int_old']} vs csv "
              f"{row['lambda_int_old']} | lam_int_new {sc['lam_int_new']} vs "
              f"{row['lambda_int_new']} | hits {sc['hits']} vs csv "
              f"t={row['hits_transport']} n={row['hits_no_convert']} "
              f"r={row['hits_recal']}")
        scenes.append(sc)

    order = ["transport", "no_convert", "recal"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.5))
    for i, (sc, row_ax) in enumerate(zip(scenes, axes)):
        meta = sc["meta"]
        lb = boxes_to_letterbox(sc["boxes"], meta) if len(sc["boxes"]) else None
        iw, ih = int(meta["iw"]), int(meta["ih"])
        ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
        for j, (ax, key) in enumerate(zip(row_ax, order)):
            alarm = sc["zB"] > sc["lams"][key]
            hit = near_boxes(sc["centers"], lb, GUARD_PX) & alarm
            # letterbox 统一显示（640x640 画布）
            ax.imshow(plt.imread(SSDD_IMG / f"{sc['stem']}.jpg"), cmap="gray",
                      extent=(px, px + iw * ratio, py + ih * ratio, py))
            for b in (lb if lb is not None else []):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=False, ec=C_ACC, lw=0.5))
            fp = alarm & ~hit
            ax.scatter(sc["centers"][fp, 0], sc["centers"][fp, 1], s=12,
                       marker="x", c=C_RED, lw=0.8, zorder=3)
            ax.scatter(sc["centers"][hit, 0], sc["centers"][hit, 1], s=9,
                       c=C_ACC, zorder=4, edgecolors="none")
            h, n = sc["hits"][key], sc["n_meas"]
            sub = ("abcdef"[i * 3 + j])
            ax.set_title(f"({sub}) {sc['stem']} " +
                         {0: r"transported $\lambda_{int}$",
                          1: r"$\lambda_{int}$ reused",
                          2: "recalibrated (oracle)"}[j] +
                         "\n" + r"$P_{fa}$=" + f"{h / n:.2e} ({h}/{n})",
                         fontsize=8.5)
            ax.set_xlim(0, 640); ax.set_ylim(640, 0)
            ax.set_xticks([]); ax.set_yticks([])
    s = _json.load(open(RES / "e5_s1" / "summary.json", encoding="utf-8"))["rows"][0]
    fig.suptitle(f"E5 setting 1 (A$\\to$B, 232 scenes): pooled $P_{{fa}}$ "
                 f"transport={s['pfa_transport']:.2e}, no-convert="
                 f"{s['pfa_no_convert']:.2e}, recal={s['pfa_recal']:.2e} "
                 f"($\\alpha$={ALPHA:g})", fontsize=9, y=0.99)
    fig.tight_layout()
    save(fig, "sample_transport_mosaic")


def sample_heat_mosaic():
    """三海况 x 每海况 2 场景：overlay 单层 2x3（raw 底图经 alpha 叠加可见）。"""
    # stems[i][j]: 场景序号 i（每海况取 2 个），海况 j
    stems = [[sorted(STRESS_IMG.glob(f"{c}_*.jpg"))[k].stem
              for c in ("c1", "c2", "c3")] for k in range(2)]
    print("heat mosaic scenes:", stems)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2))
    im = None
    vmax = 0.0
    for i in range(2):
        for j in range(3):
            stem = stems[i][j]
            d, meta = load_dump(CV3 / f"{stem}.npz")
            g = grid_of(d["prob"].astype(np.float64), d["head_sizes"], 1)
            iw, ih = int(meta["iw"]), int(meta["ih"])
            ratio = float(meta["ratio"]); px, py = (float(v) for v in meta["pad"])
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 640); ax.set_ylim(640, 0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.imshow(plt.imread(STRESS_IMG / f"{stem}.jpg"), cmap="gray",
                      extent=(px, px + iw * ratio, py + ih * ratio, py))
            im = ax.imshow(g, cmap="inferno", norm=None, alpha=0.55,
                           extent=(0, 640, 640, 0), interpolation="bilinear")
            vmax = max(vmax, float(g.max()))
            ax.set_title(f"({'abcdef'[i * 3 + j]}) {stem} "
                         f"(sea state {j + 1})", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"scene {i + 1}", fontsize=8)
    # 统一色标（全局 vmax + 幂律归一）
    for ax in axes.ravel():
        ax.images[-1].set_norm(PowerNorm(0.3, vmin=0.0, vmax=vmax))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012)
    cb.set_label(r"score $s(u)$, head 1 (power norm, $\gamma$=0.3)", fontsize=8)
    save(fig, "sample_heat_mosaic")


# ------------------------------------------------- 近岸/远洋 + 说服力折线图

def _landfrac():
    """每张 test 图的亮像素占比（>0.55），作近岸/远洋判据；缺则现算。"""
    import numpy as np
    from PIL import Image
    cache = CODE / "experiments" / "_landfrac.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows = {}
    for p in sorted(SSDD_IMG.glob("*.jpg")):
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
        rows[p.stem] = float((a > 0.55).mean())
    cache.write_text(json.dumps(rows))
    return rows


def sample_inshore_mosaic():
    """上行近岸 3 场景 / 下行远洋 3 场景（识别框 + 告警点，prob 域逐场景 λ）。"""
    lf = _landfrac()
    counts = {}
    for f in sorted(DUMPS.glob("*.npz")):
        _, m = load_dump(f)
        counts[f.stem] = len(m.get("boxes", []))

    def pick(cand, nmin):
        out = []
        for s in cand:
            if counts.get(s, 0) >= nmin:
                out.append(s)
            if len(out) == 3:
                break
        return out

    by_land = sorted(lf, key=lambda s: -lf[s])
    inshore = pick(by_land, 2) or pick(by_land, 1)
    offshore = pick(list(reversed(by_land)), 2) or pick(list(reversed(by_land)), 1)
    print("inshore:", [(s, counts[s], round(lf[s], 3)) for s in inshore])
    print("offshore:", [(s, counts[s], round(lf[s], 3)) for s in offshore])

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.5))
    for row, (grp, stems) in enumerate((("inshore", inshore),
                                         ("offshore", offshore))):
        for col, stem in enumerate(stems):
            _plot_det_panel(axes[row, col], "abcdef"[row * 3 + col],
                            stem, grp)
    axes[0, 0].scatter([], [], s=9, c=C_ACC, label="alarm on ship")
    axes[0, 0].scatter([], [], s=12, marker="x", c=C_RED, label="false alarm")
    axes[0, 0].plot([], [], c=C_ACC, lw=0.7, label="GT box")
    axes[0, 0].legend(loc="lower right", framealpha=0.85, fontsize=7.5)
    fig.suptitle(r"INT8 OM, per-scene $\lambda$ at $\alpha=10^{-3}$: "
                 "inshore (top) vs. offshore (bottom) SSDD scenes", fontsize=9)
    fig.tight_layout()
    save(fig, "sample_inshore_mosaic")


def _sweep_pfa_vs_alpha(alphas):
    """逐场景 λ 标定 + 池化 P_fa（E5S1 A 域 z 口径，σ 不量化）。"""
    tot_hits = {a: 0 for a in alphas}
    tot_meas = 0
    for f in sorted(DUMPS.glob("*.npz")):
        d, meta = load_dump(f)
        imgsz = int(meta.get("imgsz", 640))
        centers = cell_centers(d["head_sizes"], imgsz)
        boxes = np.asarray(meta.get("boxes", []), dtype=float).reshape(-1, 4)
        lb = boxes_to_letterbox(boxes, meta) if len(boxes) else None
        mc, mm, _ = build_omega(centers, lb, guard_px=GUARD_PX,
                                block_px=BLOCK_PX, keep=KEEP)
        s = _logit_prob(d["prob"])
        mu, sig = _mu_sigma_blocks(s, centers, BLOCK_PX)
        z = (s - mu) / sig
        for a in alphas:
            lam = calibrate_lambda(z, mc, a)
            tot_hits[a] += int((z[mm] > lam).sum())
        tot_meas += int(mm.sum())
    return np.array([tot_hits[a] / tot_meas for a in alphas]), tot_meas


def sample_calib_alpha():
    """P_fa vs alpha 校准曲线：契约在 4 个数量级的 alpha 上均成立。"""
    alphas = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    pfa, n = _sweep_pfa_vs_alpha(alphas)
    print("alpha sweep:", list(zip(alphas.tolist(), pfa.tolist())), "n_meas", n)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.loglog(alphas, alphas, ls="--", c=C_GRAY, lw=1, label=r"$P_{fa}=\alpha$")
    ax.loglog(alphas, pfa, "-o", c=C_INT8, ms=4.5, lw=1.5,
              label=r"INT8, per-scene $\lambda$")
    ax.axvline(ALPHA, c=C_RED, lw=0.8, ls=":", alpha=0.8)
    ax.annotate(r"$\alpha=10^{-3}$", (ALPHA * 1.2, 3e-4), fontsize=7.5,
                color=C_RED)
    for a, p in zip(alphas, pfa):
        ax.annotate(f"{p:.1e}", (a, p), textcoords="offset points",
                    xytext=(4, -10), fontsize=6.5, color=C_GRAY)
    ax.set_xlabel(r"target false-alarm rate $\alpha$")
    ax.set_ylabel(r"pooled measured $P_{fa}$")
    ax.set_title(f"Calibration holds across $\\alpha$ (232 scenes, "
                 f"$n_{{meas}}$={n:,})", fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "sample_calib_alpha")


def sample_pfa_cdf():
    """E5S1 逐场景 P_fa 的 ECDF：no_convert 大量场景失控，transport 保持契约。"""
    df = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                     dtype={"stem": str})
    df["stem"] = df["stem"].str.zfill(6)
    s = json.load(open(RES / "e5_s1" / "summary.json", encoding="utf-8"))["rows"][0]
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    styles = {"pfa_no_convert": (C_RED, r"$\lambda_{int}$ reused (no convert)"),
              "pfa_transport": (C_INT8, r"transported $\lambda_{int}$"),
              "pfa_recal": (C_GRAY, "recalibrated (oracle)")}
    pooled = {"pfa_no_convert": s["pfa_no_convert"],
              "pfa_transport": s["pfa_transport"],
              "pfa_recal": s["pfa_recal"]}
    for col, (c, lab) in styles.items():
        v = np.sort(df[col].values)
        ax.step(np.clip(v, 1e-6, None), np.arange(1, len(v) + 1) / len(v),
                where="post", c=c, lw=1.5,
                label=f"{lab}, pooled {pooled[col]:.1e}")
    ax.axvline(ALPHA, c="k", lw=0.8, ls=":")
    ax.annotate(r"$\alpha=10^{-3}$", (ALPHA * 1.15, 0.06), fontsize=7.5)
    frac = float((df["pfa_no_convert"] > 2 * ALPHA).mean())
    ax.annotate(f"{frac*100:.0f}% of scenes\nviolate 2α (reused)",
                (3e-3, 0.45), fontsize=7, color=C_RED)
    ax.set_xscale("log")
    ax.set_xlabel(r"per-scene $P_{fa}$ (hits / $n_{meas}$)")
    ax.set_ylabel("ECDF (232 scenes)")
    ax.set_title("Per-scene false-alarm control, E5 setting 1 (A→B)",
                 fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    save(fig, "sample_pfa_cdf")


def sample_inshore_pfa():
    """近岸 vs 远洋：逐场景 P_fa ECDF + 三设置池化柱状（log y）。"""
    lf = _landfrac()
    df = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                     dtype={"stem": str})
    df["stem"] = df["stem"].str.zfill(6)
    df["land"] = df["stem"].map(lf)
    df["grp"] = np.where(df["land"] >= 0.05, "inshore", "offshore")
    print("scene counts:", df["grp"].value_counts().to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    for grp, c in (("inshore", C_INT8), ("offshore", C_ACC)):
        sub = df[df["grp"] == grp]
        v = np.sort(sub["pfa_transport"].values)
        ax.step(np.clip(v, 1e-6, None), np.arange(1, len(v) + 1) / len(v),
                where="post", c=c, lw=1.5,
                label=f"{grp} (n={len(sub)}, pooled "
                      f"{sub['hits_transport'].sum() / sub['n_meas'].sum():.1e})")
    ax.axvline(ALPHA, c="k", lw=0.8, ls=":")
    ax.annotate(r"$\alpha=10^{-3}$", (ALPHA * 1.15, 0.05), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"per-scene $P_{fa}$, transported $\lambda_{int}$")
    ax.set_ylabel("ECDF")
    ax.set_title("(a) per-scene $P_{fa}$ by terrain", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)

    ax = axes[1]
    x = np.arange(3)
    w = 0.36
    for k, (grp, c) in enumerate((("inshore", C_INT8),
                                  ("offshore", C_ACC))):
        sub = df[df["grp"] == grp]
        hits = [sub["hits_transport"].sum(), sub["hits_no_convert"].sum(),
                sub["hits_recal"].sum()]
        nm = sub["n_meas"].sum()
        vals = [h / nm for h in hits]
        ax.bar(x + (k - 0.5) * w, vals, w, color=c, label=grp)
        for xi, v in zip(x + (k - 0.5) * w, vals):
            ax.annotate(f"{v:.1e}", (xi, v * 1.15), ha="center", fontsize=6.5)
    ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
    ax.annotate(r"$\alpha$", (2.42, ALPHA * 1.2), fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 2e-2)
    ax.set_xticks(x)
    ax.set_xticklabels(["transport", "no convert", "recal (oracle)"],
                       fontsize=7.5, rotation=12)
    ax.set_ylabel(r"pooled $P_{fa}$")
    ax.set_title("(b) pooled $P_{fa}$ by terrain", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Inshore vs. offshore false-alarm control (E5 setting 1)",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "sample_inshore_pfa")


def _inshore_split(min_gt=1):
    """(inshore, offshore, landfrac, counts)：按亮像素占比降/升序，GT>=min_gt。"""
    lf = _landfrac()
    counts = {}
    for f in sorted(DUMPS.glob("*.npz")):
        _, m = load_dump(f)
        counts[f.stem] = len(m.get("boxes", []))
    by_land = sorted(lf, key=lambda s: -lf[s])
    inshore = [s for s in by_land
               if lf[s] >= 0.05 and counts.get(s, 0) >= min_gt]
    offshore = [s for s in reversed(by_land)
                if lf[s] < 0.05 and counts.get(s, 0) >= min_gt]
    return inshore, offshore, lf, counts


def sample_inshore_det2():
    """近岸/远洋识别框扩展拼图（2x4）：跳过 inshore_mosaic 已用的前 3 个。"""
    inshore, offshore, lf, counts = _inshore_split(2)
    inshore = inshore[3:7] or inshore[:4]
    offshore = offshore[3:7] or offshore[:4]
    print("inshore det2:", [(s, counts[s], round(lf[s], 3)) for s in inshore])
    print("offshore det2:", [(s, counts[s], round(lf[s], 3)) for s in offshore])

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.6))
    for row, (grp, stems) in enumerate((("inshore", inshore),
                                        ("offshore", offshore))):
        for col, stem in enumerate(stems):
            _plot_det_panel(axes[row, col], "abcdefgh"[row * 4 + col],
                            stem, grp)
    axes[0, 0].scatter([], [], s=9, c=C_ACC, label="alarm on ship")
    axes[0, 0].scatter([], [], s=12, marker="x", c=C_RED, label="false alarm")
    axes[0, 0].plot([], [], c=C_ACC, lw=0.7, label="GT box")
    axes[0, 0].legend(loc="lower right", framealpha=0.85, fontsize=7.5)
    fig.suptitle(r"INT8 OM, per-scene $\lambda$ at $\alpha=10^{-3}$: "
                 "inshore (top) vs. offshore (bottom), extended scenes",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "sample_inshore_det2")


def sample_inshore_heat():
    """近岸/远洋分数场热图（2x4）：与 inshore_mosaic 同源前 4 场景，head 1 叠加。"""
    inshore, offshore, lf, counts = _inshore_split(1)
    stems = {"inshore": inshore[:4], "offshore": offshore[:4]}
    print("inshore heat:", [(s, counts[s], round(lf[s], 3)) for s in stems["inshore"]])
    print("offshore heat:", [(s, counts[s], round(lf[s], 3)) for s in stems["offshore"]])

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.2))
    im = None
    vmax = 0.0
    for row, grp in enumerate(("inshore", "offshore")):
        for col, stem in enumerate(stems[grp]):
            d, meta = load_dump(DUMPS / f"{stem}.npz")
            g = grid_of(d["prob"].astype(np.float64), d["head_sizes"], 1)
            iw, ih = int(meta["iw"]), int(meta["ih"])
            ratio = float(meta["ratio"])
            px, py = (float(v) for v in meta["pad"])
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 640); ax.set_ylim(640, 0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.imshow(plt.imread(SSDD_IMG / f"{stem}.jpg"), cmap="gray",
                      extent=(px, px + iw * ratio, py, py + ih * ratio))
            im = ax.imshow(g, cmap="inferno", alpha=0.55,
                           extent=(0, 640, 640, 0), interpolation="bilinear")
            vmax = max(vmax, float(g.max()))
            ax.set_title(f"({'abcdefgh'[row * 4 + col]}) {stem} "
                         f"(land {lf[stem]:.2f}, {counts[stem]} GT)",
                         fontsize=8)
            if col == 0:
                ax.set_ylabel(grp, fontsize=8)
    for ax in axes.ravel():
        ax.images[-1].set_norm(PowerNorm(0.3, vmin=0.0, vmax=vmax))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012)
    cb.set_label(r"score $s(u)$, head 1 (power norm, $\gamma$=0.3)",
                 fontsize=8)
    save(fig, "sample_inshore_heat")


def sample_inshore_scatter():
    """近岸/远洋逐场景 P_fa vs 陆地占比散点（transport / no_convert 双面板）。"""
    lf = _landfrac()
    df = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                     dtype={"stem": str})
    df["stem"] = df["stem"].str.zfill(6)
    df["land"] = df["stem"].map(lf)
    df["grp"] = np.where(df["land"] >= 0.05, "inshore", "offshore")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    panels = ((axes[0], "pfa_transport", r"(a) transported $\lambda_{int}$"),
              (axes[1], "pfa_no_convert", r"(b) $\lambda_{int}$ reused"))
    for ax, col, title in panels:
        for grp, c in (("inshore", C_INT8), ("offshore", C_ACC)):
            sub = df[df["grp"] == grp]
            ax.scatter(sub["land"], np.clip(sub[col], 1e-6, None), s=10,
                       alpha=0.6, c=c, edgecolors="none",
                       label=f"{grp} (n={len(sub)})")
        ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("land fraction (bright-pixel ratio)")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel(r"per-scene $P_{fa}$")
    for col in ("pfa_transport", "pfa_no_convert"):
        v = np.log10(np.clip(df[col], 1e-9, None))
        r = float(np.corrcoef(df["land"], v)[0, 1])
        print(f"corr(land, log10 {col}) = {r:+.3f}")
    fig.suptitle("Per-scene $P_{fa}$ vs. land fraction (E5 setting 1)",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "sample_inshore_scatter")


def sample_inshore_box():
    """近岸 vs 远洋逐场景 P_fa 箱线对比（3 设置 × 2 组，散点抖动叠加）。"""
    lf = _landfrac()
    df = pd.read_csv(RES / "e5_s1" / "e5_per_scene.csv", encoding="utf-8-sig",
                     dtype={"stem": str})
    df["stem"] = df["stem"].str.zfill(6)
    df["land"] = df["stem"].map(lf)
    df["grp"] = np.where(df["land"] >= 0.05, "inshore", "offshore")

    settings = (("pfa_transport", r"(a) transported $\lambda_{int}$"),
                ("pfa_no_convert", r"(b) $\lambda_{int}$ reused (no convert)"),
                ("pfa_recal", r"(c) recalibrated (oracle)"))
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=True)
    stats = {}
    for ax, col, title in zip(axes, *zip(*settings)):
        data, labels, colors = [], [], []
        for grp, c in (("inshore", C_INT8), ("offshore", C_ACC)):
            v = np.clip(df.loc[df["grp"] == grp, col].values, 1e-6, None)
            data.append(v)
            labels.append(f"{grp}\n(n={len(v)})")
            colors.append(c)
            stats[(col, grp)] = (float(np.median(v)), float(np.mean(v > 2 * ALPHA)))
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                        widths=0.55, patch_artist=True, medianprops=dict(color="k"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for i, (v, c) in enumerate(zip(data, colors)):
            ax.scatter(rng.normal(i + 1, 0.07, len(v)), v, s=6, alpha=0.45,
                       c=c, edgecolors="none", zorder=3)
        ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
        ax.axhline(2 * ALPHA, c=C_RED, lw=0.7, ls="--")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        for i, grp in enumerate(("inshore", "offshore")):
            med, viol = stats[(col, grp)]
            ax.annotate(f"med {med:.1e}\n{viol * 100:.0f}%>2α",
                        (i + 1.32, med), fontsize=6, va="center")
    axes[0].set_ylabel(r"per-scene $P_{fa}$")
    axes[2].annotate(r"$2\alpha$", (2.28, 2 * ALPHA * 1.1), fontsize=6.5,
                     color=C_RED)
    fig.suptitle("Inshore vs. offshore: per-scene $P_{fa}$ distribution "
                 "(E5 setting 1, box=IQR, jittered scenes)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for (col, grp), (med, viol) in stats.items():
        print(f"{col} {grp}: median={med:.3e} frac>2a={viol:.3f}")
    save(fig, "sample_inshore_box")


def sample_ablation_bar():
    """消融配置 vs 完整方法对比柱状汇总（E9 真实 item3/4/5 + 合成 item7）。"""
    real, synth = _e9tables()

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4))

    # (a) 块抽样 keep：P_fa 柱 + α 线 + n_eff 标注
    ax = axes[0]
    item = "3 块抽样"
    keeps = ["1", "0.25", "0.125", "0.0625"]
    pr = [_e9val(real, item, f"keep={k}", "P_fa") for k in keeps]
    nr = [_e9val(real, item, f"keep={k}", "n_eff（测量块数，均值）") for k in keeps]
    x = np.arange(4)
    ax.bar(x, pr, 0.62, color=[C_ACC, C_INT8, C_INT8, C_RED])
    for xi, v, n in zip(x, pr, nr):
        ax.annotate(f"{v:.1e}", (xi, v * 1.12), ha="center", fontsize=6.5)
        ax.annotate(f"$n_{{eff}}$={n:.0f}", (xi, 0.55e-3), ha="center",
                    fontsize=6, color="w")
    ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_ylim(0.4e-3, 9e-3)
    ax.set_xticks(x); ax.set_xticklabels(keeps, fontsize=7.5)
    ax.set_xlabel("block-sampling keep")
    ax.set_ylabel(r"$P_{fa}$ (real, 232 imgs)")
    ax.set_title("(a) block sampling", fontsize=8.5)

    # (b) 标定样本量：α 误差 mean±std 柱
    ax = axes[1]
    item = "4 标定样本量"
    ns = [16, 64, 128, 200]
    mr = np.array([_e9val(real, item, f"{n} 场景", "α 误差 |log10|（均值）")
                   for n in ns])
    sr = np.array([_e9val(real, item, f"{n} 场景", "α 误差 |log10|（std）")
                   for n in ns])
    ax.bar(ns, mr, width=np.array(ns) * 0.42, color=[C_INT8, C_ACC, C_ACC, C_RED],
           yerr=sr, capsize=2.5, error_kw=dict(lw=0.9))
    for n, m, s in zip(ns, mr, sr):
        ax.annotate(f"{m:.3f}", (n, m + s + 0.004), ha="center", fontsize=6.5)
    ax.set_xlabel("calibration scenes $n$")
    ax.set_ylabel(r"err $=|\log_{10}(P_{fa}/\alpha)|$ (dex)")
    ax.set_ylim(0, 0.24)
    ax.set_xscale("log", base=2)
    ax.set_title("(b) calibration set size", fontsize=8.5)

    # (c) 逐场景 vs 全局标定
    ax = axes[2]
    item = "5 逐场景 vs 全局"
    pv = [_e9val(real, item, s, "P_fa") for s in ("逐场景标定", "全局标定")]
    ax.bar([0, 1], pv, 0.55, color=[C_GRAY, C_ACC])
    for xi, v in zip([0, 1], pv):
        ax.annotate(f"{v:.2e}", (xi, v * 1.1), ha="center", fontsize=7)
    ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_ylim(0.5e-3, 4e-3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["per-scene\n(err 0.201)", "global\n(err 0.018)"],
                       fontsize=7.5)
    ax.set_ylabel(r"$P_{fa}$")
    ax.set_title("(c) per-scene vs. global $\\lambda$", fontsize=8.5)

    # (d) 校准集分层（合成 item7）
    ax = axes[3]
    item = "7 校准集分层"
    pf32 = _e9val(synth, item, "分层（三海况）", "P_fa FP32")
    v_strat = _e9val(synth, item, "分层（三海况）", "P_fa INT8")
    v_calm = _e9val(synth, item, "只取平静", "P_fa INT8")
    r_strat = _e9val(synth, item, "分层（三海况）", "R（压测集，INT8/FP32）")
    r_calm = _e9val(synth, item, "只取平静", "R（压测集，INT8/FP32）")
    ax.bar([0, 1], [v_strat, v_calm], 0.55, color=[C_ACC, C_RED])
    for xi, v in zip([0, 1], [v_strat, v_calm]):
        ax.annotate(f"{v:.1e}", (xi, v * 1.15), ha="center", fontsize=7)
    ax.axhline(ALPHA, c="k", lw=0.8, ls=":")
    ax.axhline(pf32, c=C_FP16, lw=0.8, ls="--")
    ax.annotate("FP32 ref", (1.28, pf32 * 0.85), fontsize=6.5, color=C_FP16)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3e-3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"stratified\n(R={r_strat:.2f})",
                        f"calm-only\n(R={r_calm:.2f})"], fontsize=7.5)
    ax.set_ylabel(r"$P_{fa}$ INT8 (synthetic)")
    ax.set_title("(d) calibration stratification", fontsize=8.5)

    fig.suptitle(r"Ablation overview: each design choice vs. the full method "
                 r"($\alpha=10^{-3}$, red = degraded)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "sample_ablation_bar")


def sample_sota_bar():
    """E8 对照/SOTA 对比：P_d、实测 P_fa vs α、时延三面板柱状。"""
    df = pd.read_csv(RES / "e8" / "table6_e8.csv", encoding="utf-8-sig")
    # 方法行（QCert 完整与 INT8 同数值，只画一次）
    rows = [("CA-CFAR\n(pixel)", "CA-CFAR（像素域）", C_GRAY),
            ("YOLO11n +\nglobal thr.", "YOLO11n + 全局阈值", C_FP16),
            ("QCert\n(FP16)", "YOLO11n + QCert(FP16)", C_INT8),
            ("QCert\n(INT8, ours)", "YOLO11n + QCert(INT8)", C_ACC)]
    labels = [r[0] for r in rows]
    pd1 = [float(df.loc[df["method"] == r[1], "pd_at_1e-3"].iloc[0]) for r in rows]
    pd2 = [float(df.loc[df["method"] == r[1], "pd_at_1e-4"].iloc[0]) for r in rows]
    pfa = [float(df.loc[df["method"] == r[1], "pfa"].iloc[0]) for r in rows]
    lat = [df.loc[df["method"] == r[1], "latency_ms"].iloc[0] for r in rows]
    colors = [r[2] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    x = np.arange(4)
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, pd1, w, color=colors, label=r"$P_d$@$\alpha=10^{-3}$")
    ax.bar(x + w / 2, pd2, w, color=colors, alpha=0.45,
           label=r"$P_d$@$\alpha=10^{-4}$")
    for xi, v in zip(x - w / 2, pd1):
        ax.annotate(f"{v:.3f}", (xi, v + 0.015), ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$P_d$"); ax.set_ylim(0, 1.15)
    ax.set_title("(a) detection probability", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    ax.bar(x, pfa, 0.55, color=colors)
    for xi, v in zip(x, pfa):
        ax.annotate(f"{v:.1e}", (xi, v * 1.15), ha="center", fontsize=7.5)
    ax.axhline(ALPHA, c="k", lw=0.9, ls=":")
    ax.axhspan(ALPHA, 2 * ALPHA, color=C_RED, alpha=0.08)
    ax.annotate(r"$\alpha$", (3.42, ALPHA * 1.1), fontsize=8)
    ax.annotate(r"contract band $[\alpha,2\alpha]$", (-0.05, 3.6e-3),
                fontsize=7, color=C_RED)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 6e-3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"measured $P_{fa}$ (nominal $10^{-3}$)")
    ax.set_title("(b) false-alarm contract", fontsize=9)

    ax = axes[2]
    lat_v = [float(v) for v in lat]
    ax.bar(x, lat_v, 0.55, color=colors)
    for xi, v in zip(x, lat_v):
        if not np.isnan(v):
            ax.annotate(f"{v:.1f}", (xi, v + 0.8), ha="center", fontsize=8)
    ax.annotate("1.31× speed-up", xy=(3, 34.68 + 2.6), xytext=(0.9, 51.5),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("latency per frame (ms)")
    ax.set_ylim(0, 58)
    ax.set_title("(c) on-board latency", fontsize=9)

    fig.suptitle("Method comparison on SSDD test (232 imgs, 546 ships, "
                 "Atlas 200I DK A2)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, "sample_sota_bar")


if __name__ == "__main__":
    args = sys.argv[1:]
    todo = {
        "det": sample_det, "heat": sample_heat, "scatter": sample_scatter,
        "transport": sample_transport, "omega": sample_omega, "ccdf": sample_ccdf,
        "intcode": sample_intcode, "ablation_h0": sample_ablation_h0,
        "ablation_keep": sample_ablation_keep, "ablation_ncal": sample_ablation_ncal,
        "ablation_scale": sample_ablation_scale,
        "det_mosaic": sample_det_mosaic,
        "transport_mosaic": sample_transport_mosaic,
        "heat_mosaic": sample_heat_mosaic,
        "inshore_mosaic": sample_inshore_mosaic,
        "inshore_det2": sample_inshore_det2,
        "inshore_heat": sample_inshore_heat,
        "inshore_scatter": sample_inshore_scatter,
        "calib_alpha": sample_calib_alpha,
        "pfa_cdf": sample_pfa_cdf,
        "inshore_pfa": sample_inshore_pfa,
        "inshore_box": sample_inshore_box,
        "ablation_bar": sample_ablation_bar,
        "sota_bar": sample_sota_bar,
    }
    for name in (args or todo):
        todo[name]()
