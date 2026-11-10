"""E9 · 消融与失败分析（paper/04 §12 的七项 + 第 8 项 floor/round 舍入）。

八项消融，每项都给出"对照组 → 指标口径"，离线在合成杂波场上跑通，
拿到板级 dump 后只需把 ``_scenes`` 换成真实分数场即可复用同一套统计。

    1 除法消除         交叉相乘 vs 先把 z 量化再比较        Δlog10 P_fa、不一致率
    2 直方图桶数 B     256 / 1024 / 4096                    α 误差、λ 误差、标定耗时
    3 块抽样 keep      1 / 1/4 / 1/8 / 1/16                 α 误差、n_eff
    4 标定样本量       16 / 64 / 256 场景                   α 误差均值与 std
    5 逐场景 vs 全局   主张的单点故障                        α 误差、P_d
    6 有 / 无 h0       z=(s-μ)/σ vs 直接对 s 取阈值          R、P_d@1e-3
    7 校准集分层       分层(三海况) vs 只取平静             sat、R（压测集）
    8 floor vs round   σ 量化舍入方式                       Δln P_fa ∝ ε_σ^p

第 8 项不在 §12 原表内，是针对 E4 实测 ``|Δln P_fa| ∝ ε_σ^2`` 的直接推论：
对称舍入（round）的一阶项相消；一旦改成有偏舍入（floor），一阶项应浮现
（p→1），从而把 Prop.2 的 ``z²ε_σ`` 精确定位为"有偏舍入下的上界"。

**统计口径**：α=1e-3 时 P_fa 是尾部计数，单场景只有个位数命中。因此除第 4 项
（本身研究样本量方差）外，所有 P_fa/P_d 都**跨场景汇总计数**（Σhits/Σcells），
而不是"逐场景比值的均值"——后者会被 0 命中与 outlier 支配。

注意 ``make_scene`` 返回的 ``z`` 是**生成杂波时的标准正态**，目标单元的 ``s``
被抬高后 ``z`` 并未同步改写；因此本文件一律用 ``_z(sc) = (s-μ)/σ`` 重建分数场。

    python -m experiments.run_e9
    python -m experiments.run_e9 --n 48 --keep 0.5 --calib-scenes 320
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from qcert.decision import IntQuantSpec, decide
from qcert.pfa_eval import (build_omega, calibrate_lambda, n_eff)
from qcert.qint import (hist_quantile_int, quant_params_from_range, quantize,
                        saturation_rate)

from .common import RESULTS, ensure_dir, json_dump, save_table
from .synth import make_scene

ALPHA = 1e-3
SEAS = ("calm", "moderate", "rough")
SEA_SIGMA0 = {"calm": 0.20, "moderate": 0.30, "rough": 0.45}
ITEM1 = "1 除法消除"
ITEM2 = "2 直方图桶数"
ITEM3 = "3 块抽样"
ITEM4 = "4 标定样本量"
ITEM5 = "5 逐场景 vs 全局"
ITEM6 = "6 有/无 h0"
ITEM7 = "7 校准集分层"
ITEM8 = "8 floor vs round"


# ------------------------------------------------------------------ 公共
def _scenes(n, seed0=1000, seas=SEAS, n_target=3, contrast=1.0,
            sigma_ln=0.45, target_snr=2.5):
    """按海况轮转生成场景；``sigma0`` 随海况变，否则量化误差没有变化源。"""
    out = []
    for i in range(int(n)):
        sea = seas[i % len(seas)]
        out.append(make_scene(
            seed=seed0 + i, n_target=n_target,
            sigma0=SEA_SIGMA0.get(sea, 0.30), sigma_ln=sigma_ln,
            sea_state=sea, contrast=contrast, target_snr=target_snr))
    return out


def _z(sc):
    """重建分数场 (s−μ)/σ（目标单元也被正确抬高）。"""
    s = np.asarray(sc["s"], dtype=np.float64)
    mu = np.asarray(sc["mu"], dtype=np.float64)
    sig = np.maximum(np.asarray(sc["sigma"], dtype=np.float64), 1e-9)
    return (s - mu) / sig


def _omega(sc, seed=0, keep=0.5, guard=8.0, block=64.0):
    return build_omega(sc["centers"], sc["boxes"], guard_px=guard,
                       block_px=block, keep=keep, seed=seed)


def _row(item, setting, metric, value, ref=float("nan"), note=""):
    v, r = float(value), float(ref)
    return {"item": item, "setting": setting, "metric": metric,
            "value": v, "ref": r,
            "delta": (v - r if np.isfinite(v) and np.isfinite(r) else float("nan")),
            "note": note}


def _qrange(scenes, key, bits=8):
    """AMCT 语义：只在标定场景上定一次量化范围。"""
    lo = min(float(np.min(s[key])) for s in scenes)
    hi = max(float(np.max(s[key])) for s in scenes)
    return quant_params_from_range(lo, hi, bits=bits, symmetric=True)


def _log10_dev(pfa, alpha):
    return float(abs(np.log10(pfa / alpha))) if pfa > 0 else float("inf")


def _fmean(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _ratio(a, b):
    return float(a / b) if b > 0 else (float("inf") if a > 0 else 1.0)


# ------------------------------------------------------------------ 1 除法消除
def _item1(scenes, alpha, seed=0, keep=0.5):
    cal = scenes[:max(1, len(scenes) // 3)]
    ds, _ = _qrange(cal, "s")
    dmu, _ = _qrange(cal, "mu")
    dsig, _ = _qrange(cal, "sigma")
    dz, _ = _qrange(cal, "z")
    spec = IntQuantSpec(ds, dmu, dsig, sigma_min_q=1, shift_bits=20)

    hits = {"fp": 0, "div": 0, "zq": 0}
    dis = {"div": 0, "zq": 0}
    n = 0
    for sc in scenes:
        mc, mm, _ = _omega(sc, seed=seed, keep=keep)
        z = _z(sc)
        lam = calibrate_lambda(z, mc, alpha)
        ref = z > lam

        pred = decide(quantize(sc["s"], ds), quantize(sc["mu"], dmu),
                      quantize(sc["sigma"], dsig), lam, spec)
        pred_zq = quantize(z, dz) > int(np.round(lam / dz))

        k = int(mm.sum())
        n += k
        hits["fp"] += int(ref[mm].sum())
        hits["div"] += int(pred[mm].sum())
        hits["zq"] += int(pred_zq[mm].sum())
        dis["div"] += int((pred != ref).sum())
        dis["zq"] += int((pred_zq != ref).sum())

    pfa = {k: _ratio(v, n) for k, v in hits.items()}
    dev_div = abs(np.log10(pfa["div"] / pfa["fp"])) if pfa["fp"] > 0 and pfa["div"] > 0 else float("nan")
    dev_zq = abs(np.log10(pfa["zq"] / pfa["fp"])) if pfa["fp"] > 0 and pfa["zq"] > 0 else float("nan")

    rows = [
        _row(ITEM1, "FP32 参照", "P_fa", pfa["fp"]),
        _row(ITEM1, "交叉相乘（整数，三路 scale）", "P_fa", pfa["div"], ref=pfa["fp"],
             note="C2：无除法、无浮点"),
        _row(ITEM1, "先量化 z 再比较", "P_fa", pfa["zq"], ref=pfa["fp"],
             note="仅作参考：整数域需一次除法，NPU 不可实现"),
        _row(ITEM1, "交叉相乘（整数，三路 scale）", "判决不一致率", dis["div"] / n),
        _row(ITEM1, "先量化 z 再比较", "判决不一致率", dis["zq"] / n),
    ]
    concl = {"pfa_fp": pfa["fp"], "pfa_div": pfa["div"], "pfa_zq": pfa["zq"],
             "divfree_dev": dev_div, "zquant_dev": dev_zq,
             "divfree_ok": bool(np.isfinite(dev_div) and dev_div <= 0.05),
             "disagree_div": dis["div"] / n, "disagree_zq": dis["zq"] / n}
    return rows, concl


# ------------------------------------------------------------------ 2 直方图桶数
def _item2(scenes, alpha, seed=0, keep=0.5, bins_list=(256, 1024, 4096), k=16):
    """直方图 vs 精确排序的**标定精度**。

    单场景只有 ~4k 个格子，α=1e-3 时尾部仅 4 个样本，桶宽与样本间距同量级，
    分辨率被抽样噪声淹没。因此本项把 α 放大到 ≥1e-2（尾部 ~40 样本），
    只考察"O(B) 的整数分位数能否替代排序"，与 E1/E4 的 1e-3 口径无关。
    """
    a = max(float(alpha), 1e-2)
    cal = scenes[:max(1, len(scenes) // 3)]
    ds, _ = _qrange(cal, "s")
    dmu, _ = _qrange(cal, "mu")
    dsig, _ = _qrange(cal, "sigma")
    to_z = ds / dsig if dsig > 0 else 1.0        # ratio/2^k → z 的换算系数
    acc = {b: {"hits": 0, "lam": [], "t": []} for b in bins_list}
    hx, t_x, n = 0, [], 0

    for sc in scenes:
        mc, mm, _ = _omega(sc, seed=seed, keep=keep)
        sq = quantize(sc["s"], ds).astype(np.int64)
        muq = quantize(sc["mu"], dmu).astype(np.int64)
        sigq = np.maximum(quantize(sc["sigma"], dsig).astype(np.int64), 1)
        ratio = np.floor_divide((sq - muq) << k, sigq)

        t0 = time.perf_counter()
        lam0 = float(np.quantile(ratio[mc].astype(np.float64), 1.0 - a))
        t_x.append((time.perf_counter() - t0) * 1e3)
        hx += int((ratio[mm] > lam0).sum())
        n += int(mm.sum())

        vals = ratio[mc]
        for b in bins_list:
            t0 = time.perf_counter()
            lam = int(hist_quantile_int(vals, 1.0 - a, bins=b))
            dt = (time.perf_counter() - t0) * 1e3
            acc[b]["hits"] += int((ratio[mm] > lam).sum())
            acc[b]["lam"].append((lam - lam0) / float(2 ** k) * to_z)  # z 单位
            acc[b]["t"].append(dt)

    pfa_x = _ratio(hx, n)
    rows = [_row(ITEM2, "精确排序（np.quantile）", "P_fa", pfa_x,
                 note=f"α={a:g}（本项放大α以分辨桶宽）"),
            _row(ITEM2, "精确排序（np.quantile）", "标定耗时 ms", _fmean(t_x))]
    concl = {"exact_pfa": pfa_x, "exact_err": _log10_dev(pfa_x, a), "alpha_used": a}
    for b in bins_list:
        pfa = _ratio(acc[b]["hits"], n)
        err = _log10_dev(pfa, a)
        lam_err = np.asarray(acc[b]["lam"], dtype=np.float64)
        rows.append(_row(ITEM2, f"直方图 B={b}", "P_fa", pfa, ref=pfa_x))
        rows.append(_row(ITEM2, f"直方图 B={b}", "α 误差 |log10(P_fa/α)|",
                         err, ref=concl["exact_err"],
                         note="含测量集抽样方差，非纯桶宽误差"))
        rows.append(_row(ITEM2, f"直方图 B={b}", "λ 误差 |z|（均值）",
                         float(np.abs(lam_err).mean())))
        rows.append(_row(ITEM2, f"直方图 B={b}", "标定耗时 ms",
                         _fmean(acc[b]["t"]), ref=_fmean(t_x)))
        concl[f"B{b}_err"] = err
        concl[f"B{b}_lam_err"] = float(np.abs(lam_err).mean())
        concl[f"B{b}_pfa"] = pfa
    concl["B4096_ok"] = bool(np.isfinite(concl["B4096_err"])
                             and abs(concl["B4096_err"] - concl["exact_err"]) <= 0.05)
    return rows, concl


# ------------------------------------------------------------------ 3 块抽样
def _item3(scenes, alpha, seed=0, keeps=(1.0, 0.25, 0.125, 0.0625)):
    rows, concl = [], {}
    for keep in keeps:
        hits, n, ne = 0, 0, []
        for sc in scenes:
            mc, mm, _ = _omega(sc, seed=seed, keep=keep)
            z = _z(sc)
            lam = calibrate_lambda(z, mc, alpha)
            hits += int((z[mm] > lam).sum())
            n += int(mm.sum())
            ne.append(n_eff(mm, sc["centers"], 64.0))
        pfa = _ratio(hits, n)
        err = _log10_dev(pfa, alpha)
        rows.append(_row(ITEM3, f"keep={keep:g}", "P_fa", pfa))
        rows.append(_row(ITEM3, f"keep={keep:g}", "α 误差 |log10(P_fa/α)|", err,
                         note=("keep=1 表示全部块，即无抽样误差" if keep >= 1.0 else "")))
        rows.append(_row(ITEM3, f"keep={keep:g}", "n_eff（测量块数，均值）", _fmean(ne)))
        concl[f"keep_{keep:g}"] = {"pfa": pfa, "err": err, "n_eff": _fmean(ne)}
    return rows, concl


# ------------------------------------------------------------------ 4 标定样本量
def _item4(pool, alpha, sizes=(16, 64, 256), reps=10, seed=0, keep=0.5):
    cal_vals, meas_vals = [], []
    for sc in pool:
        mc, mm, _ = _omega(sc, seed=seed, keep=keep)
        cal_vals.append(_z(sc)[mc])
        meas_vals.append(_z(sc)[mm])
    rng = np.random.default_rng(seed)
    rows, concl = [], {}
    for n in sizes:
        if n >= len(pool):
            continue
        devs = []
        for _ in range(int(reps)):
            idx = rng.choice(len(pool), size=int(n), replace=False)
            zc = np.concatenate([cal_vals[i] for i in idx])
            lam = float(np.quantile(zc, 1.0 - alpha))
            keep = np.ones(len(pool), dtype=bool)
            keep[idx] = False
            zm = np.concatenate([meas_vals[i] for i in np.flatnonzero(keep)])
            devs.append(_log10_dev(float(np.mean(zm > lam)), alpha))
        rows.append(_row(ITEM4, f"{n} 场景", "α 误差 |log10|（均值）",
                         _fmean(devs), note=f"{reps} 次重采样"))
        rows.append(_row(ITEM4, f"{n} 场景", "α 误差 |log10|（std）",
                         float(np.std(devs, ddof=1)) if len(devs) > 1 else float("nan")))
        concl[f"n{n}"] = {"mean": _fmean(devs),
                          "std": float(np.std(devs, ddof=1)) if len(devs) > 1 else float("nan")}
    return rows, concl


# ------------------------------------------------------------------ 5 逐场景 vs 全局
def _item5(scenes, alpha, seed=0, keep=0.5):
    masks, cal_all = [], []
    for sc in scenes:
        mc, mm, _ = _omega(sc, seed=seed, keep=keep)
        masks.append((mc, mm))
        cal_all.append(_z(sc)[mc])
    lam_g = float(np.quantile(np.concatenate(cal_all), 1.0 - alpha))

    hits_s = hits_g = 0
    n = 0
    hit_t_s = hit_t_g = 0
    n_t = 0
    for sc, (mc, mm) in zip(scenes, masks):
        z = _z(sc)
        lam_s = calibrate_lambda(z, mc, alpha)
        hits_s += int((z[mm] > lam_s).sum())
        hits_g += int((z[mm] > lam_g).sum())
        n += int(mm.sum())
        ti = sc["target_idx"]
        if ti.any():
            hit_t_s += int((z[ti] > lam_s).sum())
            hit_t_g += int((z[ti] > lam_g).sum())
            n_t += int(ti.sum())

    pfa_s, pfa_g = _ratio(hits_s, n), _ratio(hits_g, n)
    pd_s, pd_g = _ratio(hit_t_s, n_t), _ratio(hit_t_g, n_t)
    err_s, err_g = _log10_dev(pfa_s, alpha), _log10_dev(pfa_g, alpha)
    # §12 必答项：若逐场景并没有明显好于全局，就把主张从"逐场景标定"降为"零分布标定"
    keep_per_scene = bool((abs(pd_s - pd_g) >= 0.02) or (err_s <= 0.5 * err_g))
    rows = [
        _row(ITEM5, "逐场景标定", "P_fa", pfa_s, note=f"α 误差 {err_s:.3f}"),
        _row(ITEM5, "全局标定", "P_fa", pfa_g, note=f"α 误差 {err_g:.3f}"),
        _row(ITEM5, "逐场景标定", "P_d@1e-3（单元级，汇总）", pd_s),
        _row(ITEM5, "全局标定", "P_d@1e-3（单元级，汇总）", pd_g),
        _row(ITEM5, "逐场景 − 全局", "ΔP_d", pd_s - pd_g,
             note=("保留'逐场景标定'（ΔP_d ≥ 2 点或 α 误差减半）" if keep_per_scene
                   else "差异很小 → 主张改为'零分布标定'（§12 必答项）")),
    ]
    concl = {"pfa_per_scene": pfa_s, "pfa_global": pfa_g,
             "pd_per_scene": pd_s, "pd_global": pd_g, "d_pd": pd_s - pd_g,
             "err_per_scene": err_s, "err_global": err_g,
             "keep_per_scene": keep_per_scene}
    return rows, concl


# ------------------------------------------------------------------ 6 有/无 h0
def _item6(scenes, alpha, seed=0, keep=0.5):
    cal = scenes[:max(1, len(scenes) // 3)]
    ds, _ = _qrange(cal, "s")
    dmu, _ = _qrange(cal, "mu")
    dsig, _ = _qrange(cal, "sigma")
    spec = IntQuantSpec(ds, dmu, dsig, sigma_min_q=1, shift_bits=20)

    hz, ns, hs, nt, hzt, hst = 0, 0, 0, 0, 0, 0
    pfz = pfs = 0
    for sc in scenes:
        mc, mm, _ = _omega(sc, seed=seed, keep=keep)
        z, s = _z(sc), np.asarray(sc["s"], dtype=np.float64)
        ti = sc["target_idx"]

        lam = calibrate_lambda(z, mc, alpha)
        pred = decide(quantize(s, ds), quantize(sc["mu"], dmu),
                      quantize(sc["sigma"], dsig), lam, spec)
        pfz += int((z[mm] > lam).sum())
        ns += int(mm.sum())
        hz += int(pred[mm].sum())

        lam_s = calibrate_lambda(s, mc, alpha)
        pred_s = quantize(s, ds) > int(np.round(lam_s / ds))
        pfs += int((s[mm] > lam_s).sum())
        hs += int(pred_s[mm].sum())

        if ti.any():
            nt += int(ti.sum())
            hzt += int(pred[ti].sum())
            hst += int(pred_s[ti].sum())

    rows = [
        _row(ITEM6, "z = (s−μ)/σ", "R（INT8/FP32，汇总）", _ratio(hz, pfz)),
        _row(ITEM6, "s（无 h0，直接取阈值）", "R（INT8/FP32，汇总）", _ratio(hs, pfs)),
        _row(ITEM6, "z = (s−μ)/σ", "P_d@1e-3（INT8，汇总）", _ratio(hzt, nt)),
        _row(ITEM6, "s（无 h0，直接取阈值）", "P_d@1e-3（INT8，汇总）", _ratio(hst, nt)),
    ]
    concl = {"R_z": _ratio(hz, pfz), "R_s": _ratio(hs, pfs),
             "pd_z": _ratio(hzt, nt), "pd_s": _ratio(hst, nt),
             "pfa_fp_z": _ratio(pfz, ns), "pfa_fp_s": _ratio(pfs, ns)}
    return rows, concl


# ------------------------------------------------------------------ 7 校准集分层
def _item7(alpha, seed=0, n_strat=40, n_calm=40, n_stress=24, keep=0.5):
    strat = _scenes(n_strat, seed0=4000, n_target=0)
    calm = _scenes(n_calm, seed0=5000, seas=("calm",), n_target=0)
    stress = _scenes(n_stress, seed0=6000, seas=("rough",), n_target=0,
                     contrast=3.0)
    rows, concl = [], {}
    for name, cal in (("分层（三海况）", strat), ("只取平静", calm)):
        ds, _ = _qrange(cal, "s")
        dmu, _ = _qrange(cal, "mu")
        dsig, _ = _qrange(cal, "sigma")
        spec = IntQuantSpec(ds, dmu, dsig, sigma_min_q=1, shift_bits=20)
        sat, hi, hf, n = [], 0, 0, 0
        for sc in stress:
            sq = quantize(sc["s"], ds)
            sat.append(saturation_rate(sq))
            mc, mm, _ = _omega(sc, seed=seed, keep=keep)
            z = _z(sc)
            lam = calibrate_lambda(z, mc, alpha)
            hf += int((z[mm] > lam).sum())
            pred = decide(sq, quantize(sc["mu"], dmu),
                          quantize(sc["sigma"], dsig), lam, spec)
            hi += int(pred[mm].sum())
            n += int(mm.sum())
        pfa_fp, pfa_i = _ratio(hf, n), _ratio(hi, n)
        rows.append(_row(ITEM7, name, "s 上饱和率（压测集均值）", _fmean(sat)))
        rows.append(_row(ITEM7, name, "P_fa FP32", pfa_fp))
        rows.append(_row(ITEM7, name, "P_fa INT8", pfa_i))
        rows.append(_row(ITEM7, name, "R（压测集，INT8/FP32）", _ratio(pfa_i, pfa_fp)))
        concl[name] = {"sat": _fmean(sat), "pfa_fp": pfa_fp,
                       "pfa_int": pfa_i, "R": _ratio(pfa_i, pfa_fp)}
    return rows, concl


# ------------------------------------------------------------------ 8 floor vs round
def _item8(scenes, alpha, seed=0, keep=0.5, eps_list=(0.02, 0.05, 0.10, 0.20)):
    """只量化 σ 一路，隔离"分母机制"。

    s、μ 保持精确，否则它们各自的对称舍入误差会给 Δln P_fa 加一个与 ε_σ 无关的
    常数底，把小 ε_σ 端的一阶/二阶斜率压平（v1 就是这样把 round≈2 拟成 1.0 的）。
    """
    ref_sigma = float(np.median(np.concatenate([s["sigma"] for s in scenes])))
    rows, fits = [], {"round": [], "floor": []}
    lam_z = []

    for eps in eps_list:
        step = 2.0 * float(eps) * ref_sigma
        for mode in ("round", "floor"):
            hi = hf = n = 0
            for sc in scenes:
                mc, mm, _ = _omega(sc, seed=seed, keep=keep)
                z = _z(sc)
                lam = calibrate_lambda(z, mc, alpha)
                if mode == "round" and eps == eps_list[0]:
                    lam_z.append(float(lam))
                sig = np.asarray(sc["sigma"], dtype=np.float64)
                sigq = ((np.round(sig / step) if mode == "round"
                         else np.floor(sig / step)) * step)
                zq = (np.asarray(sc["s"], dtype=np.float64)
                      - np.asarray(sc["mu"], dtype=np.float64)) / np.maximum(sigq, 1e-6)
                hf += int((z[mm] > lam).sum())
                hi += int((zq[mm] > lam).sum())
                n += int(mm.sum())
            pfa_fp, pfa_q = _ratio(hf, n), _ratio(hi, n)
            d = (float(np.log(pfa_q / pfa_fp))
                 if pfa_q > 0 and pfa_fp > 0 else float("nan"))
            rows.append(_row(ITEM8, f"{mode} @ ε_σ={eps:.2f}",
                             "Δln P_fa（汇总）", d,
                             note=("对称舍入：一阶项相消" if mode == "round"
                                   else "有偏舍入：一阶项 z²ε_σ 应浮现")))
            fits[mode].append((float(eps), abs(d)))

    # Prop.2 的一阶预言：有偏舍入下 Δln P_fa ≈ z_λ² ε_σ，故该比值在 ε→0 时应 →1
    zl2 = float(np.median(lam_z)) ** 2 if lam_z else float("nan")
    for eps, v in fits["floor"]:
        rows.append(_row(ITEM8, f"floor @ ε_σ={eps:.2f}",
                         "Δln / (z_λ² ε_σ)", v / (zl2 * eps) if zl2 > 0 else float("nan"),
                         ref=1.0, note=f"z_λ²={zl2:.2f}；Prop.2 一阶预言在 ε→0 取 1"))

    expo = {}
    for mode, pts in fits.items():
        pts = [(e, v) for e, v in pts if np.isfinite(v) and v >= 1e-4]
        if len(pts) >= 2:
            x, y = np.asarray(pts).T
            expo[mode] = float(np.polyfit(np.log(x), np.log(y), 1)[0])
            # 局部指数：看清楚 ε→0 的渐近斜率（round→2 / floor→1）
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                rows.append(_row(
                    ITEM8, f"{mode} 局部 {x0:.2f}→{x1:.2f}", "指数 p", 
                    float(np.log(y1 / y0) / np.log(x1 / x0)),
                    note="渐近值 round=2（二阶）、floor=1（一阶）"))
    return rows, {"exponent": expo, "z_lambda_sq": zl2,
                  "points": {k: [list(p) for p in v] for k, v in fits.items()}}


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=48, help="各子项的场景数")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.5,
                    help="默认块抽样比例（n 小的时候要放大，否则 λ 抖动过大）")
    ap.add_argument("--calib-scenes", type=int, default=320,
                    help="第 4 项的场景池大小（须 ≥ 256）")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=str(RESULTS / "e9"))
    args = ap.parse_args(argv)

    scenes = _scenes(args.n, seed0=1000)
    pool = _scenes(max(args.calib_scenes, 64), seed0=7000, n_target=0)

    rows, concl = [], {}

    def add(key, fn, *a, **kw):
        r, c = fn(*a, **kw)
        rows.extend(r)
        concl[key] = c

    add("item1", _item1, scenes, args.alpha, args.seed, keep=args.keep)
    add("item2", _item2, scenes, args.alpha, args.seed, keep=args.keep)
    add("item3", _item3, scenes, args.alpha, args.seed)
    add("item4", _item4, pool, args.alpha,
        sizes=(16, 64, 256), reps=args.reps, seed=args.seed, keep=args.keep)
    add("item5", _item5, scenes, args.alpha, args.seed, keep=args.keep)
    add("item6", _item6, scenes, args.alpha, args.seed, keep=args.keep)
    add("item7", _item7, args.alpha, args.seed, keep=args.keep)
    add("item8", _item8, scenes, args.alpha, args.seed, keep=args.keep)

    out = ensure_dir(args.out)
    save_table(rows, out, "table3_e9", "Table 3 下 · E9 消融与失败分析")
    json_dump({"alpha": args.alpha, "n_scenes": args.n, "keep": args.keep,
               "pool": args.calib_scenes, "reps": args.reps,
               "source": "synth",
               "conclusions": concl,
               "note": "合成杂波场离线自检；板级 dump 接入后结论需重跑"},
              out / "summary.json")

    print("\n=== E9 · 消融与失败分析 ===")
    cur = None
    for r in rows:
        if r["item"] != cur:
            cur = r["item"]
            print(f"\n[{cur}]")
        ref = ("" if not np.isfinite(r["ref"]) else f"  (ref {r['ref']:.4g})")
        print(f"  {r['setting']:<28}{r['metric']:<28}{r['value']:>10.4g}{ref}"
              + (f"  {r['note']}" if r["note"] else ""))
    print("\n--- 结论 ---")
    c1 = concl.get("item1", {})
    print(f"1 除法消除: Δlog10 divfree={c1.get('divfree_dev', float('nan')):.4f}"
          f"（阈值 0.05）→ {'PASS' if c1.get('divfree_ok') else 'FAIL'}；"
          f"判决不一致率 {c1.get('disagree_div', float('nan')):.2e}"
          f" vs 先量化z {c1.get('disagree_zq', float('nan')):.2e}")
    c2 = concl.get("item2", {})
    print(f"2 B=4096 λ误差={c2.get('B4096_lam_err', float('nan')):.4f} z"
          f"（B=256 {c2.get('B256_lam_err', float('nan')):.4f}，α="
          f"{c2.get('alpha_used', float('nan')):g}）；α误差 "
          f"{c2.get('B4096_err', float('nan')):.4f} vs 精确 "
          f"{c2.get('exact_err', float('nan')):.4f}")
    c5 = concl.get("item5", {})
    print(f"5 逐场景 P_fa={c5.get('pfa_per_scene', float('nan')):.2e} vs 全局 "
          f"{c5.get('pfa_global', float('nan')):.2e}；ΔP_d="
          f"{c5.get('d_pd', float('nan')):+.4f}")
    c6 = concl.get("item6", {})
    print(f"6 R: z={c6.get('R_z', float('nan')):.3f} vs s={c6.get('R_s', float('nan')):.3f}；"
          f"P_d: z={c6.get('pd_z', float('nan')):.3f} vs s={c6.get('pd_s', float('nan')):.3f}")
    c7 = concl.get("item7", {})
    for k, v in c7.items():
        print(f"7 {k}: sat={v['sat']:.4f} R={v['R']:.3g} "
              f"(P_fa {v['pfa_fp']:.2e}→{v['pfa_int']:.2e})")
    e = concl.get("item8", {}).get("exponent", {})
    if e:
        print("8 |Δln P_fa| ∝ ε_σ^p : " +
              "  ".join(f"{k}={v:.2f}" for k, v in e.items()) +
              f"（z_λ²={concl['item8'].get('z_lambda_sq', float('nan')):.2f}；"
              "round 应≈2，floor 应≈1）")
    print(f"产物: {out}")
    return {"rows": rows, "conclusions": concl}


if __name__ == "__main__":
    main()
