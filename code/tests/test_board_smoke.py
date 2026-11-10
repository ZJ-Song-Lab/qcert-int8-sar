"""board 桥接层的 x86 自检：不依赖 pyACL，只验证通路与口径。

覆盖 paper/05 §2 的模块 1（dump_scores）、2（probe_scales）、8/9（bench/thermal）。
"""
import json

import numpy as np
import pytest

from board import bench_latency, dump_scores, runner
from qcert.scores import load_dir


# ------------------------------------------------------------------ 公共件
def test_mount_paths_and_acl_absent():
    code, deploy = runner.mount_paths()
    assert code.is_dir() and deploy.is_dir()
    assert runner.acl_available() is False          # x86 上无 pyACL


def _write_png(path, w=320, h=240, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def test_preprocess_image_is_letterbox(tmp_path):
    p = _write_png(tmp_path / "a.png", 320, 240)
    x, ratio, pad, (ih, iw) = runner.preprocess_image(p, 640)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert ratio == pytest.approx(2.0) and (ih, iw) == (240, 320)
    assert pad == (0.0, 80.0)                        # 短边方向居中填充 80 px


class _FakeInt8Model:
    """模拟 acl_infer.AclModel.infer_raw 的 INT8 分支。"""

    def __init__(self, nc=1, n_cells=256):
        self.nc = nc
        self.n_cells = n_cells

    def infer_raw(self, x):
        rng = np.random.default_rng(0)
        out = rng.integers(-128, 128, (1, 4 + self.nc, self.n_cells)).astype(np.int8)
        return [(out, out.dtype)]


def test_scores_once_int8_channel_convention():
    model = _FakeInt8Model(nc=1, n_cells=256)
    s = runner.scores_once(model, np.zeros((1, 3, 640, 640), np.float32), 1,
                           delta=0.25)
    assert s.shape == (256,)
    # 必须是 ch_first[4:5] 的整数码 × delta，范围必然落在 int8 可表示区间
    assert s.min() >= -128 * 0.25 - 1e-9
    assert s.max() <= 127 * 0.25 + 1e-9


def test_frame_fn_reports_pfa_only_when_lambda_given():
    model = _FakeInt8Model()
    x = np.zeros((1, 3, 640, 640), np.float32)
    rec = runner.frame_fn_from_model(model, [x], 1, delta=0.25)()
    assert "t_model_ms" in rec and "pfa" not in rec

    rec = runner.frame_fn_from_model(model, [x], 1, delta=0.25, lam=0.0)()
    assert 0.0 <= rec["pfa"] <= 1.0


# ------------------------------------------------------------------ 模块 1
def test_dump_scores_dry_run_writes_loadable_npz(tmp_path):
    out = tmp_path / "demo_int8"
    info = dump_scores.main(["--dry-run", "--n", "4", "--out", str(out),
                             "--impl", "int8"])
    assert info["source"] == "dry-run" and info["n"] == 4

    fields = load_dir(out)
    assert len(fields) == 4
    f = fields[sorted(fields)[0]]
    assert f.is_int8 and f.prob.ndim == 1
    assert (out / "manifest.json").exists()


def test_dump_scores_dry_run_fp32_has_no_delta(tmp_path):
    out = tmp_path / "demo_fp32"
    info = dump_scores.main(["--dry-run", "--n", "3", "--out", str(out),
                             "--impl", "fp32"])
    fields = load_dir(out)
    assert len(fields) == 3
    assert info["delta"] is None
    assert all(not f.is_int8 for f in fields.values())


# ------------------------------------------------------------------ 模块 8 / 9
def test_bench_dry_run_merges_latency_json(tmp_path):
    lat = tmp_path / "lat.json"
    info = bench_latency.main(["bench", "--dry-run", "--iters", "50",
                               "--impl", "int8", "--merge-into", str(lat)])
    assert info["n"] == 50 and info["source"] == "dry-run"
    assert info["p99_over_mean"] >= 1.0
    assert json.loads(lat.read_text(encoding="utf-8"))["int8"] == pytest.approx(
        info["mean_ms"])


def test_thermal_dry_run_is_stable(tmp_path):
    res = bench_latency.main(["thermal", "--dry-run", "--minutes", "10",
                              "--window-min", "5", "--out-dir", str(tmp_path / "e6")])
    drift = res["drift"]
    assert drift["n_windows"] == 2 and drift["n_pfa_windows"] == 2
    assert drift["pass_pfa"] and drift["pass_p99"]
    summary = json.loads((tmp_path / "e6" / "summary.json").read_text(encoding="utf-8"))
    assert summary["source"] == "dry-run" and summary["verdict"] == "PASS"


def test_thermal_dry_run_detects_drift(tmp_path):
    res = bench_latency.main(["thermal", "--dry-run", "--minutes", "10",
                              "--window-min", "5", "--dry-pfa-drift", "1.0",
                              "--out-dir", str(tmp_path / "e6")])
    assert res["drift"]["pfa_drift_log10"] > 0.15       # 漂移必须被抓到
    assert not res["drift"]["pass_pfa"]


def test_lambda_from_dump_pooled_quantile(tmp_path):
    out = tmp_path / "calib"
    dump_scores.main(["--dry-run", "--n", "4", "--out", str(out), "--impl", "fp32"])
    n_cells = sum(f.n for f in load_dir(out).values())
    lam, n = bench_latency._lambda_from_dump(out, 1e-2)
    assert n == n_cells and n > 0
    assert np.isfinite(lam)


# ------------------------------------------------------------------ 模块 2
def test_probe_scales_roundtrip(tmp_path):
    from board.probe_scales import _dry_run

    rep = _dry_run(tmp_path)
    assert rep["roundtrip_ok"] is True
    assert rep["output_entry"]["delta"] == pytest.approx(0.0078125)


def test_pick_output_by_name():
    from qcert.scale_table import ScaleEntry, ScaleTable

    t = ScaleTable()
    t.add(ScaleEntry(name="backbone.conv0", delta=0.01))
    t.add(ScaleEntry(name="model.22.output0", delta=0.02))
    from board.probe_scales import pick_output

    e = pick_output(t)
    assert e is not None and e.name == "model.22.output0"
