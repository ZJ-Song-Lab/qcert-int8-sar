"""asc-sar/deploy/amct_ptq.py 的离线可测部分：分层校准集与配置审计。

AMCT 本身只能上板/CANN 环境跑，这里只测纯函数：
- ``stratified_pick`` 按对比度三层等量抽样且可复现；
- ``float_layers`` 能识别混合精度回退（R3）；
- ``dump_config`` 把 AMCT 配置转成规范 ScaleTable，且对无 scale 的配置
  退化为 0 条而不是崩（A3 判定由 a3_readable 给出）。
"""
import json

import numpy as np
import pytest
from PIL import Image

import amct_ptq


def _write_png(path, level, seed):
    """level 越大 -> 方差越大，用来模拟"粗糙海况"。"""
    rng = np.random.default_rng(seed)
    arr = np.clip(120 + level * rng.normal(0, 1, (64, 64, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path


def test_stratified_pick_is_balanced_and_reproducible(tmp_path):
    imgs = [_write_png(tmp_path / f"img{i:02d}.png", 1 + (i % 9), i)
            for i in range(30)]
    a = amct_ptq.stratified_pick(imgs, num=9, seed=7)
    b = amct_ptq.stratified_pick(imgs, num=9, seed=7)
    assert [str(p) for p, _ in a] == [str(p) for p, _ in b]

    labels = [s for _, s in a]
    assert len(labels) == 9
    for s in amct_ptq.STRATA:
        assert labels.count(s) == 3


def test_contrast_of_orders_rough_above_calm(tmp_path):
    calm = _write_png(tmp_path / "calm.png", 0.2, 0)
    rough = _write_png(tmp_path / "rough.png", 8.0, 1)
    assert amct_ptq.contrast_of(rough) > amct_ptq.contrast_of(calm)


def test_float_layers_detects_mixed_precision():
    cfg = {"quant_config": {
        "conv1": {"activation_quant_params": {"quant_dtype": "INT8"}},
        "conv2": {"activation_quant_params": {"quant_dtype": "FP32"}},
    }}
    hits = amct_ptq.float_layers(cfg)
    assert len(hits) == 1 and "conv2" in hits[0]


def test_float_layers_clean_config_is_empty():
    cfg = {"quant_config": {"conv1": {"activation_quant_params": {"quant_dtype": "INT8"}}}}
    assert amct_ptq.float_layers(cfg) == []


def test_dump_config_writes_table_and_report(tmp_path):
    cfg = {"quant_config": {
        "model.22.conv": {"weight_quant_params": {"scale": 0.001, "offset": 0},
                          "activation_quant_params": {"scale": 0.02, "offset": 0}},
    }}
    src = tmp_path / "quant_config.json"
    src.write_text(json.dumps(cfg), encoding="utf-8")

    out = tmp_path / "scale.json"
    rep_json = tmp_path / "report.json"
    rep = amct_ptq.dump_config(str(src), str(out), str(rep_json))

    assert out.exists() and rep_json.exists()
    back = json.loads(rep_json.read_text(encoding="utf-8"))
    assert set(back) >= {"a3_readable", "output_entry", "n_entries", "float_layers"}
    assert rep["float_layers"] == []


def test_sar_calib_dataset_uses_letterbox(tmp_path):
    torch = pytest.importorskip("torch")
    p = _write_png(tmp_path / "a.png", 4.0, 3)
    ds = amct_ptq.SarCalibDataset([p], imgsz=128)
    x = ds[0]
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == (1, 3, 128, 128)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
