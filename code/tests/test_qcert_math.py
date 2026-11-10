"""qcert 的数学与换算：量化往返、整数判决等价性、阈值搬运、scale 表。"""
import numpy as np
import pytest

from qcert import decision, qint, scale_table, transfer


def test_quantize_roundtrip_within_half_delta():
    x = np.linspace(-3.0, 3.0, 1001)
    q = qint.quantize(x, delta=3.0 / 127, zero_point=0, bits=8)
    back = qint.dequantize(q, delta=3.0 / 127, zero_point=0)
    err = np.abs(back - x)
    assert err.max() <= 3.0 / 127 / 2 + 1e-9
    assert q.min() >= -128 and q.max() <= 127


def test_decide_matches_float_reference():
    """整数判决 vs FP32 参照：不一致率应远低于 1%。"""
    rng = np.random.default_rng(0)
    n = 200_000
    s_q = rng.integers(-128, 128, n)
    mu_q = rng.integers(-20, 20, n)
    sigma_q = rng.integers(1, 30, n)

    spec = decision.IntQuantSpec(delta_s=1 / 64, delta_mu=1 / 64,
                                 delta_sigma=1 / 256, shift_bits=20)
    lam = 3.0
    pred_int = decision.decide(s_q, mu_q, sigma_q, lam, spec)
    z = decision.z_from_int(s_q, mu_q, sigma_q, spec)
    pred_float = decision.decide_float(z, lam)

    rep = decision.compare_decisions(pred_int, pred_float)
    assert rep["disagree_rate"] < 1e-2, rep


def test_decide_simple_equals_general_when_scales_match():
    rng = np.random.default_rng(1)
    n = 50_000
    s_q = rng.integers(-128, 128, n)
    mu_q = rng.integers(-20, 20, n)
    sigma_q = rng.integers(0, 30, n)          # 含 0 以验证 sigma_min_q 兜底

    k, lam_q = 16, 3 * 512
    simple = decision.decide_simple(s_q, mu_q, sigma_q, k, lam_q, sigma_min_q=1)

    spec = decision.IntQuantSpec(delta_s=1.0, delta_mu=1.0, delta_sigma=1.0,
                                 sigma_min_q=1, shift_bits=k)
    general = decision.decide(s_q, mu_q, sigma_q, lam=lam_q / 2 ** k, spec=spec)
    assert np.array_equal(simple, general)


def test_calibrate_int_matches_float_quantile():
    rng = np.random.default_rng(2)
    n = 100_000
    s_q = rng.integers(-60, 100, n)
    mu_q = np.zeros(n, dtype=np.int64)
    sigma_q = rng.integers(1, 40, n)
    spec = decision.IntQuantSpec(1 / 64, 1 / 64, 1 / 256, shift_bits=20)
    mask = np.ones(n, dtype=bool)

    lam_i = decision.calibrate_int(s_q, mu_q, sigma_q, mask, 1e-3, spec)
    lam_f = decision.calibrate_float(
        decision.z_from_int(s_q, mu_q, sigma_q, spec), mask, 1e-3)
    assert abs(lam_i - lam_f) < 0.05 * max(abs(lam_f), 1.0)


def test_lambda_int_roundtrip():
    spec = decision.IntQuantSpec(1 / 128, 1 / 128, 1 / 512, shift_bits=16)
    for lam in (0.0, 1.5, 3.09, 6.0):
        q = decision.lambda_int(spec, lam, k=16)
        back = q * spec.delta_s / (spec.delta_sigma * 2 ** 16)
        assert abs(back - lam) <= spec.delta_s / spec.delta_sigma / 2 ** 16 * 1.5


def test_transfer_selfcheck_and_roundtrip():
    a, b = transfer._selfcheck()
    assert abs(a - b) <= 1

    old = transfer.TransportSpec(1 / 128, 1 / 512, 16)
    new = transfer.TransportSpec(1 / 100, 1 / 400, 16)
    lam = 3.09
    lam_q_old = transfer.lambda_int_from_float(lam, old)
    lam_q_new = transfer.transport_lambda_int(lam_q_old, old, new)
    assert abs(lam_q_new - transfer.lambda_int_from_float(lam, new)) <= 1
    assert abs(transfer.lambda_float_from_int(lam_q_new, new) - lam) < 0.05


def test_transfer_rejects_shift_mismatch():
    old = transfer.TransportSpec(1 / 128, 1 / 512, 16)
    new = transfer.TransportSpec(1 / 128, 1 / 512, 20)
    with pytest.raises(ValueError):
        transfer.transport_lambda_int(1000, old, new)


def test_predicted_dln_pfa_and_drift():
    assert transfer.predicted_dln_pfa(3.0, 0.02) == pytest.approx(0.18)
    assert transfer.drift_log10(1e-3, 1e-3) == pytest.approx(0.0)
    assert transfer.drift_log10(2e-3, 1e-3) == pytest.approx(np.log10(2.0))


def test_scale_table_npz_json_roundtrip(tmp_path):
    p = tmp_path / "s.npz"
    np.savez(p, delta=np.float32(1 / 128), zero_point=np.int32(-3))
    t = scale_table.ScaleTable.from_npz(p, name="output0")
    e = t.require("output0")
    assert e.delta == pytest.approx(1 / 128) and e.zero_point == -3

    j = tmp_path / "s.json"
    t.to_json(j)
    back = scale_table.ScaleTable.from_json(j)
    assert back.require("output0").delta == pytest.approx(e.delta)

    spec = transfer.TransportSpec.from_scale_table(back, "output0", "output0")
    assert spec.delta_s == pytest.approx(1 / 128)


def _make_ascend_quant_onnx(path, x_scale=0.02, wx=0.05):
    """构造 modelslim 风格的最小量化 ONNX（AscendQuant -> Conv -> AscendDequant）。

    两个非显然约定（来自 modelslim dag/graph 源码）：
    - AscendQuant 的 ``scale`` 属性存的是 *倒数* 1/x_scale；
    - AscendDequant 的 input[1] 是 UINT64，低 32 位是 w_scale*x_scale 的
      float32 位模式。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inv = 1.0 / x_scale
    dq_bits = np.array([wx], dtype=np.float32).view(np.uint32).astype(np.uint64)

    nodes = [
        helper.make_node("AscendQuant", ["X"], ["aq_output"], name="aq0",
                         offset=0.0, quant_bit=8, scale=float(inv)),
        helper.make_node("Conv", ["aq_output", "W"], ["c_output"], name="c0",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("AscendDequant", ["c_output", "c0_dequant_scale"],
                         ["out0"], name="ad0"),
    ]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, 1, 8, 8])],
        [numpy_helper.from_array(np.zeros((1, 3, 3, 3), np.float32), "W"),
         numpy_helper.from_array(dq_bits, "c0_dequant_scale")])
    onnx.save(helper.make_model(graph, producer_name="test"), path)


def test_scale_table_reads_ascend_quant_nodes(tmp_path):
    """from_onnx_quant 必须还原 AscendQuant 倒数属性与 AscendDequant 位模式。"""
    pytest.importorskip("onnx")
    p = tmp_path / "m.onnx"
    _make_ascend_quant_onnx(p, x_scale=0.02, wx=0.05)

    t = scale_table.ScaleTable.from_onnx_quant(p)
    q = t.require("aq0")
    assert q.delta == pytest.approx(0.02)
    assert q.zero_point == 0 and q.bits == 8 and q.source == "AscendQuant"
    d = t.require("ad0")
    assert d.delta == pytest.approx(0.05, rel=1e-6)
    assert d.source == "AscendDequant"


def test_e5_counts_are_pooled_not_per_scene_mean():
    """P_fa 必须按 hits/n 池化：逐场景比值求均值会被 0 命中场景拉偏。"""
    from experiments.run_e5 import _counts

    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, True, True, True, False])
    c = _counts(z, mask, 1.5)           # 严格大于 1.5 的是 {2, 3}
    assert (c["n"], c["hits"]) == (4, 2)
    assert c["pfa"] == pytest.approx(0.5)


def test_e5_smoke_pools_hits_over_scenes(tmp_path):
    """E5 端到端：聚合行里的 P_fa 必须等于 hits/n，且样本量足够分辨 1e-3。"""
    from experiments import run_e5

    s = run_e5.main(["--synth", "--n-scenes", "2", "--alpha", "1e-3",
                     "--out", str(tmp_path)])
    for r in s["rows"]:
        # keep=0.5 下每个场景 n_meas ≈ 4k，池化两个场景后足以分辨 1e-3
        assert r["n_meas"] > 1000
        assert 0.0 <= r["pfa_transport"] <= 1.0
        assert isinstance(r["pass"], bool)
        assert r["pass"] == (not r["insufficient"] and
                             r["drift_transport"] <= 1.5 * max(r["drift_recal"], 1e-12))
    assert s["n_rows"] == 8             # 2 场景 x 4 配置
