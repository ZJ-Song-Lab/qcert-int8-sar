"""modelslim 训练后量化 (PTQ) 标定脚本: FP32 ONNX -> INT8 ONNX。

在开发板(Atlas 200I DK / Ascend 310B, CANN 7.0.RC1)上运行。历史版本走
``amct_pytorch``，但其 .so 只编译到 cp37/38/39，板上 Python 3.10 无法加载，
故改用 CANN 自带的 ``modelslim.onnx.post_training_quant``（纯 Python 实现，
内部用 onnxruntime.quantization 做静态量化，再把 QDQ 块转成
AscendQuant -> Conv -> AscendDequant 形态）。

核心原则:
- 校准数据必须来自 SAR 训练集真实图像(域外校准可致 ~25% mAP 损失);
- 校准预处理必须与板上推理逐像素一致: 复用 ``postprocess.preprocess``
  (letterbox), modelslim 要求 calib_data 与模型输入 dtype/shape 严格一致,
  不匹配的样本会被 *静默丢弃*(只 warning), 用别的预处理等于悄悄缩水校准集;
- 输出 ``<out>.onnx`` 后再用 atc_convert.sh 转 OM, 即为 INT8 模型。

v3 三个开关(paper/05 §4):
- ``--no-fallback`` : 全 INT8。量化后审计仍存在未量化的 Conv/Gemm/MatMul
  (即混合精度回退)就直接失败, 而不是"悄悄变好"(风险 R3);
- ``--stratified``  : 按对比度三层(平静/中等/粗糙)等量抽样校准, 而不是
  取前 N 张; 分层是 E9 第 7 项证明的"压测集饱和率 8.97% -> 0.51%"的前提;
- ``--dump-config`` : 把量化 ONNX 里的逐层 scale(AscendQuant/AscendDequant)
  转成规范 ``ScaleTable`` json(A3 假设), 供 ``board/probe_scales`` 与
  ``qcert.transfer`` 使用。

板上注意事项(modelslim 实测):
- 板上没有 onnxoptimizer, ``is_optimize_graph`` 必须置 False(该开关只影响
  前置图优化, 量化主流程不受影响);
- calib_data 全部驻留内存: 256 张 640x640 float32 约 1.26GB, 板上内存不足
  时降低 --num-calib;
- 推荐流程: 本地先用 --data + --stratified 抽好 256 张推到板上, 板上用
  --images 指向该目录(直接全用, 不重复分层, 保证与本地抽样一致)。
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parent
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from postprocess import load_image_rgb, preprocess  # noqa: E402

CODE = DEPLOY.parents[1] / "code"        # .../dinghuibianyuan/code
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
STRATA = ("calm", "moderate", "rough")
LF_OP_TYPES = ("Conv", "Gemm", "MatMul")  # modelslim LF_QUANTIZED_OP_TYPES


class SarCalibDataset:
    """按给定图像列表做与推理一致的 letterbox 预处理。

    只实现 ``__len__/__getitem__``，不继承 ``torch.utils.data.Dataset``，
    这样 ``--dump-config`` 等路径即使没装 torch 也能 import 本模块。
    板上没有 torch: 此时 ``__getitem__`` 直接返回 numpy 数组(modelslim
    要的正是 ndarray)。
    """

    def __init__(self, images, imgsz: int = 640):
        self.imgs = [Path(p) for p in images]
        assert self.imgs, "校准图像列表为空"
        self.imgsz = imgsz

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x, _ratio, _pad = preprocess(self.imgs[idx], self.imgsz)   # (1,3,H,W) float32
        try:
            import torch
        except ImportError:
            return x
        return torch.from_numpy(x)


# ------------------------------------------------------------------ 校准集
def collect_images(data_yaml=None, images_dir=None):
    """取校准图像列表。

    - ``data_yaml``: ultralytics data.yaml, 取 train 目录全部图像(本地全量分层用);
    - ``images_dir``: 平铺图像目录(板上用: 本地分层抽好的 256 张直接推上去)。
    """
    if data_yaml:
        import yaml

        cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
        root = Path(cfg["path"]) / cfg["train"]
        imgs = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
        assert imgs, f"训练集为空: {root}"
        return imgs
    assert images_dir, "需要 data_yaml 或 images_dir"
    root = Path(images_dir)
    imgs = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
    assert imgs, f"校准图像目录为空: {root}"
    return imgs


def contrast_of(path) -> float:
    """对比度代理量: 灰度图的 ``std / (mean + eps)``。

    SSDD 没有海况标签, 用这个量做分层代理: 平静海面灰度平坦, 粗糙海面
    杂波强、方差大。分层只是把校准集在对比度上摊平, 不宣称等价于真海况。
    """
    g = load_image_rgb(path).astype(np.float32).mean(axis=2)
    return float(g.std() / (g.mean() + 1e-6))


def stratified_pick(imgs, num: int, seed: int = 0):
    """按对比度三等分后等量抽样，返回 ``[(path, stratum), ...]``（按路径排序）。"""
    vals = np.asarray([contrast_of(p) for p in imgs], dtype=np.float64)
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    idx = np.digitize(vals, [q1, q2])                     # 0/1/2
    rng = np.random.default_rng(seed)

    counts = [num // 3] * 3
    for i in range(num - sum(counts)):
        counts[i] += 1

    picked = []
    for s, need in enumerate(counts):
        pool = [imgs[i] for i in np.flatnonzero(idx == s)]
        if not pool:
            raise SystemExit(f"分层 {STRATA[s]} 无图像，校准集太小或数据退化")
        order = rng.permutation(len(pool))[:need]
        picked += [(pool[j], STRATA[s]) for j in order]
    picked.sort(key=lambda t: str(t[0]))
    print(f"[calib] 对比度三分位 {q1:.4f} / {q2:.4f}，"
          f"各层样本 {counts}，共 {len(picked)}")
    return picked


def build_calib(args):
    imgs = collect_images(args.data, args.images)
    if len(imgs) < int(args.num_calib):
        print(f"[calib] 注意: 可用图像 {len(imgs)} < 请求 {args.num_calib}，全部使用")
    if not args.stratified:
        sel = imgs[:int(args.num_calib)]
        return sel, [{"path": str(p), "stratum": "all"} for p in sel]
    picked = stratified_pick(imgs, int(args.num_calib), int(args.seed))
    return ([p for p, _ in picked],
            [{"path": str(p), "stratum": s} for p, s in picked])


# ------------------------------------------------------------------ 配置审计
_FLOAT_TOKENS = ("float", "fp16", "fp32", "bf16")
_DTYPE_KEYS = ("dtype", "quant_dtype", "precision", "quant_precision")


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}/{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def float_layers(cfg) -> list:
    """配置里被标成浮点的层名（旧 AMCT json 混合精度回退的痕迹；
    modelslim 流程直接用 ``audit_quant_onnx`` 审量化 ONNX）。"""
    out = []
    for path, val in _walk(cfg):
        key = path.rsplit("/", 1)[-1].split("[")[0].lower()
        if key in _DTYPE_KEYS and isinstance(val, str) \
                and any(t in val.lower() for t in _FLOAT_TOKENS):
            out.append(path)
    return out


def audit_quant_onnx(path) -> dict:
    """审计量化 ONNX: LF 算子(Conv/Gemm/MatMul)有多少真正挂上了量化(R3)。

    判定标准: 算子的输入由 AscendQuant / QuantizeLinear 产出。
    - 输入含 initializer(权重)却未量化 = 真正的"悄悄回退成 float";
    - 无 initializer 输入的 MatMul(如 C2PSA attention 的 ``q @ k^T`` /
      ``attn @ v``，纯激活×激活) 静态量化本就不可达，属已知豁免，
      显式列入 ``exempt`` —— 不豁免它就没有 INT8 attention 实现可言。
    """
    import onnx

    model = onnx.load(str(path))
    quant_feeds = set()
    for node in model.graph.node:
        if node.op_type in ("AscendQuant", "QuantizeLinear"):
            quant_feeds.update(node.output)

    inits = {i.name for i in model.graph.initializer}
    total = quantized = 0
    unquantized, exempt = [], []
    for node in model.graph.node:
        if node.op_type in LF_OP_TYPES:
            total += 1
            if any(i in quant_feeds for i in node.input):
                quantized += 1
            elif any(i in inits for i in node.input):
                unquantized.append(node.name or node.op_type)
            else:
                exempt.append(node.name or node.op_type)
    return {"total_lf": total, "quantized_lf": quantized,
            "n_unquantized": len(unquantized),
            "unquantized": unquantized[:20],
            "n_exempt": len(exempt), "exempt": exempt[:20]}


def dump_config(config_file, out_json: str, report_json: str | None = None):
    """量化产物 -> 规范 ScaleTable json，并报告输出层 scale 是否可读（A3）。

    ``config_file`` 支持两种形态:
    - ``*.onnx``: 量化后的 ONNX(modelslim 产物, AscendQuant/AscendDequant),
      走 ``ScaleTable.from_onnx_quant``; 此时报告里的 ``float_layers``
      是 ``audit_quant_onnx`` 审出的未量化 LF 层(R3);
    - ``*.json``: 旧 AMCT ``*_config.json``, 走 ``ScaleTable.from_amct_json``。
    """
    src = Path(config_file)
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    try:
        from qcert.scale_table import ScaleTable
    except Exception as exc:                              # pragma: no cover
        print(f"[dump-config] 跳过：qcert 不可 import（{exc}）")
        return None

    if src.suffix.lower() == ".onnx":
        table = ScaleTable.from_onnx_quant(src)
        floats = audit_quant_onnx(src)["unquantized"]
        source = "modelslim_onnx"
    else:
        cfg = json.loads(src.read_text(encoding="utf-8"))
        table = ScaleTable.from_amct_json(src)
        floats = float_layers(cfg)
        source = "amct"

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    table.to_json(out_json)
    print(table.summary())

    entry = None
    # 候选: 输出层命名约定。cls/cv3 = 分类头, box/cv2/dfl = 回归头,
    # yolo11 检测头节点名是 /model.23/cv2.* 或 /model.23/cv3.*, 无 "cls" 字样。
    for cand in ("output", "scores", "conv2d", "cls", "cv3", "cv2", "dfl"):
        for name in table.names():
            if cand in name.lower():
                entry = table.get(name)
                break
        if entry is not None:
            break
    report = {"source": source, "config": str(config_file),
              "scale_json": out_json, "n_entries": len(table.names()),
              "output_entry": entry.to_dict() if entry is not None else None,
              "float_layers": floats,
              "a3_readable": bool(entry is not None)}
    if report_json:
        Path(report_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dump-config] scale -> {out_json}  (A3 {'PASS' if entry else 'FAIL'})")
    return report


# ------------------------------------------------------------------ 主流程
def _patch_modelslim_for_ort116():
    """CANN 7.0.RC1 的 modelslim 无条件给 ``quantize_static`` 传
    ``optimize_model=False``，但板上 onnxruntime 1.16.3 已移除该形参
    (1.14 废弃；新版量化前本就不做图优化) -> ``TypeError``。
    在 modelslim 的模块命名空间打补丁: 吞掉该关键字, 语义等价。
    """
    from modelslim.onnx.post_training_quant.label_free import quantize_tool as _qt
    if "optimize_model" in inspect.signature(_qt.quantize_static).parameters:
        return False

    _orig = _qt.quantize_static

    def _qs_no_optimize_model(*args, **kwargs):
        kwargs.pop("optimize_model", None)
        return _orig(*args, **kwargs)

    _qt.quantize_static = _qs_no_optimize_model
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="modelslim PTQ: FP32 ONNX -> INT8 ONNX (板上 CANN 环境)")
    ap.add_argument("--model", help="输入 FP32 ONNX(如 best.onnx / yolo11n.onnx)")
    ap.add_argument("--data", default=None,
                    help="ultralytics data.yaml(取 train 目录做校准, 本地分层用)")
    ap.add_argument("--images", default=None,
                    help="校准图像平铺目录(板上用, 指向推上来的图)")
    ap.add_argument("--num-calib", type=int, default=256, help="校准图片数(建议 >=256)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="best_quant", help="输出名(生成 <out>.onnx)")
    ap.add_argument("--seed", type=int, default=0, help="--stratified 的抽样种子")
    ap.add_argument("--calib-method", type=int, default=0, choices=(0, 1, 2),
                    help="校准统计: 0=min-max 1=percentile 2=entropy")
    ap.add_argument("--exclude", default="",
                    help="逗号分隔的不量化节点名(敏感层手动回退)")
    ap.add_argument("--amp", type=int, default=0,
                    help="modelslim 自动混合精度回退层数(R3 风险, 默认 0=全 INT8)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="全 INT8: --amp>0 或量化后仍有 LF 层未量化即失败(R3)")
    ap.add_argument("--stratified", action="store_true",
                    help="按对比度三层等量抽样校准(替代取前 N 张)")
    ap.add_argument("--dump-config", action="store_true",
                    help="把逐层 scale 转成规范 ScaleTable json(A3)")
    args = ap.parse_args(argv)

    # --dump-config 可以在不重新量化、也没有 modelslim 的情况下单独跑
    if args.dump_config and not args.model:
        cands = [f"{args.out}.onnx", f"{args.out}_config.json"]
        src = next((c for c in cands if Path(c).exists()), None)
        if src is None:
            raise SystemExit(f"找不到 {' 或 '.join(cands)}，先跑一次完整 PTQ")
        dump_config(src, f"{args.out}_scale.json", f"{args.out}_scale_report.json")
        return

    if not args.model:
        ap.error("需要 --model（FP32 ONNX）；旧 amct_pytorch --weights 流程已弃用")
    if not (args.data or args.images):
        ap.error("需要 --data 或 --images（校准图必须来自 SAR 训练集）")
    if args.no_fallback and args.amp > 0:
        raise SystemExit("[no-fallback] 与 --amp>0 互斥（R3：不允许混合精度回退）")

    try:
        from modelslim.onnx.post_training_quant import QuantConfig, run_quantize
    except Exception as exc:                              # pragma: no cover
        raise SystemExit(
            f"无法 import modelslim（{exc}）。\n"
            "请 source /usr/local/Ascend/ascend-toolkit/set_env.sh 后重试。"
            "modelslim 是 CANN 自带的纯 Python PTQ 包(板上 Python 3.10 可用);"
            "amct_pytorch 的 .so 只编译到 cp37/38/39, 已弃用。")

    if _patch_modelslim_for_ort116():
        print("[patch] 板上 onnxruntime>=1.16 的 quantize_static 无 optimize_model "
              "形参，已为 modelslim 打兼容补丁（语义等价：量化前不做图优化）")

    calib_imgs, calib_records = build_calib(args)
    Path(f"{args.out}_calib.json").write_text(
        json.dumps({"n": len(calib_imgs), "stratified": bool(args.stratified),
                    "seed": int(args.seed), "items": calib_records},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # 1) 校准数据: 与推理逐像素一致的 letterbox 输出, dtype/shape 严格匹配
    ds = SarCalibDataset(calib_imgs, args.imgsz)
    per_img = 3 * args.imgsz * args.imgsz * 4
    print(f"[calib] {len(ds)} 张 × {per_img / 1e6:.1f}MB/张 ≈ "
          f"{len(ds) * per_img / 2 ** 30:.2f}GB 内存(modelslim 会全部驻留)")
    calib_data = []
    for i in range(len(ds)):
        x = ds[i]
        calib_data.append(x.numpy() if hasattr(x, "numpy") else np.asarray(x))
        if (i + 1) % 32 == 0 or i + 1 == len(ds):
            print(f"校准数据准备: {i + 1}/{len(ds)}")

    # 2) 量化 + 标定
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()] or None
    quant_onnx = f"{args.out}.onnx"
    cfg = QuantConfig(
        quant_mode=1,                   # 1=量化(产出 AscendQuant/AscendDequant 块)
        is_signed_quant=True,
        is_per_channel=True,
        calib_data=calib_data,          # [(1,3,H,W) float32]; 不匹配项会被静默丢弃
        calib_method=int(args.calib_method),
        exclude_nodes=exclude,
        amp_num=int(args.amp),
        is_optimize_graph=False,        # 板上没有 onnxoptimizer
    )
    run_quantize(args.model, quant_onnx, cfg)
    print(f"INT8 ONNX: {quant_onnx}")

    # 3) 审计: R3 — 是否有 LF 层悄悄回退成 float
    audit = audit_quant_onnx(quant_onnx)
    tail = (f"；无浮点回退"
            if not audit["n_unquantized"]
            else f"；未量化(浮点回退): {audit['unquantized']}")
    if audit["n_exempt"]:
        tail += f"；豁免(激活×激活MatMul, 静态量化不可达): {audit['exempt']}"
    print(f"[audit] LF 算子 {audit['quantized_lf']}/{audit['total_lf']} 已量化{tail}")
    if args.no_fallback and audit["n_unquantized"]:
        raise SystemExit(f"[no-fallback] 仍有 {audit['n_unquantized']} 个 LF 层未量化"
                         f"（例如 {audit['unquantized'][:3]}），按 R3 拒绝继续。")

    # 4) 顺手把 scale 表落成规范 json（A3）
    if args.dump_config:
        dump_config(quant_onnx, f"{args.out}_scale.json",
                    f"{args.out}_scale_report.json")


if __name__ == "__main__":
    main()
