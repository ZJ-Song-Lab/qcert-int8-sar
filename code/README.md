# QCert · 整数域误警率契约（实现代码）

本目录是 `paper/04_实验设计与结果.md` 中 **E0–E9** 的可运行实现。方法本体（v3）：
阈值在 FP32 标定、推理在 INT8 执行是**静默失效**的范式错误；QCert 把阈值搬进整数域，
使 `P_fa` 在 Atlas 200I DK A2 上可标定、可搬运、可审计。

> 论文口径：E0（存在性）→ E1（Prop.2）→ **E2/E3/E5/E6/E7 只有定点板能做**。
> 离线可跑的部分在 x86 上复现；板级部分必须上板，见 §4。

---

## 1. 目录

```
code/
├─ qcert/                 方法库（只依赖 numpy，可离线导入，全量单测覆盖）
│  ├─ scores.py           分数场容器：s / μ / σ 三个场 + letterbox 几何
│  ├─ scale_table.py      .om 量化 scale/zp 表读取与换算（A3）
│  ├─ qint.py             整数分位数（直方图 B 桶 / 精确排序）、量化、饱和率
│  ├─ decision.py         整数判决（交叉相乘，无除法）
│  ├─ pfa_eval.py         Ω_cal/Ω_meas 构造、λ 标定、P_fa/P_d 测量
│  ├─ audit.py            逐层饱和审计（sat_l / 并列数）
│  ├─ transfer.py         阈值搬运（旧模型 → 新模型）
│  ├─ seastate.py         海况分层标定
│  ├─ cfar.py             CA-CFAR 基线（E8 必需对照）
│  └─ thermal.py          热稳态漂移统计
├─ experiments/           E0–E9 驱动脚本 + 合成场景
│  ├─ common.py           路径约定、场景配对、Ω 构造、表格落盘（csv/md/json）
│  ├─ synth.py            合成杂波场（对数正态 + 相关 + 量化误差注入）
│  └─ run_e0.py … run_e9.py
├─ board/                 仅上板可用；顶层不 import acl / amct
│  ├─ runner.py           pyACL 推理封装（延迟 import）
│  ├─ dump_scores.py      跑 .om 落分数场 npz
│  ├─ probe_scales.py     从 .om 读 scale/zp 表
│  └─ bench_latency.py    时延/功率/NPU 占用
├─ tests/                 pytest：数学一致性 + 板级 smoke（无板则 skip）
├─ dumps/                 demo 分数场（离线演示用，非论文数据）
└─ results/               E0–E9 产物（每个子目录含 summary.json + 表格三件套）
```

## 2. 环境

| 环境 | Python | 可跑 |
|---|---|---|
| Windows x86（本机） | 3.12 | `qcert/`、`experiments/`、`tests/` 全部；`board/` 只能 dry-run |
| Atlas 200I DK A2 | 3.10（CANN 自带） | 全部 + `board/` 真机路径 |

依赖：`numpy scipy pyyaml opencv-python pillow torch onnx onnxruntime pytest`
（`torch/onnx` 仅 `asc-sar` 转换用，本目录离线脚本不强制）。
板端额外由 CANN 提供 `acl`、`amct`。

**编码约束**：板端是 3.10，禁止 3.11+ 语法；所有模块已加 `from __future__ import annotations`。
`qcert/` 顶层**不得** `import acl`，否则 x86 上整个包不可导入。

## 3. 离线快速开始

在本目录（`code/`）下执行，包路径按当前工作目录解析：

```powershell
python -m pytest tests -q                 # 数学 + smoke，无板自动 skip（当前 29 passed）
python -m experiments.run_e0 --fp32 dumps/demo_fp32 --int8 dumps/demo_int8
```

### 3.1 一键复跑全部离线实验

```powershell
python -m experiments.run_e0  --fp32 dumps/demo_fp32 --int8 dumps/demo_int8 --out results/e0
python -m experiments.run_e1  --out results/e1
python -m experiments.run_e2  --synth --out results/e2
python -m experiments.run_e3  --synth --n-scenes 8 --out results/e3
python -m experiments.run_e4  --eps-sigma 0.02 0.05 0.10 0.20 --out results/e4
python -m experiments.run_e5  --synth --out results/e5
python -m experiments.run_e6  --dry-run --minutes 10 --window 5 --out results/e6
python -m experiments.run_e7  --synth --n 12 --out results/e7
python -m experiments.run_e8  --synth --out results/e8
python -m experiments.run_e9  --n 48 --keep 0.5 --out results/e9
```

> 所有脚本统一 `--out <dir>`（早期版本混用 `--out-dir` 已废弃）。
> 每个 `results/<eid>/` 下有 `summary.json` 与 `table*.{csv,md,json}`。
> `results/_smoke`、`results/*_dry`、`results/e0_smoke` 是调试产物，**不可入论文**。

## 4. 板级：M0 → M1（硬门）

**M0（环境核对 + scale 表可读 + 分数场打通）不做完，不要投入训练成本。**

```bash
# M0-a  拷代码与模型
scp -r code qcert experiments board  root@<板子IP>:/home/HwHiAiUser/
# M0-b  scale 表可读（真 A3）
python -m board.probe_scales --om model_int8.om --out results/m0/scales.json
# M0-c  分数场 dump 打通
python -m board.dump_scores --om model_int8.om --images <dir> --out dumps/real_int8
python -m board.dump_scores --om model_fp16.om --images <dir> --out dumps/real_fp16
# M0-d  时延/功率/NPU
python -m board.bench_latency --om model_int8.om --n 200 --out results/m0/lat.json
```

**M1（E0 存在性，硬门 `R ≥ 1.5`）**

```bash
python -m experiments.run_e0 --fp32 dumps/real_fp16 --int8 dumps/real_int8 \
       --labels <label_dir> --map-json mAP.json --out results/e0_real
```

`R < 1.5` → 走 `paper/04` §13.2 的降级方案（饱和主线），不要改判据凑数。

**其余板级实验**：E2（`--fp32/--int8/--layers` 换成真实 dump）、E3（`--repeat` 需
`acl_infer.py --repeat 1000` 产出的 json 才能验 A1）、E5（`--old/--new` 两版 `.om`）、
E6（`--om --image`，40 min × 5 min 窗）、E7（`--fp16/--int8 --latency-json`）。

## 5. 统计纪律（结果能不能进论文，看这三条）

1. **`--keep` 默认 0.5，别调小。**
   `Ω_meas` 只有 `n_meas` 格 → `P_fa` 最小非零粒度是 `1/n_meas`。E9 第 3 项的实测：

   | keep | P_fa | α 误差 \|log10\| | n_eff |
   |---|---|---|---|
   | 1     | 1.094e-03 | 0.039 | 49.9 |
   | 0.25  | 1.348e-03 | 0.130 | 25 |
   | 0.125 | 1.839e-03 | 0.265 | 12 |
   | 1/16  | 2.779e-03 | **0.444** | **6** |

   1/16 已经偏 0.44 decade，`α=1e-3` 完全不可分辨。
2. **`n_meas·α ≥ 10` 是硬门。** 不满足时脚本把 `insufficient=true` 并强制 `pass=false`，
   不要手改。要提分辨率就加大 `--keep` 或换更小的 `α`。
3. **表里必须带 `keep` 与 `source`。** `source="dry-run"` / `"synth"` 的表是自检，
   不是实验证据；只有 `source="dump"` / `"board"` 才能进论文。`summary.json` 已自动落这两个字段。

## 6. 离线结果的正确读法（避免误判）

- **E0 离线 `R ≈ 0.99–1.03`、判 NO-GO 是预期的**：demo dump 是合成场，INT8 与 FP32
  几乎没有差别。`R ≥ 1.5` 只有在真实模型 + 真实杂波压测下才有意义，**必须板级复核**。
- **E4 的 P1（≥5×）在离线合成上必然 FAIL**：log/逐通道量化只把 `ε_σ` 的 p95 改善约 1.5×，
  中位数几乎不动，5× 数学上不可达。`|Δln P_fa| ∝ ε_σ^p` 的**指数**（实测 1.93–1.97）
  才是这一项的有效证据；倍数判据待 `paper/04` §7 修订。
- **E8 的 `--synth` 判 FAIL 不算结论**：合成亮矩形上 CA-CFAR 是理想检测器（`P_d=1.000`），
  真实分数场必须用 `--fp16/--int8` 复跑。
- **E9 第 8 项**：`round` 的局部指数 2.36→2.02→1.66（趋于 2，一阶项相消），
  `floor` 1.43→1.26→1.04（趋于 1，一阶项 `z²ε_σ` 浮现）。所以 Prop.2 的
  `Δln P_fa ≈ z²ε_σ` 应表述为「**有偏舍入下的上界/一阶项**」。
- **E9 第 2 项**：`B=4096` 的 α 误差 0.0836 大于精确排序的 0.0775 → 瓶颈是**测量集抽样
  方差**，不是桶宽。论文里写「直方图 B=4096 已足够」，不要宣称桶宽是误差主项。
- **E9 第 4 项**要求 `--calib-scenes ≥ 320`（脚本内部 `max(calib_scenes, 64)`），
  否则 256 场景档会重复抽样。

## 7. 常见坑

| 现象 | 原因 |
|---|---|
| `ModuleNotFoundError: qcert` | 没在 `code/` 目录下执行；用 `python -m experiments.run_eX` 而非直接跑文件 |
| `ImportError: acl` | x86 上调了 `board/*` 真机路径；加 `--dry-run` 或上板 |
| `unrecognized arguments: --out-dir` | 旧参数名，统一改 `--out` |
| `R` 抖动大 / P_fa 是 `1/n_meas` 的整数倍 | `--keep` 太小，见 §5 |
| pytest 报 letterbox 尺寸不符 | 小图会被**放大**到正方形 `(new_shape, new_shape)`，不是只补到 stride 倍数 |

## 8. 与 `asc-sar/` 的关系

`asc-sar/deploy/amct_ptq.py` 已改为 modelslim PTQ（板上 Python 3.10 无 amct_pytorch .so），
保留三开关（`--no-fallback / --stratified / --dump-config`）并复用 `preprocess`，
量化后的 `.onnx`/`.om` 供 `board/dump_scores.py` 使用，逐层 scale 走
`qcert/scale_table.py` 的 `from_onnx_quant`（支持 AscendQuant/AscendDequant）；
`asc-sar/deploy/postprocess.py` 的 letterbox 已修到正方形补边。CANN 侧实跑待上板。
