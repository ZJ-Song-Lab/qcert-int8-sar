# QCert: Certified False-Alarm Rate for INT8 SAR Ship Detection

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2024-blue)](https://neurips.cc)
[![IEEE TGRS](https://img.shields.io/badge/IEEE-TGRS-blue)](https://ieeexplore.ieee.org)

## 核心主张

在定点 NPU 上，**误警率契约**必须活在整数算术里。本文证明：

1. **解析界**（Prop.2）：逐层 scale 传播为 P_fa 漂移的闭式上界
2. **整数阈值搬运**：scale 比值纯函数映射，零数据、零重标定
3. **板级实测**：Atlas 200I DK A2 上 P_d@α=1e-3 达 0.991，CA-CFAR 仅 0.139

## 目录结构

```
code/
  qcert/           # QCert 核心库
    pfa_eval.py    # P_fa 评估、λ 标定、Ω 构建
    scores.py      # 分数场、grid、cell centers
    transfer.py    # 阈值搬运 TransportSpec
  experiments/
    make_figs.py       # Fig.1-7
    make_figs_v2.py    # sample_* 系列
    viz_new_figs.py    # Fig.31-35
    e0-e9/             # 全部实验脚本
  results/         # 实验结果 CSV/JSON
  dumps/           # 分数场 dump（e5s1 等）
paper/
  figs/            # 论文图（PNG+PDF）
  latex/           # elsarticle 模板
    main.tex
    body.tex
  QCert论文_AppliedSoftComputing格式.pdf
```

## 快速开始

```bash
# 安装依赖
pip install numpy pandas matplotlib opencv-python

# 重新生成论文图
cd code/experiments
python make_figs.py
python make_figs_v2.py
python viz_new_figs.py

# 编译论文
cd ../../paper/latex
xelatex main.tex
xelatex main.tex
```

## 实验结果索引

| 实验 | 结果文件 | 关键数字 |
|------|---------|---------|
| E0 基线 | `results/e0/e0_per_scene.csv` | FP16 vs INT8 mAP 差 <1% |
| E4 量化标度 | `results/e4/table3_e4.csv` | Δln P_fa ∝ ε_σ² |
| E5S1 场景 | `results/e5_s1/e5_per_scene.csv` | 逐场景 λ 标定 |
| E7 近岸/远洋 | `results/e7/e7_per_frame_pfa.csv` | 近岸 P_fa 尾部 2α+ |
| E8 SOTA 对比 | `results/e8/table6_e8.json` | P_d 0.991 vs 0.139 |
| E9 消融 | `results/e9/table3_e9.json` | 8 个消融 item |

## 硬件环境

- **板卡**: Atlas 200I DK A2 (Ascend 310B3, 8 TOPS INT8)
- **工具链**: CANN 7.0 + AMCT + ATC
- **精度验证**: Vitis AI 口径（FP16 ±1 mAP 门限）

## 引用

```bibtex
@article{qcert2026,
  title={QCert: Certified False-Alarm Rate for INT8 SAR Ship Detection},
  author={...},
  journal={Applied Soft Computing},
  year={2026}
}
```

## 许可证

MIT License（代码）/ CC-BY-4.0（论文与数据）
