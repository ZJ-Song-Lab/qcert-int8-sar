# ASC-SAR 复现工程

SAR 舰船检测在 Atlas 200I DK（Ascend 310B）上的端到端复现基线，对应 kimipaper 论文方案的阶段 0–1：
YOLOv11n 基线在 SSDD 上训练 → 导出无 NMS 的 ONNX → 板上 ATC 转换（FP16 / AMCT INT8）→ ACL 推理 + mAP/FPS/功耗实测。

## 目录结构

```
asc-sar/
├── datasets/
│   ├── prepare_ssdd.py     # SSDD(VOC) -> YOLO格式, 官方尾号1/9测试协议
│   └── prepare_hrsid.py    # HRSID(COCO) -> YOLO格式
├── deploy/
│   ├── aipp.cfg            # AIPP 预处理配置(可选, 默认不启用)
│   ├── atc_convert.sh      # ATC: ONNX -> OM (Ascend310B1/B4)
│   ├── amct_ptq.py         # modelslim 训练后量化标定 (PTQ -> INT8 ONNX)
│   ├── postprocess.py      # 解码 + NMS (numpy, 不依赖torchvision)
│   └── acl_infer.py        # PyACL 推理 + 评测 + benchmark (开发板运行)
├── tools/
│   └── eval_map.py         # mAP@0.5 / mAP@0.5:0.95 离线评估
├── train.py                # 训练 (Windows/Linux GPU)
└── export_onnx.py          # 导出无NMS的ONNX (Windows/Linux)
```

## 复现流程

**步骤 1–3 在 Windows 训练机执行；步骤 4–6 在 Atlas 200I DK 执行。**

```bash
# 1. 数据准备（SSDD VOC 原始目录: JPEGImages/ + Annotations/）
python datasets/prepare_ssdd.py --ssdd-dir D:\datasets\SSDD --out datasets\ssdd

# 2. 训练基线
python train.py --data datasets\ssdd\ssdd.yaml --model yolo11n.pt --epochs 150

# 3. 导出 ONNX（无NMS, FP32, 静态shape, batch=1, opset=12）
python export_onnx.py --weights runs\ssdd_yolo11n\weights\best.pt

# ---- scp best.onnx 到开发板 ----

# 4. 板上环境（每次新开终端）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 5a. FP16 转换 + 推理评测
bash deploy/atc_convert.sh best.onnx models/ssdd_fp16 Ascend310B1
python3 deploy/acl_infer.py --model models/ssdd_fp16.om \
    --images datasets/ssdd/images/test --labels datasets/ssdd/labels/test

# 5b. INT8: modelslim 标定(板上, Python3.10 可用; amct_pytorch 已弃用) -> ATC -> 评测
python3 deploy/amct_ptq.py --model best.onnx --images /root/asc_sar/data/calib \
    --num-calib 256 --no-fallback --dump-config --out best_quant
bash deploy/atc_convert.sh best_quant.onnx models/ssdd_int8 Ascend310B1
python3 deploy/acl_infer.py --model models/ssdd_int8.om \
    --images datasets/ssdd/images/test --labels datasets/ssdd/labels/test

# 6. 性能实测（FPS/延迟/功耗）
python3 deploy/acl_infer.py --model models/ssdd_int8.om --images datasets/ssdd/images/test \
    --labels datasets/ssdd/labels/test --benchmark --iters 200
```

输出预测 txt 到 `--save-dir`（YOLO 格式），可用 `tools/eval_map.py` 复算 mAP。

## 关键设计（对应论文卖点）

- **无 NMS 导出**：310B 不支持 `NonMaxSuppression` 算子，ONNX 中剥离，NMS 在 CPU 端 numpy 实现（`deploy/postprocess.py`）。
- **静态 shape**：固定 `1,3,640,640`、batch=1，避免 ATC 动态 shape 调优坑。
- **FP32 ONNX 导出**：不量化、不 half，FP16/INT8 分别由 ATC / modelslim 完成，保证两条链路公平可比。
- **量化校准必须用 SAR 域数据**：`amct_ptq.py` 从训练集抽真实 SAR 图像校准（域外校准可致 ~25% mAP 损失）。
- **AIPP 可选**：弱 CPU 场景把 /255 归一化下沉 AIPP（`deploy/aipp.cfg`），默认关闭以保证链路简单。

## 已知坑清单（实测时逐项核对）

| 现象 | 原因 | 对策 |
|---|---|---|
| ATC 报 `Unsupported op type: NonMaxSuppression` | 导出含 NMS | 已用 `nms=False` 导出；若仍报错检查 `end2end` 开关 |
| INT8 后 mAP 大幅下降 | 校准数据非 SAR 域 | 用 `--num-calib 256` 以上真实数据重标定 |
| ACL 输出与 PyTorch 对不上 | force_fp16 数值差异 | 对比时置信度阈值放宽到 0.05 看趋势，sigmoid/exp 层精度差 |
| 加载 OM 失败 | soc_version 不匹配 | Atlas 200I DK A2 为 `Ascend310B1`（部分批次 B4，用 `npu-smi` 查） |
