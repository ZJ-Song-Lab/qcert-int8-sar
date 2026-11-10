#!/bin/bash
# E5 设置 1（校准集 A→B）板上全流程（paper/04 §8）：
#   1. modelslim PTQ 双跑：A=calm-256 / B=rough-256 两份单层校准集
#      -> yolo11n_ssdd_int8_calmA.onnx / _roughB.onnx（+ _scale.json）
#   2. atc 转 OM x2（同一套默认参数，保证 A/B 可比）
#   3. 双 dump 分数场（232 张 SSDD test，raw logit 域）
# 前置：E6 必须已结束（PTQ 校准数据 1.26GB 驻留 + onnxruntime CPU 前向，
#       与 E6 并行会争内存/CPU 并扰动 E6 的 P99 时延）。
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/asc_sar

M=models
D=deploy
DATA=data
OUT=code/dumps/e5s1
mkdir -p "$OUT"

for AB in calmA roughB; do
    case $AB in
        calmA)  CAL=$DATA/calib_calm256 ;;
        roughB) CAL=$DATA/calib_rough256 ;;
    esac
    echo "=== [1/3] PTQ $AB  calib=$CAL ==="
    python3 -u "$D/amct_ptq.py" --model "$M/yolo11n_ssdd.onnx" \
        --images "$CAL" --out "$M/yolo11n_ssdd_int8_$AB" --dump-config
done

for AB in calmA roughB; do
    echo "=== [2/3] ATC $AB ==="
    bash "$D/atc_convert.sh" "$M/yolo11n_ssdd_int8_$AB.onnx" \
        "$M/yolo11n_ssdd_int8_$AB"
done

cd /root/asc_sar/code
for AB in calmA roughB; do
    echo "=== [3/3] DUMP $AB ==="
    ASC_DEPLOY=/root/asc_sar/deploy python3 -u -m board.dump_scores \
        --model "/root/asc_sar/models/yolo11n_ssdd_int8_$AB.om" \
        --images /root/asc_sar/data/images \
        --labels /root/asc_sar/data/labels \
        --scale-json "/root/asc_sar/models/yolo11n_ssdd_int8_${AB}_scale.json" \
        --out "$OUT/int8_$AB" --impl int8
done

echo "=== E5S1 ALL DONE ==="
ls -la "$OUT"/int8_calmA "$OUT"/int8_roughB | head -5
