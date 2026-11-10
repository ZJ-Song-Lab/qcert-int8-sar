#!/bin/bash
# ATC 转换: ONNX -> OM
# 用法: bash atc_convert.sh <onnx路径> <输出om名(不含后缀)> [soc_version] [aipp_cfg(可空)] [implmode]
# 例:   bash atc_convert.sh best.onnx best                  # 自动探测 soc, FP16
#       bash atc_convert.sh best.onnx best_int8 "" aipp.cfg # 带 AIPP(输入 uint8)
#       bash atc_convert.sh yolo11n_int8.onnx yolo11n_int8_hp "" "" high_precision
# implmode: high_performance(默认) / high_precision —— 量化 Conv 实现模式,
#           ATC 默认 high_performance, 量化算子累加精度可能受损(B 问题嫌疑)。
# pmode(第6参): allow_fp32_to_fp16(默认) / must_keep_origin_dtype ——
#           后者保持 fp32, 避免中间算子(MaxPool/MatMul/部分Conv)降 fp16。
#           注意: CANN 7.0.RC1(310B4) 无 enforce_keep_origin_dtype,
#           合法值含 must_keep_origin_dtype / force_fp32 / cube_fp16in_fp32out。
set -e

ONNX=$1
OUTPUT=$2
SOC=$3
AIPP=$4
IMPL=$5
PMODE=${6:-allow_fp32_to_fp16}

if [ -z "$ONNX" ] || [ -z "$OUTPUT" ]; then
    echo "用法: bash atc_convert.sh <onnx> <output> [soc_version] [aipp_cfg]"
    exit 1
fi

# 自动探测芯片型号, 如 Ascend310B4
if [ -z "$SOC" ]; then
    SOC=$(npu-smi info 2>/dev/null | grep -oE '310[BP][0-9]' | head -n1)
    if [ -n "$SOC" ]; then
        SOC="Ascend${SOC}"
    else
        SOC="Ascend310B4"
    fi
fi
case "$SOC" in
    Ascend*) ;;
    *) SOC="Ascend${SOC}" ;;
esac
echo "soc_version = $SOC"

CMD="atc --framework=5 --model=$ONNX --output=$OUTPUT \
    --input_format=NCHW --input_shape=\"images:1,3,640,640\" \
    --precision_mode=$PMODE \
    --soc_version=$SOC --log=error"

if [ -n "$AIPP" ]; then
    CMD="$CMD --insert_op_conf=$AIPP"
fi

if [ -n "$IMPL" ]; then
    CMD="$CMD --op_select_implmode=$IMPL"
fi

echo "$CMD"
eval $CMD
echo "转换完成: $OUTPUT.om"
