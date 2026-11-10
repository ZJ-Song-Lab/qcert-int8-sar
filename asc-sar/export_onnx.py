"""导出无 NMS 的 FP32 ONNX (用于 ATC 转换)。

要点(对应 Atlas 310B 部署约束):
- nms=False    : 剥离 NMS (310B 无 NonMaxSuppression 算子), 由 deploy/postprocess.py 在 CPU 端完成
- half=False   : 保持 FP32, FP16/INT8 分别交给 ATC / AMCT 处理, 保证两条链路公平可比
- dynamic=False: 静态 shape, 避免 ATC 动态 shape 调优
- batch=1      : 端侧单路推理
- opset=12     : ATC 对低版本 opset 兼容性更好
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--device", default="cpu", help="导出设备, cpu 可避免与训练抢占显存")
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        half=False,
        nms=False,
        simplify=True,
        opset=args.opset,
        dynamic=False,
        device=args.device,
    )
    print(f"ONNX 导出完成: {out}")


if __name__ == "__main__":
    main()
