"""训练 SAR 舰船检测基线模型 (Ultralytics YOLO)。"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ssdd/ssdd.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="基线权重, 可选 yolo11s.pt / yolo11m.pt 作教师")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="ssdd_yolo11n")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="开启 AMP(会触发联网下载 yolo* 权重做校验)")
    ap.add_argument("--cache", action="store_true", help="缓存数据集到内存/磁盘")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        amp=args.amp,
        cache=args.cache,
        patience=50,
        close_mosaic=10,
    )
    # 在测试集上评估(输出 mAP50 / mAP50-95)
    metrics = model.val(data=args.data, split="test")
    print(f"[test] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
