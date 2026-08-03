#!/usr/bin/env python3
"""
本地 Triton YOLO 推理测试脚本

用法示例：
    # 默认使用 config 中的 yolo11_plan 测试单张图
    python tests/local/test_triton.py -i assets/images/test.jpg

    # 指定模型、尺寸、关闭共享内存
    python tests/local/test_triton.py -i assets/images/test.jpg \
        --url localhost:38000 \
        --model yolo11_plan \
        --input-size 640 \
        --no-shm

    # 多轮压测并保存结果图
    python tests/local/test_triton.py -i assets/images/test.jpg -n 100 --save results/triton_test.jpg

    # 只检测 person 类别
    python tests/local/test_triton.py -i assets/images/test.jpg --classes 0
"""

import os
import sys
import argparse
import time
import statistics
from typing import List, Optional

# 把 src/ 加入 sys.path，确保能 import detect/ utils/ config/
# 当前文件位于 tests/local/，向上回退两级才是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
# stream/ 下的 gRPC 生成代码使用绝对导入，需把 stream/ 目录也加入路径
sys.path.insert(0, os.path.join(SRC_DIR, "stream"))

import cv2
import numpy as np

from detect.triton_client_fast import YOLOTritonFast
from utils.logger import setup_logger

logger = setup_logger("test_triton")


def parse_classes(raw: Optional[str]) -> Optional[List[int]]:
    """解析 --classes 参数为整数类别 ID 列表"""
    if raw is None or raw.strip() == "":
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--classes 必须是逗号分隔的整数: {raw}") from e


def draw_boxes(frame: np.ndarray, boxes) -> np.ndarray:
    """在图像上绘制检测框和标签"""
    canvas = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.box)
        color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_text = f"{b.label} {b.score:.2f}"
        cv2.putText(
            canvas, label_text, (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Triton YOLO 本地测试")
    parser.add_argument("-i", "--image", required=True, help="输入图片路径")
    parser.add_argument("--url", default="localhost:38000", help="Triton HTTP 地址")
    parser.add_argument("--model", default="yolo26_ensemble", help="模型名（Triton ensemble）")
    parser.add_argument("--input-size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument(
        "--output-format", default="yolo_v8_v11",
        choices=["yolo_v8_v11", "yolo_v12_end2end"],
        help="模型输出格式（已集成到 Triton ensemble，此参数仅保留兼容）"
    )
    parser.add_argument("--classes", type=str, default=None, help="只检测指定类别 ID，逗号分隔，如 0,1,2")
    parser.add_argument("-n", "--iterations", type=int, default=1, help="推理轮数（用于压测）")
    parser.add_argument("--no-shm", action="store_true", help="关闭共享内存，使用 HTTP numpy 传输")
    parser.add_argument("--save", default=None, help="保存绘制结果图的路径")
    parser.add_argument("--input-name", default="raw_image", help="输入 tensor 名（ensemble 固定为 raw_image）")
    parser.add_argument("--output-name", default="detection_boxes", help="输出 tensor 名（ensemble 固定，此参数仅保留兼容）")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        logger.error(f"图片不存在: {args.image}")
        sys.exit(1)

    frame = cv2.imread(args.image)
    if frame is None:
        logger.error(f"无法读取图片: {args.image}")
        sys.exit(1)

    classes = parse_classes(args.classes)

    logger.info(
        f"初始化 Triton 客户端 | url={args.url} model={args.model} "
        f"input_size={args.input_size} shm={not args.no_shm}"
    )

    client = YOLOTritonFast(
        url=args.url,
        model_name=args.model,
        input_size=args.input_size,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        input_name=args.input_name,
        output_name=args.output_name,
        output_format=args.output_format,
        use_shared_memory=not args.no_shm,
        warmup=True,
    )

    # 第一次推理：打印详细耗时
    logger.info(f"开始推理 | image={args.image}, shape={frame.shape}, iterations={args.iterations}")
    t_start = time.time()
    boxes = client.predict(frame, classes=classes)
    t_first = (time.time() - t_start) * 1000

    logger.info(f"第 1 轮推理完成 | 耗时={t_first:.1f}ms | 检测到 {len(boxes)} 个目标")
    for b in boxes:
        logger.info(f"  {b}")

    # 多轮压测
    latencies = [t_first]
    if args.iterations > 1:
        for idx in range(2, args.iterations + 1):
            t0 = time.time()
            _ = client.predict(frame, classes=classes)
            latency = (time.time() - t0) * 1000
            latencies.append(latency)

        avg = statistics.mean(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        logger.info(
            f"压测完成 | 轮数={len(latencies)} | "
            f"avg={avg:.1f}ms min={min_lat:.1f}ms max={max_lat:.1f}ms"
        )

    # 保存结果图
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        result_frame = draw_boxes(frame, boxes)
        cv2.imwrite(args.save, result_frame)
        logger.info(f"结果图已保存: {args.save}")

    client.close()


if __name__ == "__main__":
    main()
