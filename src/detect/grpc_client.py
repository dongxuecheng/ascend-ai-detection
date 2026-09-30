# Copyright (C) 2025 YIQISOFT
#
# SPDX-License-Identifier: Apache-2.0
#
"""Ascend Triton gRPC 图片联调，复用生产客户端的 BGR 输入和原图坐标解析。"""

import argparse
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detect.triton_client_fast import YOLOTritonFast


def main():
    parser = argparse.ArgumentParser(description="Ascend Triton YOLO gRPC client")
    parser.add_argument("--url", default="localhost:54246", help="Triton gRPC 地址")
    parser.add_argument("--image", default="input.jpg", help="输入图片")
    parser.add_argument("--model", default="YOLO26_DET_PRE_YUV_ENSEMBLE")
    parser.add_argument("--save", default="output_detection.jpg")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"无法读取图片: {args.image}")
    client = YOLOTritonFast(url=args.url, model_name=args.model, protocol="grpc", warmup=False)
    try:
        start = time.perf_counter()
        boxes = client.predict(image)
        print(f"Detected {len(boxes)} objects in {(time.perf_counter() - start) * 1000:.2f} ms")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{box.label}: {box.score:.2f}", (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if not cv2.imwrite(args.save, image):
            raise OSError(f"无法保存结果: {args.save}")
        print(f"Result image saved to {args.save}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
