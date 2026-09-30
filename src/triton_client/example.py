#!/usr/bin/env python3
"""Example: invoke a Triton ensemble with gRPC, HTTP or shared memory.

Usage:
    python3 triton_client/example.py --protocol grpc
    python3 triton_client/example.py --protocol http
    python3 triton_client/example.py --protocol shm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from triton_client import TritonClient
from utils.image_formats import bgr_to_nv12


def main():
    parser = argparse.ArgumentParser(description="Triton client wrapper example")
    parser.add_argument(
        "--protocol",
        choices=["grpc", "http", "shm"],
        default="grpc",
        help="Inference protocol",
    )
    parser.add_argument(
        "--model",
        default="YOLO26_DET_PRE_YUV_ENSEMBLE",
        help="Model or ensemble name",
    )
    parser.add_argument(
        "--image",
        default="images/bus.jpg",
        help="Input image path (default: images/bus.jpg, relative to workspace dir)",
    )
    parser.add_argument(
        "--grpc-url",
        default="localhost:54246",
        help="gRPC server URL",
    )
    parser.add_argument(
        "--http-url",
        default="localhost:54245",
        help="HTTP/SHM server URL",
    )
    args = parser.parse_args()

    url = args.grpc_url if args.protocol == "grpc" else args.http_url

    image_path = Path(args.image)
    if not image_path.exists() and args.image == "images/bus.jpg":
        image_path = Path("workspace") / args.image
    img_np = cv2.imread(str(image_path))  # BGR [H, W, 3]
    if img_np is None:
        raise ValueError(f"Cannot read image: {image_path}")
    input_name = "YUV" if args.model.endswith("_YUV_ENSEMBLE") else "IMAGE"
    if input_name == "YUV":
        img_np = bgr_to_nv12(img_np)[..., None]

    outputs = [
        "NUM_DETS",
        "DETECTION_BOXES",
        "DETECTION_SCORES",
        "DETECTION_CLASSES",
    ]

    # Shared memory requires explicit output buffer specs.
    output_specs = {
        "NUM_DETS": ([1], "int32"),
        "DETECTION_BOXES": ([300, 4], "float32"),
        "DETECTION_SCORES": ([300], "float32"),
        "DETECTION_CLASSES": ([300], "int32"),
    }

    print(f"Protocol: {args.protocol}")
    print(f"URL:      {url}")
    print(f"Model:    {args.model}")
    print(f"Image:    {args.image}")

    with TritonClient(url=url, protocol=args.protocol) as client:
        print(f"Server ready: {client.is_server_ready()}")
        print(f"Model ready:  {client.is_model_ready(args.model)}")

        infer_kwargs = {}
        if args.protocol == "shm":
            infer_kwargs["output_specs"] = output_specs

        result = client.infer(
            model_name=args.model,
            inputs={input_name: img_np},
            outputs=outputs,
            **infer_kwargs,
        )

    print(f"num_dets: {result['NUM_DETS']}")
    print(f"top 5 boxes:  {result['DETECTION_BOXES'][:5]}")
    print(f"top 5 scores: {result['DETECTION_SCORES'][:5]}")
    print(f"top 5 classes:{result['DETECTION_CLASSES'][:5]}")


if __name__ == "__main__":
    main()
