#!/usr/bin/env python3
"""Benchmark Triton inference via gRPC, HTTP and system shared memory.

Run from the workspace directory:
    python3 triton_client/benchmark.py --protocol grpc
    python3 triton_client/benchmark.py --protocol http
    python3 triton_client/benchmark.py --protocol shm
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from triton_client import TritonClient
from utils.image_formats import bgr_to_nv12


def load_image(path: str):
    """Load an unbatched BGR image."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def run_benchmark(
    protocol: str,
    url: str,
    model_name: str,
    image_path: str,
    count: int,
    warmup: int,
):
    img_np = load_image(image_path)
    input_name = "YUV" if model_name.endswith("_YUV_ENSEMBLE") else "IMAGE"
    if input_name == "YUV":
        img_np = bgr_to_nv12(img_np)[..., None]

    outputs = [
        "NUM_DETS",
        "DETECTION_BOXES",
        "DETECTION_SCORES",
        "DETECTION_CLASSES",
    ]

    output_specs = {
        "NUM_DETS": ([1], "int32"),
        "DETECTION_BOXES": ([300, 4], "float32"),
        "DETECTION_SCORES": ([300], "float32"),
        "DETECTION_CLASSES": ([300], "int32"),
    }

    print(f"Protocol: {protocol}")
    print(f"URL:      {url}")
    print(f"Model:    {model_name}")
    print(f"Image:    {image_path}")
    print(f"Warmup:   {warmup}")
    print(f"Requests: {count}")

    with TritonClient(url=url, protocol=protocol) as client:
        if not client.is_server_ready():
            raise RuntimeError("Triton server is not ready")
        if not client.is_model_ready(model_name):
            raise RuntimeError(f"Model {model_name} is not ready")

        infer_kwargs = {}
        if protocol == "shm":
            infer_kwargs["output_specs"] = output_specs

        inputs = {input_name: img_np}

        # Warmup.
        print("\nWarming up ...")
        for _ in range(warmup):
            client.infer(model_name, inputs, outputs, **infer_kwargs)

        # Benchmark.
        print("Benchmarking ...")
        latencies = []
        for i in range(count):
            t0 = time.perf_counter()
            client.infer(model_name, inputs, outputs, **infer_kwargs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    total_time_s = latencies.sum() / 1000.0

    print("\n========== Benchmark Results ==========")
    print(f"Protocol:       {protocol}")
    print(f"Requests:       {count}")
    print(f"Total time:     {total_time_s:.3f} s")
    print(f"Throughput:     {count / total_time_s:.2f} req/s")
    print(f"Mean latency:   {np.mean(latencies):.2f} ms")
    print(f"Median latency: {np.median(latencies):.2f} ms")
    print(f"P90 latency:    {np.percentile(latencies, 90):.2f} ms")
    print(f"P95 latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 latency:    {np.percentile(latencies, 99):.2f} ms")
    print(f"Min latency:    {np.min(latencies):.2f} ms")
    print(f"Max latency:    {np.max(latencies):.2f} ms")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton inference protocols"
    )
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
        help="Input image path (default: images/bus.jpg)",
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
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of inference requests (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup requests (default: 10)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists() and args.image == "images/bus.jpg":
        image_path = Path("workspace") / args.image

    url = args.grpc_url if args.protocol == "grpc" else args.http_url

    run_benchmark(
        protocol=args.protocol,
        url=url,
        model_name=args.model,
        image_path=str(image_path),
        count=args.count,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
