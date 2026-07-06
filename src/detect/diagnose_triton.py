"""
Triton 性能诊断脚本
直接调用 Triton 的 model metadata 和 stats API，排查推理慢的原因。
"""

import sys
import time

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def diagnose(url: str = "localhost:38001", model_name: str = "yolo"):
    print(f"\n{'='*60}")
    print(f"Triton 性能诊断 | {url} | {model_name}")
    print(f"{'='*60}\n")

    client = grpcclient.InferenceServerClient(url=url)

    # 1. 服务状态
    print("[1] 服务状态")
    print(f"    Server Live: {client.is_server_live()}")
    print(f"    Server Ready: {client.is_server_ready()}")
    print(f"    Model Ready: {client.is_model_ready(model_name)}")

    # 2. 模型配置
    print("\n[2] 模型配置")
    try:
        config = client.get_model_config(model_name, as_json=True)
        c = config["config"]
        print(f"    Platform: {c.get('platform', 'N/A')}")
        print(f"    Max Batch Size: {c.get('max_batch_size', 'N/A')}")
        print(f"    Instance Groups: {c.get('instance_group', [])}")
        print(f"    Dynamic Batching: {c.get('dynamic_batching', 'Disabled')}")
        print(f"    Optimization: {c.get('optimization', {})}")
    except Exception as e:
        print(f"    获取配置失败: {e}")

    # 3. 模型统计（关键：看推理时间分布）
    print("\n[3] 模型统计 (最近推理延迟)")
    try:
        stats = client.get_inference_statistics(model_name, as_json=True)
        model_stats = stats.get("model_stats", [])
        if model_stats:
            for ms in model_stats:
                bv = ms.get("batch_stats", [])
                iv = ms.get("inference_stats", {})
                print(f"    Version: {ms.get('version', 'N/A')}")
                print(f"    Count: {iv.get('success', {}).get('count', 0)}")
                # Triton 返回的是 ns，转成 ms
                exec_ns = iv.get('compute_infer', {}).get('ns', {})
                if exec_ns:
                    print(f"    Compute Infer: p50={exec_ns.get('p50',0)/1e6:.2f}ms "
                          f"p90={exec_ns.get('p90',0)/1e6:.2f}ms "
                          f"p99={exec_ns.get('p99',0)/1e6:.2f}ms")
                queue_ns = iv.get('queue', {}).get('ns', {})
                if queue_ns:
                    print(f"    Queue Wait:    p50={queue_ns.get('p50',0)/1e6:.2f}ms")
        else:
            print("    暂无统计（还没跑过推理）")
    except Exception as e:
        print(f"    获取统计失败: {e}")

    # 4. 实际跑一轮推理，分段计时
    print("\n[4] 实际推理计时")
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

    # warmup
    for _ in range(3):
        _infer(client, model_name, dummy)

    # 正式测 10 轮
    infer_times = []
    for _ in range(100):
        t0 = time.time()
        _infer(client, model_name, dummy)
        t1 = time.time()
        infer_times.append((t1 - t0) * 1000)

    print(f"    100 轮平均: {np.mean(infer_times):.2f}ms")
    print(f"    最快: {min(infer_times):.2f}ms | 最慢: {max(infer_times):.2f}ms")

    if np.mean(infer_times) > 100:
        print("\n    ⚠️ 警告: 推理延迟 > 100ms，RTX 4090 上不正常！")
        print("    请检查: (1) engine 是否真 FP16  (2) Triton 日志是否有 CPU fallback")

    print(f"\n{'='*60}\n")


def _infer(client, model_name, tensor):
    from tritonclient.grpc import InferInput, InferRequestedOutput
    inputs = [InferInput("image_input", tensor.shape, "FP32")]
    inputs[0].set_data_from_numpy(tensor)
    outputs = [InferRequestedOutput("final_boxes")]
    client.infer(model_name, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "localhost:38001"
    model = sys.argv[2] if len(sys.argv) > 2 else "yolo11_ensemble"
    diagnose(url, model)
