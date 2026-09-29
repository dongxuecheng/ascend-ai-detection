#!/usr/bin/env python3
"""项目入口启动器：将 src/ 加入 Python 路径后调用主程序"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
STREAM_DIR = os.path.join(SRC_DIR, "stream")
# src/ 供包引用（stream.remote_capture 等）
sys.path.insert(0, SRC_DIR)
# stream/ 也加入路径，因为 gRPC 生成的代码使用绝对导入（stream_service_pb2 / _grpc）
sys.path.insert(0, STREAM_DIR)

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
