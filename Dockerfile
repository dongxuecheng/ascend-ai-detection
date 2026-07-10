# AI 视频安监检测系统
# 基于 Python 3.13 的 Slim 镜像（Debian bookworm）
# 运行时通过挂载 -v $(pwd):/app 挂载代码

# 基础镜像可通过 --build-arg BASE_IMAGE=<镜像名> 覆盖，便于在 Docker Hub 访问受限的环境使用镜像仓库
ARG BASE_IMAGE=python:3.13-slim-bookworm
FROM ${BASE_IMAGE}

# 避免交互式配置提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 换源：apt 使用阿里云镜像，pip 使用清华镜像
# 新镜像的源配置在 /etc/apt/sources.list.d/debian.sources，这里直接重写为传统 sources.list
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    printf '%s\n' \
        'deb http://mirrors.aliyun.com/debian bookworm main contrib non-free non-free-firmware' \
        'deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware' \
        'deb http://mirrors.aliyun.com/debian bookworm-updates main contrib non-free non-free-firmware' \
        > /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 120 && \
    pip config set global.retries 3

# 设置工作目录
WORKDIR /app

# 安装系统依赖与 Python 依赖
# build-essential / python3-dev 仅用于编译 cython-bbox、filterpy 等跟踪器依赖，完成后卸载以减小镜像
# 注意：tritonclient[all] 会限制 grpcio<1.68，但 stream/ 下生成的 gRPC 代码需要 grpcio>=1.78.1。
# 因此先安装 tritonclient[all]，再单独覆盖安装 grpcio==1.81.1 + grpcio-tools==1.81.1。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libturbojpeg0 \
        libgeos-c1v5 \
        build-essential \
        python3-dev \
    && pip install --no-cache-dir \
        pydantic \
        python-dotenv \
        pyyaml \
        requests \
        opencv-python-headless \
        numpy \
        shapely \
        protobuf \
        PyTurboJPEG \
        posix_ipc \
        openai \
        scipy \
        filterpy \
        lap \
        cython-bbox \
        tritonclient[all] \
    && apt-get purge -y --auto-remove build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    grpcio==1.81.1 \
    grpcio-tools==1.81.1

# 容器启动命令
CMD ["python", "main.py"]
