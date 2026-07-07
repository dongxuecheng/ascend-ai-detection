# AI 视频安监检测系统

基于 AI 的工业安全视频检测系统，通过 RTSP 拉取视频流，调用 SAM3 视觉大模型进行目标检测，在本地进行空间关系分析，实时上报违规告警。

## 功能特性

- **多任务复用**：同一视频流下的多个检测任务共享帧数据和 SAM3 推理结果
- **动态任务管理**：后台线程自动同步远程任务列表，支持热更新
- **异步告警上报**：独立队列上传报警截图，不阻塞视频分析主流程
- **OSD 可视化**：报警截图自动绘制检测框、违规高亮、围栏、时间戳

## 技术栈

- **Python 3.13**
- **Pydantic** + `python-dotenv`：配置管理
- **OpenCV** + **NumPy**：图像处理
- **shapely**：多边形几何计算
- **gRPC** + **Protocol Buffers**：视频流生命周期管理
- **PyTurboJPEG**：JPEG 解码
- **POSIX 共享内存**：零拷贝帧传输
- **SAM3**：外部视觉大模型推理服务

## 项目结构

```
AIDetection/
├── main.py                      # 入口启动器（将 src/ 加入 sys.path 后运行）
├── config/                      # 业务配置文件
│   ├── algorithms.yaml          # 算法码、SAM3 prompt、YOLO 模型配置
│   ├── coco.names               # YOLO COCO 80 类别名文件
│   └── yolo12s_face.names       # 自定义模型类别名文件（示例）
├── src/                         # 核心源代码
│   ├── main.py                  # 真正入口
│   ├── analyze/                 # 各类检测算法逻辑
│   │   ├── helmet.py
│   │   ├── glove.py
│   │   ├── zone.py
│   │   ├── single.py
│   │   ├── smoke.py
│   │   └── ...
│   ├── config/                  # Pydantic 配置
│   │   └── config.py
│   ├── core/                    # 编排与调度
│   │   ├── analyzer.py
│   │   ├── stream_worker.py
│   │   └── orchestrator.py
│   ├── detect/                  # 推理客户端
│   │   ├── triton_client_fast.py
│   │   ├── sam3.py
│   │   └── ...
│   ├── llm/                     # 大模型视觉复核
│   ├── stream/                  # RTSP 视频流捕获（RTSPClient：gRPC + 共享内存）
│   ├── task/                    # 任务/围栏/上传
│   └── utils/                   # 工具类
├── tests/                       # 本地测试脚本
│   └── local/
│       ├── test_local_image.py
│       └── test_triton.py
├── assets/                      # 测试图片与结果
│   ├── images/
│   └── results/
├── logs/                        # 运行日志
├── README.md
└── AGENTS.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install pydantic python-dotenv pyyaml requests opencv-python numpy shapely grpcio protobuf PyTurboJPEG posix_ipc
```

### 2. 环境变量配置（可选）

项目根目录已提供 `.env.example` 模板。首次部署时复制为 `.env` 并根据实际环境修改：

```bash
cp .env.example .env
# 编辑 .env 修改外部服务地址等配置
vim .env
```

`src/config/config.py` 会使用 `.env` 中的同名环境变量覆盖默认值。也可以通过 shell 直接导出：

```bash
export SAM3_URL=http://your-sam3-server:18002/predict
export TRITON_YOLO_URL=your-triton-server:38000
```

### 3. 运行

```bash
python main.py
```

## 核心架构

### 数据流

```
远程任务平台 ──HTTP──► TaskManager ──┐
                                    │
SAM3 推理服务 ◄──HTTP──┐            ▼
                       │      Orchestrator
RTSP 流媒体后端 ──gRPC/SHM──► StreamWorker
                       │         │
                       │    拉流 + SAM3 推理
                       │         │
                       │    合并 Prompt 一次请求
                       │         │
                       │    ┌────┼────┐
                       │    ▼    ▼    ▼
                       │  TaskA TaskB TaskC
                       │  分析   分析   分析
                       │    │    │    │
                       └────┴────┴────┘
                              │
                        EventUploader
                              │
                              ▼
                         告警平台
```

### 按流复用设计

同一 RTSP 地址下的多个任务（如一个摄像头同时配置安全帽 + 手套检测）：

- **只拉一次流**
- **只请求一次 SAM3**（合并所有任务的 Prompt）
- **多个任务共享 Box 列表**，各自独立分析
- **各自独立上报**

### 动态生命周期

```
TaskManager 每 10 秒同步远程任务
        │
        ▼
Orchestrator diff 任务列表
        │
   ┌────┼────┐
   ▼    ▼    ▼
 新增  删除  变更
 流    流    任务列表
```

## 配置说明

### 通过环境变量覆盖配置

配置统一集中在 `src/config/config.py` 中，默认值可通过同名环境变量覆盖，无需再维护环境子类。

例如自定义 SAM3 地址：

```bash
export SAM3_URL=http://your-sam3-server:18002/predict
python main.py
```

### 新增算法

1. 在 `config/algorithms.yaml` 中：
   - `supported_codes` 添加算法代码
   - `sam3_prompts` 添加 Prompt 映射
   - `fence_algorithms` 中配置是否需要围栏（如需要）
   - `algorithm_detectors` 中配置检测器类名

2. 在 `analyze/` 目录下新建分析模块

3. 在 `core/analyzer.py` 的 `analyze_for_task` 中注册分发逻辑

## 外部依赖服务

| 服务 | 地址 | 作用 |
|------|------|------|
| SAM3 推理服务 | `http://172.16.20.193:28005/predict` | 图像分割/检测 |
| 任务/告警平台 | `http://36.7.84.146:28801/open/api/operate/*` | 任务列表、围栏、流地址、上报 |
| RTSP 流媒体后端 | `127.0.0.1:50051` (gRPC) | C++ 实现的 RTSP 解码与推流服务 |

## 电子围栏

电子围栏数据通过 `TaskFence` 单例管理器自动同步：

- 后台线程每 30 秒轮询远程围栏 API
- 一个实例管理所有任务的围栏数据
- 支持多区域围栏（以 `||` 分隔）
- 围栏格式：`"x1#y1,x2#y2,||x3#y3,x4#y4,"`

## 日志

日志支持通过环境变量灵活配置：

```bash
# 输出 DEBUG 级别到控制台 + 文件
AIDETECTION_LOG_LEVEL=DEBUG python main.py

# 只写入文件，控制台静默
AIDETECTION_LOG_LEVEL=INFO AIDETECTION_LOG_DISABLE_CONSOLE=1 python main.py

# 自定义日志文件
AIDETECTION_LOG_FILE=/var/log/ai.log python main.py
```

日志格式：
```
2026-04-22 09:24:21 [INFO] [StreamWorker-1] stream_worker - StreamWorker 启动
```

## 演示脚本

```bash
# RTSP 统一客户端演示（包含 gRPC JPEG / 共享内存两种模式）
PYTHONPATH=src:src/stream python -m stream.remote_capture
```

## 安全注意事项

1. 生产部署时请确认内网 IP 的防火墙规则
2. `TaskManager` 和 SAM3 请求均无身份验证，建议在可信网络中使用
3. 共享内存路径 `/dev/shm/{stream_id}` 需确保 `stream_id` 不被恶意构造
4. 日志中可能包含设备凭据，正式环境需脱敏处理
