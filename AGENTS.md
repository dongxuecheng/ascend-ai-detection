# AGENTS.md — AI 视频安监检测系统

> 本文档供 AI Coding Agent 阅读。项目主要使用中文注释，因此本文档使用中文撰写。

---

## 项目概述

本项目是一个基于 AI 的工业安全视频检测系统，主要功能包括：

- **安全帽检测**（算法代码 `"0"`）：检测人员是否佩戴安全帽
- **手套检测**（算法代码 `"99"`）：检测手部是否佩戴手套
- **电子围栏/区域入侵检测**：检测人员是否进入受限多边形区域

系统通过 RTSP 拉取视频流，调用外部 SAM3 推理服务进行目标分割/检测，然后在本地对检测结果进行空间关系分析，最终上报违规告警。

**当前状态：开发中（Work in Progress）**。核心的流式传输基础设施相对成熟，但顶层编排（`main.py`）、告警上报、日志、OSD 绘图等模块尚未完成或存在缺陷。

---

## 技术栈

- **语言**：Python 3.13
- **配置管理**：Pydantic `BaseModel` + `python-dotenv`
- **HTTP 客户端**：`requests`
- **图像处理**：OpenCV (`cv2`)、`numpy`
- **几何计算**：`shapely`（多边形包含检测）
- **视频流**：gRPC + 自定义 Protocol Buffers
- **JPEG 解码**：PyTurboJPEG (`turbojpeg`)
- **共享内存**：POSIX `mmap` + `ctypes` 信号量封装

### 外部依赖服务

| 服务 | 地址 | 作用 |
|------|------|------|
| SAM3 推理服务 | `http://172.16.20.193:28005/predict` | 图像分割/检测 |
| SAM3 目标检测 | `http://172.16.20.193:28005/predict-object` | 目标级推理 |
| 任务/告警平台 | `http://36.7.84.146:28801/open/api/operate/*` | 任务列表、围栏、流地址、上报 |
| RTSP 流媒体后端 | `127.0.0.1:50052` (gRPC) | C++ 实现的 RTSP 解码与推流服务 |

> **注意**：项目中没有 `requirements.txt`、`pyproject.toml` 或 `setup.py`，依赖需要手动安装。

---

## 项目结构

```
AIDetection/
├── main.py                     # 入口启动器（将 src/ 加入 sys.path 后运行）
├── config/                     # 业务配置文件
│   ├── algorithms.yaml         # 算法码、SAM3 prompt、YOLO 模型配置
│   ├── coco.names              # YOLO COCO 80 类别名文件
│   └── yolo12s_face.names      # 自定义模型类别名文件（示例）
├── src/                        # 核心源代码
│   ├── main.py                 # 真正入口
│   ├── config/
│   │   └── config.py           # Pydantic 配置，支持多环境切换
│   ├── core/
│   │   ├── analyzer.py         # 分析函数（算法分发、prompt 合并）
│   │   ├── stream_worker.py    # 按 RTSP 流的视频拉取与推理线程
│   │   └── orchestrator.py     # 任务生命周期管理与 worker 调度
│   ├── task/
│   │   ├── getTask.py          # TaskManager：后台线程同步远程任务列表
│   │   ├── getFence.py         # TaskFence 数据类
│   │   └── upload.py           # EventUploader：异步报警上传
│   ├── stream/
│   │   ├── remote_capture.py   # RTSPClient：统一 RTSP/gRPC/SHM 客户端（JPEG 与共享内存双模式）
│   │   ├── stream_service.proto    # gRPC 服务定义
│   │   ├── stream_service_pb2.py   # protoc 生成的 protobuf 代码
│   │   └── stream_service_pb2_grpc.py  # protoc 生成的 gRPC 代码
│   ├── analyze/
│   │   ├── helmet.py           # 安全帽检测逻辑
│   │   ├── glove.py            # 手套检测逻辑
│   │   ├── zone.py             # 区域入侵检测逻辑
│   │   ├── single.py           # 单人作业检测逻辑
│   │   └── smoke.py            # 吸烟检测逻辑
│   ├── detect/
│   │   ├── triton_client_fast.py   # YOLO Triton 高性能客户端
│   │   ├── sam3.py             # SAM3 推理封装
│   │   └── ...                 # 其他推理客户端
│   ├── llm/
│   │   └── llm.py              # 大模型视觉复核客户端
│   └── utils/
│       ├── obj.py              # Box 类：检测框几何运算
│       ├── filter.py           # 简单过滤函数
│       ├── rle.py              # RLE 编码/解码
│       ├── osd.py              # OSD 绘图工具
│       └── logger.py           # 日志工具
├── tests/
│   └── local/                  # 本地测试脚本
│       ├── test_local_image.py
│       └── test_triton.py
├── assets/                     # 测试图片与结果
│   ├── images/
│   └── results/
├── logs/                       # 运行日志
├── README.md
└── AGENTS.md
```

---

## 模块职责与交互

### 1. 配置层 (`config/`)

`src/config/config.py` 使用 Pydantic `BaseModel` 定义配置，支持通过环境变量 `AIDETECTION_ENV` 或 `.env` 文件切换环境。算法相关配置统一放在 `config/algorithms.yaml` 中，使用 YAML 格式以便阅读和注释。

当前注册的环境：
- `company`（默认）
- `mhwj`（梅花味精环境）

关键配置项：
- `SAM3_URL` / `SAM3_URL_OBJ`：外部 AI 推理地址
- `GET_TASK_URL`：获取检测任务列表
- `GET_FENCE_URL`：获取电子围栏
- `GET_RTSP_URL`：获取预览流地址
- `UPLOAD_URL`：告警上报接口
- `ALGORITHM_CODES`：`["8", "34"]`（从 `config/algorithms.yaml` 的 `supported_codes` 读取）
- `ALGORITHM_SAM3_PROMPT`：算法代码到 SAM3 文本提示词的映射
- `FENCE_ALGORITHMS`：需要使用围栏的算法码列表
- `USE_YOLO_BOXES`：需要把 YOLO 检测框合并到 SAM3 结果中的算法码列表
- `YOLO_MODEL_CONFIGS`：YOLO 模型配置，按 `name` 索引，支持 `label_map_file` 指向 `.names` 文件
- `ALGM_PRE_YOLO_MODEL_DETECT_CLASSES`：每个算法码在各 YOLO 模型上要预检的类别
- `ALGORITHM_DETECTORS`：算法码到检测器类名的映射
- `ALERT_DEDUP_CONFIG`：各算法码的报警去重参数
- `REQUEST_INTERVAL`：任务同步间隔（秒），默认 10

> `config/algorithms.yaml` 中所有按算法码配置的地方统一使用列表内联对象格式，每项必须含 `code` 字段，例如 `{code: '8'}`、`{code: '34', detectors: [ZoneDetector]}`。`yolo_model_configs` 则使用列表内联对象，每项必须含 `name` 字段。

### 2. 任务管理层 (`task/`)

- **`TaskManager`**（`getTask.py`）：后台守护线程定期从远程 API 拉取任务列表，按 `deviceId` 分组。
  - `get_tasks_by_device()`：按设备 ID 分组返回任务
  - `get_all_tasks()` / `get_task(task_id)` / `task_alive(task_id)`
- **`Task`**（`getTask.py`）：包含 17 个字段的数据类（设备信息、通道、算法码、围栏、凭据等）。
- **`TaskFence`**（`getFence.py`）：围栏数据类。

### 3. 视频流层 (`stream/`)

`RTSPClient`（`stream/remote_capture.py`）与 C++ gRPC RTSP 流媒体后端交互，支持两种帧传输模式：

| 模式 | 方法 | 适用场景 | 特点 |
|------|------|----------|------|
| JPEG-over-gRPC | `start_stream(..., use_shared_mem=False)` + `read()` | 调试/测试 | 完整 gRPC 客户端，支持轮询和流式读取，TurboJPEG 解码 |
| 共享内存 (SHM) | `start_stream(..., use_shared_mem=True)` + `read()` | **生产环境** | gRPC 仅做流生命周期管理，帧数据通过 POSIX 共享内存零拷贝传输 |

**SHM 帧布局**（由 C++ 后端定义）：
- 内存路径：`/dev/shm/{stream_id}`
- 8 槽位环形缓冲区，64 字节对齐
- 每帧头部包含：size、width、height、timestamp、channels、depth、step
- 像素数据直接映射为 `numpy` 数组
- 支持 POSIX 命名信号量（`/{stream_id}_notify`）阻塞等待，降级为自适应轮询

`RTSPClient` 提供统一的 API：`connect()` / `disconnect()`、`start_stream()` / `stop_stream()`、`read()`、`update_stream_url()`、`check_stream()`。`core/stream_worker.py` 默认使用 SHM 模式拉流。

### 4. 分析层 (`analyze/`)

所有分析模块接收 `Box` 对象列表（来自 SAM3 推理结果），返回违规目标列表。

- **`helmet.py`**：
  1. 筛选 `person`（面积 ≥ 1000，分数 ≥ 0.5）
  2. 筛选 `helmet` / `hard hat`（面积 ≥ 500，分数 ≥ 0.5）
  3. 筛选 `full head`（面积 ≥ 500，分数 ≥ 0.5）
  4. 将 head 与 person 上半部匹配（`cut_box('top', 0.5)`）
  5. 检查 helmet 是否与 head 相交（`iom > 0`）
  6. 返回未佩戴安全帽的 `[person, head]` 对

- **`glove.py`**：
  1. 筛选 `person`、`hand`、`glove`
  2. 保留与 person 相交的 hand/glove（`iom > 0`）
  3. 检查每个 hand 是否有 glove 重叠（`iom > 0.5`）
  4. 返回未戴手套的 `[hand]`

- **`zone.py`**：
  1. 将围栏坐标转为 `shapely.Polygon`
  2. 筛选 `person`
  3. 检查 person 的 `bottom_center` 是否在围栏多边形内
  4. 返回入侵的 `[person]`

- **`car.py`**：
  1. 将 `Box` 的 RLE mask 解码为 `shapely.Polygon`
  2. 筛选围栏内的 `truck bed`（凸包补全，避免车厢被遮挡成 U 型）
  3. 筛选含 `leg` 的完整 `person`
  4. 判断 person 与 truck bed 的 mask 重叠比例（IoA）是否超过阈值
  5. 返回违规的 `[person]` 及现场相关的 `[truck bed]`

- **`extinguisher.py`**：
  1. 分别过滤 `person`、`fire/flame`、`extinguisher`
  2. 当同时存在 person 和 fire/flame，且未检测到 extinguisher 时，判定为违规
  3. 返回火情目标 `[fire/flame]` 作为违规位置

### 5. 工具层 (`utils/`)

- **`obj.py`**：核心 `Box` 类。
  - 属性：`label`、`score`、`box`（4 坐标列表）、`mask`（RLE 字典或 None）
  - 位置函数：`top_left`、`top_right`、`bottom_left`、`bottom_right`、`center`、`top_center`、`bottom_center`、`left_center`、`right_center`
  - 几何运算：`area()`、`intersection(other)`、`iou(other)`、`iom(other)`、`fence_iou(fence)`、`fence_iom(fence)`
  - `cut_box(position, ratio)` / `expand_box(position, ratio)`：按方位裁剪/扩展检测框
  - `rle_to_mask()`：将 RLE 掩码解码为二进制 `numpy` 数组

- **`filter.py`**：`area_filter`、`score_filter`、`label_filter`

- **`rle.py`**：`binary_mask_to_rle(mask)`、`rle_to_binary_mask(...)` —— numpy 向量化 RLE 编解码

### 6. 入口 (`main.py`）

当前仅为桩代码：
1. 实例化 `TaskManager`
2. 等待 1 秒让后台线程完成首次同步
3. 打印按设备分组的所有任务

**尚未实现**：视频流拉取、SAM3 推理、分析判断、告警上报。

---

## 目标架构数据流

```
┌─────────────────────────────────────────────────────────────┐
│                           main.py                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ TaskManager │  │ RTSPClient  │  │ analyze.*           │ │
│  │ (task/...)  │  │ (stream/...)│  │ (helmet/glove/zone/car/extinguisher) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼────────────┘
          │                │                    │
    HTTP REST API    gRPC @ :50052      Box 列表处理
          │                │                    │
          ▼                ▼                    ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ 36.7.84.146  │  │ C++ Stream   │  │ SAM3         │
   │ (Task/Fence/ │  │ Service      │  │ 172.16.20.193│
   │  Upload)     │  │ (RTSP / SHM) │  │ :28005       │
   └──────────────┘  └──────────────┘  └──────────────┘
```

完整流程：
1. `TaskManager` 从远程 API 获取任务（设备 ID、通道、算法码、围栏坐标）。
2. 为每个任务构造 RTSP URL，通过 `RTSPClient` 打开视频流（默认启用共享内存）。
3. 从共享内存获取帧（零拷贝）。
4. 将帧和算法对应的提示词发送给 SAM3 推理服务。
5. SAM3 返回检测框（`person`、`head`、`helmet`、`hand`、`glove` 等）。
6. `analyze.*` 模块对 Box 列表进行过滤和空间推理，识别违规。
7. 将违规结果上报至告警 API（`upload.py` — 尚未实现）。

---

## 已知问题与缺陷

| 问题 | 位置 | 严重程度 | 说明 |
|------|------|----------|------|
| `STREAM_API_URL` 未定义 | `src/task/getTask.py:65` | 🔴 运行时错误 | `_get_stream_url` 中使用了未定义的变量 |
| 导入路径错误 | `src/task/getTask.py:4` | 🔴 运行时错误 | `from config import config` 应为 `from config.config import config` |
| 导入路径错误 | `src/utils/filter.py:2` | 🔴 运行时错误 | `from utils.box import Box` 应为 `from utils.obj import Box` |
| `ny2` 未定义 | `src/utils/obj.py:39` | 🔴 运行时错误 | `cut_box('bottom')` 分支使用了未定义的 `ny2` |
| `ny1` 未定义 | `src/utils/obj.py:66` | 🔴 运行时错误 | `expand_box(...)` 的默认分支使用了未定义的 `ny1` |
| `rle_to_binary_mask` 索引风险 | `src/utils/rle.py:44-45` | 🟡 逻辑错误 | 在展平数组上写入后裁剪二维切片，可能索引越界 |
| 文件为空 | `src/task/upload.py` | 🟡 未实现 | 告警上报功能缺失 |
| 文件为空 | `src/utils/logger.py` | 🟡 未实现 | 日志工具缺失 |
| 文件为空 | `src/utils/osd.py` | 🟡 未实现 | OSD 绘图工具缺失 |
| 文件为空 | `src/stream/stream.py` | 🟡 未实现 | — |
| `main.py` 不完整 | `main.py` | 🟡 开发中 | 仅打印任务列表，未串联完整流程 |
| YOLO 预检测空结果跳过任务 | `src/core/stream_worker.py` | 🟢 已修复 | 当 YOLO 未检测到目标时仍继续进入分析流程，仅跳过 SAM3 调用，以支持时间累计类算法 |
| SAM3/YOLO 空结果跳过分析 | `src/core/stream_worker.py` | 🟢 已修复 | 即使当前帧未检测到任何目标，也会对每个 ready 任务调用分析器，确保时间累计状态被更新 |

---

## 构建与运行

### 安装依赖

项目没有依赖清单文件，需手动安装以下包：

```bash
pip install pydantic python-dotenv pyyaml requests opencv-python numpy shapely grpcio protobuf PyTurboJPEG
```

### 运行入口

```bash
python main.py
```

### 运行演示脚本

```bash
# RTSP 统一客户端演示（包含 JPEG / SHM 两种模式示例）
# 需要将 src/ 与 src/stream/ 加入 PYTHONPATH
PYTHONPATH=src:src/stream python -m stream.remote_capture
```

### 重新生成 gRPC 代码

当 `src/stream/stream_service.proto` 发生变更时：

```bash
python -m grpc_tools.protoc \
  --python_out=. \
  --grpc_python_out=. \
  -I. \
  src/stream/stream_service.proto
```

> 当前生成的代码基于 protobuf v6.31.1、grpcio v1.78.1。

---

## 代码风格与开发约定

1. **注释语言**：使用中文注释和文档字符串。
2. **导入风格**：使用绝对导入，但当前部分文件存在相对导入错误（见已知问题）。
3. **类/函数命名**：
   - 类名使用大驼峰（如 `TaskManager`、`RTSPClient`、`Box`）
   - 函数/方法名使用小写下划线（如 `get_tasks_by_device`、`point_in_fence`）
4. **配置扩展**：新增环境时，在 `src/config/config.py` 的 `CONFIG_MAP` 中注册新的 `BaseConfig` 子类即可，无需修改其他逻辑。
5. **数据类**：任务相关对象使用普通类手动定义 `__init__` 和 `__str__`，未使用 `@dataclass`。
6. **线程安全**：`TaskManager` 使用 `threading.Lock()` 保护任务字典的读写。

---

## 测试

**当前项目没有任何测试文件**（无 `test_*.py`、`*_test.py`、`conftest.py`）。

建议补充的测试方向：
- `Box` 的几何运算（IoU、IoM、cut_box、expand_box）
- RLE 编解码的 round-trip 测试
- `TaskManager` 的 Mock HTTP 测试
- `RTSPClient` 的 Mock gRPC / 共享内存测试
- `analyze.*` 模块的边界场景测试（如部分遮挡、多人场景）

---

## 安全注意事项

1. **硬编码 IP 地址**：多处存在内网 IP 硬编码（`172.16.20.193`、`36.7.84.146`、`127.0.0.1`），生产部署时需确认网络可达性和防火墙规则。
2. **无身份验证**：`TaskManager` 向远程 API 发送的 HTTP 请求以及 SAM3 推理请求均未见认证机制。
3. **共享内存路径**：`/dev/shm/{stream_id}` 的命名需确保 `stream_id` 不会被恶意构造以访问其他共享内存段。
4. **日志脱敏**：当前日志/打印中可能包含设备凭据（`videoName`、`videoPassword`），正式环境需做脱敏处理。
