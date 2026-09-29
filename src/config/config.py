import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, List

import yaml
# 自动加载项目根目录或当前目录下的 .env 文件
load_dotenv()


def _load_algorithms_yaml() -> dict:
    """
    读取算法主配置 config/algorithms.yaml 和模型配置 config/models.yaml，合并后返回。

    拆分目的：
      - algorithms.yaml：算法码、SAM3 prompt、检测器映射、间隔、去重等业务规则配置。
      - models.yaml：YOLO 模型、分类模型等模型推理参数，便于独立维护。

    合并规则：models.yaml 中的同名键会覆盖 algorithms.yaml 中的值（当前模型配置已完全迁出，
    实际无冲突）。models.yaml 为可选文件，缺失时仅使用 algorithms.yaml。
    """
    # src/config/config.py -> 项目根目录 -> config/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(project_root, "config")
    main_path = os.path.join(config_dir, "algorithms.yaml")
    models_path = os.path.join(config_dir, "models.yaml")

    if not os.path.isfile(main_path):
        raise FileNotFoundError(
            f"算法配置文件不存在: {main_path}\n"
            f"请在项目根目录 config/ 下创建 algorithms.yaml 并填写算法配置。"
        )

    with open(main_path, "r", encoding="utf-8") as f:
        main_data = yaml.safe_load(f)
    if not isinstance(main_data, dict):
        raise ValueError(f"algorithms.yaml 根节点必须是字典/对象，实际类型: {type(main_data).__name__}")

    models_data = {}
    if os.path.isfile(models_path):
        with open(models_path, "r", encoding="utf-8") as f:
            models_data = yaml.safe_load(f) or {}
        if not isinstance(models_data, dict):
            raise ValueError(f"models.yaml 根节点必须是字典/对象，实际类型: {type(models_data).__name__}")

    merged = dict(main_data)
    merged.update(models_data)
    return merged


def _load_label_map_from_file(label_map_file: str) -> Dict[str, str]:
    """
    从 YOLO names 文件加载 label_map。
    每行一个类别名，空行和以 # 开头的行忽略，行号即为类别 ID。
    """
    path = label_map_file
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"label_map 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        names = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    return {str(i): name for i, name in enumerate(names)}


# 启动时加载一次，供 BaseConfig 使用
_ALGO_CONFIG = _load_algorithms_yaml()

# 算法默认检测间隔（秒），在模块级计算以避免 Pydantic 字段默认值中无法引用其他字段的问题
_DEFAULT_ALGORITHM_INTERVAL = float(_ALGO_CONFIG.get("default_algorithm_interval", 1.0))
_ALGORITHM_INTERVALS_RAW = _ALGO_CONFIG.get("algorithm_intervals", [])
_ALGORITHM_INTERVALS = {
    str(_item.get("code")): float(_item.get("interval", _DEFAULT_ALGORITHM_INTERVAL))
    for _item in _ALGORITHM_INTERVALS_RAW
    if isinstance(_item, dict) and "code" in _item
}

# 算法码 -> VL 大模型二次复核配置
# 结构示例：{code: '58', enabled: true, module: height_work}
_ALGORITHM_VL_CONFIG_RAW = _ALGO_CONFIG.get("algorithm_vl_config", [])
_ALGORITHM_VL_CONFIG = {
    str(_item.get("code")): {
        "enabled": bool(_item.get("enabled", False)),
        "module": str(_item.get("module", "")),
    }
    for _item in _ALGORITHM_VL_CONFIG_RAW
    if isinstance(_item, dict) and "code" in _item
}


class BaseConfig(BaseModel):
    """公共配置基类，各环境共享的默认值在此定义。

    所有外部服务接口地址统一在此管理，避免在业务代码中硬编码。
    各环境子类只需覆盖与默认值不同的字段即可。
    """

    # ---- SAM3 推理服务 ----
    # 支持通过环境变量覆盖，便于 Docker / K8s 部署时注入
    SAM3_URL: str = os.getenv("SAM3_URL", "http://192.168.100.75:18002/predict")
    # 兼容已有部署变量；Ascend 服务只有 /predict，不再区分小目标接口。
    SAM3_URL_OBJ: str = os.getenv("SAM3_URL_OBJ") or SAM3_URL
    SAM3_TIMEOUT_SECONDS: float = float(os.getenv("SAM3_TIMEOUT_SECONDS", "30"))

    # ---- 任务/告警平台 ----
    # 上传报警记录的接口地址
    UPLOAD_URL: str = os.getenv("UPLOAD_URL", "http://192.168.100.73/open/api/operate/upload")
    # 获取电子围栏的接口地址
    GET_FENCE_URL: str = os.getenv("GET_FENCE_URL", "http://192.168.100.73/open/api/operate/fence")
    # 获取任务列表的接口地址
    GET_TASK_URL: str = os.getenv("GET_TASK_URL", "http://192.168.100.73/open/api/operate/taskList")
    # 获取流地址的接口地址
    GET_RTSP_URL: str = os.getenv("GET_RTSP_URL", "http://192.168.100.73/open/api/operate/previewURLs")

    # 任务同步间隔（秒），docker-compose 中可通过 REQUEST_INTERVAL 覆盖
    REQUEST_INTERVAL: int = int(os.getenv("REQUEST_INTERVAL", "180"))

    # ---- gRPC RTSP 流媒体后端（C++ 服务）----
    # 主服务端口（共享内存/SHM 模式），业务代码实例化 RTSPClient 时从此读取并传入
    STREAM_SERVER_ADDRESS: str = os.getenv("STREAM_SERVER_ADDRESS", "192.168.100.74:50051")
    # 流恢复：两次重建之间的最小冷却（秒）
    STREAM_RECOVERY_COOLDOWN_SEC: float = float(os.getenv("STREAM_RECOVERY_COOLDOWN_SEC", "10.0"))

    # 违章原图本地保存根目录
    VIOLATION_IMAGE_SAVE_DIR: str = os.getenv("VIOLATION_IMAGE_SAVE_DIR", "/mnt/yolo/images")

    # ---- Triton YOLO 推理服务 ----
    # HTTP 端口（共享内存/HTTP 调用时使用）
    TRITON_YOLO_URL: str = os.getenv("TRITON_YOLO_URL", "192.168.100.74:38000")
    # gRPC 端口（推荐，避免 gevent/libev 文件描述符开销）
    TRITON_YOLO_GRPC_URL: str = os.getenv("TRITON_YOLO_GRPC_URL", "192.168.100.74:38001")
    # 分类模型默认共用同一 Triton 服务的 gRPC 端口
    TRITON_CLASSIFIER_GRPC_URL: str = os.getenv("TRITON_CLASSIFIER_GRPC_URL", "192.168.100.74:38001")

    # ---- VL 大模型配置 ----
    VL_ENABLED: bool = os.getenv("VL_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    VL_API_URL: str = os.getenv("VL_API_URL", "http://192.168.100.75:18000/v1")
    VL_API_KEY: str = os.getenv("VL_API_KEY", "EMPTY")
    VL_MODEL: str = os.getenv("VL_MODEL", "/models/qwen3-vl-4b")
    # 是否开启大模型思考链（reasoning/thinking），默认关闭以加快响应
    VL_ENABLE_THINKING: bool = os.getenv("VL_ENABLE_THINKING", "false").lower() in ("1", "true", "yes", "on")

    # ---- 算法相关配置：必须从 algorithms.yaml 读取，无默认值 ----
    ALGORITHM_CODES: List[str] = _ALGO_CONFIG["supported_codes"]

    # 可用的 GPU 编码（GPU ID）列表，StreamWorker 创建时按轮询方式依次使用
    GPU_CODES: List[int] = [
        int(_code)
        for _code in _ALGO_CONFIG.get("gpu_codes", [0])
        if _code is not None
    ] or [0]

    # 解析 code_descriptions，用于本地保存时按类别分目录
    _code_desc_raw: List[dict] = _ALGO_CONFIG.get("code_descriptions", [])
    CODE_DESCRIPTIONS: Dict[str, str] = {
        str(_item.get("code")): str(_item.get("descriptions", ""))
        for _item in _code_desc_raw
        if isinstance(_item, dict) and "code" in _item
    }

    # 解析 sam3_prompts（新格式：列表，每项含 code/prompts/return_mask）
    _sam3_prompts_raw: List[dict] = _ALGO_CONFIG.get("sam3_prompts", [])
    ALGORITHM_SAM3_PROMPT: Dict[str, List[str]] = {}
    ALGORITHM_SAM3_RETURN_MASK: Dict[str, bool] = {}
    for _item in _sam3_prompts_raw:
        if isinstance(_item, dict) and "code" in _item:
            _code = str(_item["code"])
            ALGORITHM_SAM3_PROMPT[_code] = _item.get("prompts", [])
            ALGORITHM_SAM3_RETURN_MASK[_code] = _item.get("return_mask", False)

    # 解析 sam3_url_groups（按 URL 分组，codes 为算法码列表）。
    # 未在分组中配置的算法码，由调用方回退到 config.SAM3_URL_OBJ。
    _sam3_url_groups_raw: List[dict] = _ALGO_CONFIG.get("sam3_url_groups", [])
    ALGORITHM_SAM3_URL: Dict[str, str] = {}
    for _group in _sam3_url_groups_raw:
        if not isinstance(_group, dict):
            continue
        _group_url = str(_group.get("url", ""))
        _group_codes = _group.get("codes", [])
        if not _group_url or not isinstance(_group_codes, list):
            continue
        for _code in _group_codes:
            if _code is not None:
                ALGORITHM_SAM3_URL[str(_code)] = _group_url

    # 解析 fence_algorithms（统一格式：列表，每项含 code）
    _fence_raw: List[dict] = _ALGO_CONFIG.get("fence_algorithms", [])
    FENCE_ALGORITHMS: List[str] = [
        str(_item.get("code"))
        for _item in _fence_raw
        if isinstance(_item, dict) and "code" in _item
    ]

    # 解析 yolo_pre_detect（列表，每项含 code + model_name -> classes）
    _yolo_raw: List[dict] = _ALGO_CONFIG.get("yolo_pre_detect", [])
    ALGM_PRE_YOLO_MODEL_DETECT_CLASSES: Dict[str, Dict[str, List[str]]] = {}
    for _item in _yolo_raw:
        if not isinstance(_item, dict) or "code" not in _item:
            continue
        _code = str(_item["code"])
        ALGM_PRE_YOLO_MODEL_DETECT_CLASSES[_code] = {
            str(_model_name): list(_classes)
            for _model_name, _classes in _item.items()
            if _model_name != "code"
        }

    # 解析 yolo_model_configs（统一格式：列表，每项含 name + 模型参数）
    # 支持通过 label_map_file 指定 YOLO names 文件，自动转换为 label_map 字典
    _raw_yolo_configs: List[dict] = _ALGO_CONFIG.get("yolo_model_configs", [])
    YOLO_MODEL_CONFIGS: Dict[str, dict] = {}
    for _model_cfg in _raw_yolo_configs:
        if not isinstance(_model_cfg, dict) or "name" not in _model_cfg:
            continue
        _model_name = str(_model_cfg["name"])
        _cfg = {k: v for k, v in _model_cfg.items() if k != "name"}
        _label_map_file = _cfg.pop("label_map_file", None)
        if _label_map_file:
            _cfg["label_map"] = _load_label_map_from_file(_label_map_file)
        YOLO_MODEL_CONFIGS[_model_name] = _cfg

    # 哪些算法代码需要把 YOLO 检测框合并到 SAM3 结果中参与分析（统一格式：列表，每项含 code）
    _yolo_boxes_raw: List[dict] = _ALGO_CONFIG.get("use_yolo_boxes", [])
    USE_YOLO_BOXES: List[str] = [
        str(_item.get("code"))
        for _item in _yolo_boxes_raw
        if isinstance(_item, dict) and "code" in _item
    ]

    # 算法代码 -> 检测器类名列表（在 core/analyzer.py 中动态实例化）
    _detector_raw: List[dict] = _ALGO_CONFIG.get("algorithm_detectors", [])
    ALGORITHM_DETECTORS: Dict[str, List[str]] = {
        str(_item.get("code")): [str(cls) for cls in _item.get("detectors", [])]
        for _item in _detector_raw
        if isinstance(_item, dict) and "code" in _item
    }

    # 算法代码 -> 检测间隔（秒），StreamWorker 用其控制任务分析频率
    # 未在 algorithms.yaml 中配置的算法码，使用 DEFAULT_ALGORITHM_INTERVAL
    DEFAULT_ALGORITHM_INTERVAL: float = _DEFAULT_ALGORITHM_INTERVAL
    ALGORITHM_INTERVALS: Dict[str, float] = _ALGORITHM_INTERVALS

    # 哪些算法码必须拉取全部帧（不能仅拉关键帧）
    # 一个视频流上只要关联了任一 full_frame 算法，StreamWorker 就会以 only_key_frames=False 打开流
    _full_frame_raw: List[dict] = _ALGO_CONFIG.get("full_frame_algorithms", [])
    FULL_FRAME_ALGORITHMS: List[str] = [
        str(_item.get("code"))
        for _item in _full_frame_raw
        if isinstance(_item, dict) and "code" in _item
    ]

    # 算法代码 -> VL 大模型二次复核配置
    # 未在 algorithms.yaml 中配置的算法码，默认不启用
    # 每项为 {enabled: bool, module: str}
    ALGORITHM_VL_CONFIG: Dict[str, dict] = _ALGORITHM_VL_CONFIG

    # 分类模型配置（统一格式：列表，每项含 name + 模型参数）
    _raw_cls_configs: List[dict] = _ALGO_CONFIG.get("classification_configs", [])
    CLASSIFICATION_MODEL_CONFIGS: Dict[str, dict] = {}
    for _cls_cfg in _raw_cls_configs:
        if not isinstance(_cls_cfg, dict) or "name" not in _cls_cfg:
            continue
        _cls_name = str(_cls_cfg["name"])
        CLASSIFICATION_MODEL_CONFIGS[_cls_name] = {k: v for k, v in _cls_cfg.items() if k != "name"}

    # 算法代码 -> 分类器名称列表（在 core/classifier.py 中使用）
    _classifier_raw: List[dict] = _ALGO_CONFIG.get("algorithm_classifiers", [])
    ALGORITHM_CLASSIFIERS: Dict[str, List[str]] = {
        str(_item.get("code")): [str(cls) for cls in _item.get("classifiers", [])]
        for _item in _classifier_raw
        if isinstance(_item, dict) and "code" in _item
    }

    # 需要线程隔离的检测器类名列表（如 ByteTrack 等带跟踪状态的检测器）
    # 这些检测器在每个 StreamWorker 线程中拥有独立实例，避免多路视频状态互相干扰
    THREAD_LOCAL_DETECTOR_CLASSES: List[str] = [
        str(_name)
        for _name in _ALGO_CONFIG.get("thread_local_detectors", [])
        if isinstance(_name, str)
    ]

    # 算法代码 -> 报警去重配置
    _dedup_raw: List[dict] = _ALGO_CONFIG.get("alert_dedup", [])
    ALERT_DEDUP_CONFIG: Dict[str, Dict] = {}
    for _item in _dedup_raw:
        if not isinstance(_item, dict) or "code" not in _item:
            continue
        _code = str(_item["code"])
        ALERT_DEDUP_CONFIG[_code] = {
            "enabled": bool(_item.get("enabled", False)),
            "cooldown_seconds": float(_item.get("cooldown_seconds", 30.0)),
            "iou_thresh": float(_item.get("iou_thresh", 0.5)),
        }
    
        # 本地图片测试配置（tests/local/test_local_image.py 使用）。
    # 结构：{default_image_dir: str, enable_classifier: bool, enable_vl: bool,
    #        algorithms: [{code, image?, fence?, mode?}]}
    _test_local_raw: dict = _ALGO_CONFIG.get("test_local", {}) or {}
    TEST_LOCAL_DEFAULT_IMAGE_DIR: str = str(
        _test_local_raw.get("default_image_dir", "./asserts/images")
    )
    # 本地测试时是否启用分类器 / VL 大模型二次确认（默认为 True）
    TEST_LOCAL_ENABLE_CLASSIFIER: bool = bool(
        _test_local_raw.get("enable_classifier", True)
    )
    TEST_LOCAL_ENABLE_VL: bool = bool(
        _test_local_raw.get("enable_vl", True)
    )
    TEST_LOCAL_ALGORITHMS: Dict[str, dict] = {}
    for _item in _test_local_raw.get("algorithms", []):
        if not isinstance(_item, dict) or "code" not in _item:
            continue
        _t_code = str(_item["code"])
        TEST_LOCAL_ALGORITHMS[_t_code] = {
            "image": str(_item.get("image", "") or ""),
            "fence": str(_item.get("fence", "") or ""),
            "mode": str(_item.get("mode", "") or ""),
        }


def get_config(env: str | None = None) -> BaseConfig:
    """
    获取配置实例。

    历史遗留函数，原用于按 env 参数选择环境子类。现已简化为只返回 BaseConfig，
    所有差异化配置都通过同名环境变量覆盖，无需再维护环境子类。
    """
    return BaseConfig()


# 导出当前激活的配置实例，供项目全局使用
config = BaseConfig()
