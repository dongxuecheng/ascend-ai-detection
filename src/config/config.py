import os
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Type, List

import yaml
# 自动加载项目根目录或当前目录下的 .env 文件
load_dotenv()


def _load_algorithms_yaml() -> dict:
    """
    读取外置算法配置文件 config/algorithms.yaml。
    文件不存在、解析失败或格式错误时直接抛出异常，强制要求配置必须存在。
    """
    # src/config/config.py -> 项目根目录 -> config/algorithms.yaml
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yaml_path = os.path.join(project_root, "config", "algorithms.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"算法配置文件不存在: {yaml_path}\n"
            f"请在项目根目录 config/ 下创建 algorithms.yaml 并填写算法配置。"
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"algorithms.yaml 根节点必须是字典/对象，实际类型: {type(data).__name__}")
    return data


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


class BaseConfig(BaseModel):
    """公共配置基类，各环境共享的默认值在此定义"""
    SAM3_URL: str = "http://172.16.20.193:18001/predict"
    SAM3_URL_OBJ: str = "http://172.16.20.193:18001/predict"

    # 上传报警记录的接口地址
    UPLOAD_URL: str = "http://172.16.20.193:28080/open/api/operate/upload"
    # 获取电子围栏的接口地址
    GET_FENCE_URL: str = "http://172.16.20.193:28080/open/api/operate/fence"
    # 获取任务列表的接口地址
    GET_TASK_URL: str = "http://172.16.20.193:28080/open/api/operate/taskList"
    # 获取流地址的接口地址
    GET_RTSP_URL: str = "http://172.16.20.193:28080/open/api/operate/previewURLs"

    REQUEST_INTERVAL: int = 60

    # gRPC RTSP 流媒体后端地址
    STREAM_SERVER_ADDRESS: str = "192.168.100.74:50051"
    # 流恢复：两次重建之间的最小冷却（秒）
    STREAM_RECOVERY_COOLDOWN_SEC: float = 10.0

    # 违章原图本地保存根目录
    VIOLATION_IMAGE_SAVE_DIR: str = "/mnt/yolo/images"

    TRITON_YOLO_URL: str = "192.168.100.74:38000"

    # ---- VL 大模型配置 ----
    VL_ENABLED: bool = os.getenv("VL_ENABLED", "true").lower() in ("1", "true", "yes", "on")
    VL_API_URL: str = os.getenv("VL_API_URL", "https://api.siliconflow.cn/v1")
    VL_API_KEY: str = os.getenv("VL_API_KEY", "sk-koduolhsfnmpzojdeeehwvxglwbrxoybwthlclzirskjkszr")
    VL_MODEL: str = os.getenv("VL_MODEL", "Qwen/Qwen3.5-397B-A17B")
    # 是否开启大模型思考链（reasoning/thinking），默认关闭以加快响应
    VL_ENABLE_THINKING: bool = os.getenv("VL_ENABLE_THINKING", "false").lower() in ("1", "true", "yes", "on")

    # ---- 算法相关配置：必须从 algorithms.yaml 读取，无默认值 ----
    ALGORITHM_CODES: List[str] = _ALGO_CONFIG["supported_codes"]

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

    # 解析 sam3_urls（统一格式：列表，每项含 code/url；未配置则回退到 SAM3_URL_OBJ）
    _sam3_urls_raw: List[dict] = _ALGO_CONFIG.get("sam3_urls", [])
    ALGORITHM_SAM3_URL: Dict[str, str] = {
        str(_item.get("code")): str(_item.get("url"))
        for _item in _sam3_urls_raw
        if isinstance(_item, dict) and "code" in _item and "url" in _item
    }

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

class CompanyConfig(BaseConfig):
    """公司内部环境"""
    pass


class MHWJConfig(BaseConfig):
    """梅花味精环境"""
    # 上传报警记录的接口地址
    UPLOAD_URL: str = "http://172.16.20.193:28080/open/api/operate/upload"
    # 获取电子围栏的接口地址
    GET_FENCE_URL: str = "http://172.16.20.193:28080/open/api/operate/fence"
    # 获取任务列表的接口地址
    GET_TASK_URL: str = "http://172.16.20.193:28080/open/api/operate/taskList"
    # 获取流地址的接口地址
    GET_RTSP_URL: str = "http://172.16.20.193:28080/open/api/operate/previewURLs"


class HDZPConfig(BaseConfig):
    """华电漳平环境"""
    SAM3_URL: str = "http://192.168.100.75:18002/predict"
    SAM3_URL_OBJ: str = "http://192.168.100.75:18002/predict-person-about-small-object"
    # 上传报警记录的接口地址
    UPLOAD_URL: str = "http://192.168.100.73/open/api/operate/upload"
    # 获取电子围栏的接口地址
    GET_FENCE_URL: str = "http://192.168.100.73/open/api/operate/fence"
    # 获取任务列表的接口地址
    GET_TASK_URL: str = "http://192.168.100.73/open/api/operate/taskList"
    # 获取流地址的接口地址
    GET_RTSP_URL: str = "http://192.168.100.73/open/api/operate/previewURLs"

# 配置注册表：新增环境只需在这里注册，无需修改其他逻辑
CONFIG_MAP: Dict[str, Type[BaseConfig]] = {
    "company": CompanyConfig,
    "mhwj": MHWJConfig,
    "hdzp": HDZPConfig
}

# 默认环境（当环境变量和 .env 均未设置时使用）
_DEFAULT_ENV = "hdzp"


def get_config(env: str | None = None) -> BaseConfig:
    """
    获取指定环境的配置实例
    :param env: 环境名称，为 None 时依次从环境变量、.env 文件、默认值读取
    """
    name = (env or os.getenv("AIDETECTION_ENV", _DEFAULT_ENV)).lower()
    if name not in CONFIG_MAP:
        raise ValueError(
            f"Unknown config '{name}'. Available: {list(CONFIG_MAP.keys())}"
        )
    return CONFIG_MAP[name]()


# 导出当前激活的配置实例，供项目全局使用
config = get_config()
