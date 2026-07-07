from typing import Dict, List, Tuple

from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
from utils.filter import label_filter, score_filter
from analyze.helmet import HelmetDetector
from analyze.glove import GloveDetector
from analyze.zone import ZoneDetector
from analyze.skin import ExposedArmLegDetector
from analyze.shield import ShieldDetector
from analyze.safety import SafetyDetector
from analyze.single import SingleDetector
from analyze.smoke import SmokingDetector
from analyze.car import CarDetector
from analyze.extinguisher import ExtinguisherDetector
from analyze.departure import DepartureDetector
from analyze.play_phone import PlayPhoneDetector
from analyze.height_work import HeightWorkDetector
from analyze.coal import CoalDetector

logger = setup_logger("analyzer")

# 检测器类名 -> 类 的映射
_DETECTOR_CLASSES = {
    "HelmetDetector": HelmetDetector,
    "GloveDetector": GloveDetector,
    "ZoneDetector": ZoneDetector,
    "ExposedArmLegDetector": ExposedArmLegDetector,
    "ShieldDetector": ShieldDetector,
    "SafetyDetector": SafetyDetector,
    "SingleDetector": SingleDetector,
    "SmokingDetector": SmokingDetector,
    "CarDetector": CarDetector,
    "ExtinguisherDetector": ExtinguisherDetector,
    "DepartureDetector": DepartureDetector,
    "PlayPhoneDetector": PlayPhoneDetector,
    "HeightWorkDetector": HeightWorkDetector,
    "CoalDetector": CoalDetector,
}

# 检测器类名 -> 单例实例
_detector_instances: Dict[str, object] = {}


def _get_detector_instance(class_name: str):
    """获取或创建指定检测器的单例实例"""
    if class_name not in _detector_instances:
        cls = _DETECTOR_CLASSES.get(class_name)
        if cls is None:
            logger.error(f"未找到检测器类: {class_name}，请在 _DETECTOR_CLASSES 中注册")
            return None
        _detector_instances[class_name] = cls()
    return _detector_instances[class_name]


# 根据 config/algorithms.yaml 中的 algorithm_detectors 配置，构建 code -> 检测器实例列表
detectors: Dict[str, List] = {}
for _code, _class_names in config.ALGORITHM_DETECTORS.items():
    _instances = []
    for _name in _class_names:
        _inst = _get_detector_instance(_name)
        if _inst is not None:
            _instances.append(_inst)
    detectors[_code] = _instances

logger.info(
    f"检测器注册完成 | "
    f"{ {k: [c.__class__.__name__ for c in v] for k, v in detectors.items()} }"
)


def analyze_for_task(boxes: List[Box], task, fences=None, image_width: int = 0, image_height: int = 0) -> List[Box]:
    """
    根据任务的算法代码，对 Box 列表进行对应的违规分析

    :param boxes: SAM3 返回的全部检测框
    :param task: 任务对象
    :param fences: 电子围栏坐标列表（优先使用外部传入，为 None 时回退到 task.electricFence）
    :return: 违规目标 Box 列表
    """
    algo_code = str(task.algorithmCode)
    device_id = task.deviceId or ""
    task_name = getattr(task, 'alarmTaskName', '') or getattr(task, 'taskName', '') or '未知任务'

    # 预统计传入的类别，方便排查
    label_counts = {}
    for b in boxes:
        label_counts[b.label] = label_counts.get(b.label, 0) + 1
    if len(boxes) > 0:
        logger.info(
            f"[device={device_id}] analyze_for_task 开始 | "
            f"task_name={task_name}, algo={algo_code}, 输入框总数={len(boxes)}, 类别分布={label_counts}"
        )

    if algo_code not in detectors:
        logger.warning(f"[device={device_id}] analyze_for_task 未找到算法对应的检测器，algo_code={algo_code}")
        return []
    else:
        result = []
        for detector in detectors[algo_code]:
            try:
                detector_result = detector.detect(boxes, fences=fences, device_id=device_id, image_width=image_width, image_height=image_height)
                if len(boxes) > 0:
                    logger.info(f"[device={device_id}] analyze_for_task 检测器 {detector.__class__.__name__} 返回 {len(detector_result)} 个违规目标")
                result.extend(detector_result)
            except Exception as e:
                logger.error(f"[device={device_id}] analyze_for_task 检测器 {detector.__class__.__name__} 运行出错: {e}")

        for r in result:
            logger.info(f"[device={device_id}] analyze_for_task 违规目标: {r}")
        return result


def merge_prompts_for_tasks(tasks: List) -> Tuple[List[str], bool]:
    """
    合并多个任务的 SAM3 prompt，去重后返回。
    同时判断是否需要 return_mask：只要合并的任务中有一个要求 return_mask，
    整体就返回 True（SAM3 一次推理统一返回 mask）。
    """
    merged = set()
    return_mask = False
    for task in tasks:
        algo_code = str(task.algorithmCode)
        prompts = config.ALGORITHM_SAM3_PROMPT.get(algo_code, [])
        # 电子围栏也需要 person
        if algo_code not in config.ALGORITHM_SAM3_PROMPT and task.electricFence:
            prompts = ["person"]
        merged.update(prompts)
        if config.ALGORITHM_SAM3_RETURN_MASK.get(algo_code, False):
            return_mask = True
    return list(merged), return_mask
