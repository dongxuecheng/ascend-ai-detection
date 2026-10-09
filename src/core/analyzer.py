import inspect
import threading
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np

from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
from utils.filter import label_filter, score_filter
from analyze.helmet import HelmetDetector
from analyze.glove import GloveDetector
from analyze.zone import ZoneDetector
from analyze.car_zone import CarZoneDetector
from analyze.skin import ExposedArmLegDetector
from analyze.shield import ShieldDetector
from analyze.vest import VestDetector
from analyze.safety import SafetyDetector
from analyze.single import SingleDetector
from analyze.smoke import SmokingDetector
from analyze.sleep_duty import SleepDutyDetector
from analyze.belt_deviation import BeltDeviationDetector
from analyze.car import CarDetector
from analyze.extinguisher import ExtinguisherDetector
from analyze.fire import FireDetector
from analyze.departure import DepartureDetector
from analyze.play_phone import PlayPhoneDetector
from analyze.move_phone import MoveUsePhoneDetector
from analyze.height_work import HeightWorkDetector
from analyze.coal import CoalDetector
from analyze.coal_foreign_object import CoalForeignObjectDetector
from analyze.coal_foreign_object_5 import CoalForeignObjectDetector5
from analyze.empty_truck import EmptyTruckDetector
from analyze.person_approaching_moving_vehicle import PersonApproachingMovingVehicleDetector
from analyze.unsupervised_person import UnsupervisedDetector


logger = setup_logger("analyzer")

# 检测器类名 -> 类 的映射
_DETECTOR_CLASSES = {
    "HelmetDetector": HelmetDetector,
    "GloveDetector": GloveDetector,
    "ZoneDetector": ZoneDetector,
    "CarZoneDetector": CarZoneDetector,
    "ExposedArmLegDetector": ExposedArmLegDetector,
    "ShieldDetector": ShieldDetector,
    "VestDetector": VestDetector,
    "SafetyDetector": SafetyDetector,
    "SingleDetector": SingleDetector,
    "SmokingDetector": SmokingDetector,
    "SleepDutyDetector": SleepDutyDetector,
    "CarDetector": CarDetector,
    "BeltDeviationDetector": BeltDeviationDetector,
    "ExtinguisherDetector": ExtinguisherDetector,
    "FireDetector": FireDetector,
    "DepartureDetector": DepartureDetector,
    "PlayPhoneDetector": PlayPhoneDetector,
    "MoveUsePhoneDetector": MoveUsePhoneDetector,
    "HeightWorkDetector": HeightWorkDetector,
    "CoalDetector": CoalDetector,
    "CoalForeignObjectDetector": CoalForeignObjectDetector,
    "CoalForeignObjectDetector5": CoalForeignObjectDetector5,
    "EmptyTruckDetector": EmptyTruckDetector,
    "PersonApproachingMovingVehicleDetector": PersonApproachingMovingVehicleDetector,
    "UnsupervisedDetector": UnsupervisedDetector,
}

# 全局单例实例（非线程隔离的检测器共享使用）
_detector_instances: Dict[str, object] = {}

# 线程本地实例（线程隔离的检测器，每个 StreamWorker 线程拥有独立实例）
_thread_local_detectors = threading.local()

# 需要线程隔离的检测器类名集合（如 ByteTrack 等带跟踪状态的检测器）
_THREAD_LOCAL_DETECTOR_CLASSES: set = set(config.THREAD_LOCAL_DETECTOR_CLASSES)


def _get_detector_instance(class_name: str):
    """
    获取或创建指定检测器的实例。

    支持两种模式：
      - 全局单例：所有线程共享同一个实例，适用于无状态或状态已按 device_id 隔离的检测器。
      - 线程隔离：每个 StreamWorker 线程拥有独立实例，适用于 ByteTrack 等带帧间跟踪状态的检测器。
    """
    cls = _DETECTOR_CLASSES.get(class_name)
    if cls is None:
        logger.error(f"未找到检测器类: {class_name}，请在 _DETECTOR_CLASSES 中注册")
        return None

    if class_name in _THREAD_LOCAL_DETECTOR_CLASSES:
        # 线程隔离：每个 StreamWorker 线程独立实例
        if not hasattr(_thread_local_detectors, "instances"):
            _thread_local_detectors.instances = {}
        if class_name not in _thread_local_detectors.instances:
            _thread_local_detectors.instances[class_name] = cls()
            logger.info(
                f"线程隔离检测器创建: {class_name} @ "
                f"{threading.current_thread().name}"
            )
        return _thread_local_detectors.instances[class_name]
    else:
        # 全局单例：所有线程共享
        if class_name not in _detector_instances:
            _detector_instances[class_name] = cls()
        return _detector_instances[class_name]


# 根据 config/algorithms.yaml 中的 algorithm_detectors 配置，构建 code -> 检测器类名列表
# 实例在 analyze_for_task 调用时按需创建，确保线程隔离的检测器在当前 StreamWorker 线程中实例化
_detector_class_names: Dict[str, List[str]] = dict(config.ALGORITHM_DETECTORS)

logger.info(
    f"检测器注册完成 | "
    f"{ {k: v for k, v in _detector_class_names.items()} }"
)


def analyze_for_task(
    boxes: List[Box],
    task,
    fences=None,
    image_width: int = 0,
    image_height: int = 0,
    frame: Optional[np.ndarray] = None,
    frame_provider: Optional[Callable[[], np.ndarray]] = None,
) -> List[Box]:
    """
    根据任务的算法代码，对 Box 列表进行对应的违规分析

    :param boxes: SAM3 / YOLO 返回的全部检测框
    :param task: 任务对象
    :param fences: 电子围栏坐标列表（优先使用外部传入，为 None 时回退到 task.electricFence）
    :param image_width: 原图宽度
    :param image_height: 原图高度
    :param frame: 当前帧图像（BGR numpy 数组），部分检测器（如 MoveUsePhoneDetector）需要缓存历史帧
    :param frame_provider: 按需获取 BGR 图像；仅当检测器接收 frame 参数时调用
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

    if algo_code not in _detector_class_names:
        logger.warning(f"[device={device_id}] analyze_for_task 未找到算法对应的检测器，algo_code={algo_code}")
        return []
    else:
        result = []
        for class_name in _detector_class_names[algo_code]:
            detector = _get_detector_instance(class_name)
            if detector is None:
                continue
            try:
                # 仅当检测器 detect 方法支持 frame 参数时才传入，保持向后兼容
                detect_kwargs = {
                    "fences": fences,
                    "device_id": device_id,
                    "image_width": image_width,
                    "image_height": image_height,
                }
                if "frame" in inspect.signature(detector.detect).parameters:
                    if frame is None and frame_provider is not None:
                        frame = frame_provider()
                    if frame is not None:
                        detect_kwargs["frame"] = frame
                detector_result = detector.detect(boxes, **detect_kwargs)
                if len(boxes) > 0 or frame is not None:
                    logger.info(f"[device={device_id}] task_name={task_name} analyze_for_task 检测器 {detector.__class__.__name__} 返回 {len(detector_result)} 个违规目标")
                result.extend(detector_result)
            except Exception as e:
                logger.error(f"[device={device_id}] task_name={task_name} analyze_for_task 检测器 {detector.__class__.__name__} 运行出错: {e}")

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
