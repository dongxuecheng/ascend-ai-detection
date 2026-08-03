"""
分类器二次确认模块。

根据 config/algorithms.yaml 中的 algorithm_classifiers 配置，
对规则引擎检出的违规目标裁剪后调用 Triton 分类模型进行过滤。

调用位置：core/stream_worker.py 中 analyze_for_task 之后、vl_analyze_for_task 之前。
"""

import threading
from typing import Dict, List, Optional

import numpy as np

from config.config import config
from detect.classification_client import TritonClassificationClient
from utils.logger import setup_logger
from utils.obj import Box

logger = setup_logger("classifier")

# 分类器名称 -> 客户端实例（线程安全单例）
_classifier_clients: Dict[str, TritonClassificationClient] = {}
_classifier_lock = threading.Lock()


def _get_classifier_client(name: str) -> Optional[TritonClassificationClient]:
    """获取或创建指定名称的分类器客户端"""
    client = _classifier_clients.get(name)
    if client is not None:
        return client

    with _classifier_lock:
        client = _classifier_clients.get(name)
        if client is not None:
            return client

        cfg = config.CLASSIFICATION_MODEL_CONFIGS.get(name)
        if not cfg:
            logger.error(f"未找到分类器配置: {name}")
            return None

        try:
            client = TritonClassificationClient(
                url=config.TRITON_YOLO_URL,
                model_name=cfg.get("model_name", name),
                input_name=cfg.get("input_name", "raw_image"),
                output_name=cfg.get("output_name", "output"),
                input_size=cfg.get("input_size", 224),
                labels=cfg.get("labels"),
                mean=cfg.get("mean"),
                std=cfg.get("std"),
                protocol="shm",
            )
            _classifier_clients[name] = client
            logger.info(f"分类器客户端创建成功: {name}")
        except Exception as e:
            logger.error(f"分类器客户端创建失败 {name}: {e}")
            return None

    return client


def _crop_around_box(
    frame: np.ndarray,
    box: List[float],
    image_width: int,
    image_height: int,
    expand_ratio: float = 2.0,
) -> np.ndarray:
    """
    以 box 为中心按 expand_ratio 扩展后裁剪，返回 BGR 图像。
    expand_ratio=2.0 表示最终宽高为原框的 2 倍。
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)
    cx1 = max(0, cx - new_w // 2)
    cy1 = max(0, cy - new_h // 2)
    cx2 = min(image_width, cx1 + new_w)
    cy2 = min(image_height, cy1 + new_h)

    # 如果扩展后贴边，尽量保持裁剪框大小
    if cx2 - cx1 < new_w and cx1 > 0:
        cx1 = max(0, cx2 - new_w)
    if cy2 - cy1 < new_h and cy1 > 0:
        cy1 = max(0, cy2 - new_h)

    return frame[cy1:cy2, cx1:cx2]


def classify_for_task(
    frame: np.ndarray,
    task,
    rule_violations: List[Box],
    image_width: int = 0,
    image_height: int = 0,
) -> List[Box]:
    """
    对规则引擎检出的违规目标进行分类二次确认。

    :param frame: 当前视频帧
    :param task: 任务对象
    :param rule_violations: 规则引擎检出的违规目标列表
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :return: 分类器确认后的违规目标列表
    """
    algo_code = str(task.algorithmCode)
    classifier_names = config.ALGORITHM_CLASSIFIERS.get(algo_code, [])
    if not classifier_names:
        return rule_violations

    if image_width == 0 or image_height == 0:
        image_height, image_width = frame.shape[:2]

    confirmed: List[Box] = []
    for idx, v in enumerate(rule_violations):
        keep = True
        for name in classifier_names:
            client = _get_classifier_client(name)
            if client is None:
                continue

            cfg = config.CLASSIFICATION_MODEL_CONFIGS.get(name, {})
            expand_ratio = cfg.get("expand_ratio", 2.0)
            target_class = cfg.get("target_class")
            conf_thresh = cfg.get("conf_thresh", 0.5)

            crop = _crop_around_box(
                frame, v.box, image_width, image_height, expand_ratio
            )
            if crop.size == 0:
                logger.warning(f"分类裁剪区域为空 | algo={algo_code} | idx={idx}")
                continue

            try:
                class_id, confidence, probs = client.predict(crop)
                predicted_label = client.labels[class_id] if class_id < len(client.labels) else str(class_id)
                logger.info(
                    f"分类结果 | classifier={name} | algo={algo_code} | idx={idx} | "
                    f"label={v.label} | predict={predicted_label} | conf={confidence:.3f} | probs={probs}"
                )

                # 如果配置了目标类别，且预测不是目标类别，则过滤掉该违规
                if target_class is not None:
                    target_id = client.labels.index(target_class) if target_class in client.labels else -1
                    if target_id != -1 and class_id != target_id:
                        logger.info(
                            f"分类器过滤违规 | classifier={name} | algo={algo_code} | "
                            f"idx={idx} | 目标={target_class} | 预测={predicted_label}"
                        )
                        keep = False
                        break

                # 如果置信度低于阈值，也过滤（可选）
                if confidence < conf_thresh:
                    logger.info(
                        f"分类器置信度不足过滤 | classifier={name} | algo={algo_code} | "
                        f"idx={idx} | conf={confidence:.3f} < threshold={conf_thresh}"
                    )
                    keep = False
                    break
            except Exception as e:
                logger.error(f"分类推理异常 | classifier={name} | algo={algo_code} | idx={idx}: {e}")
                continue

        if keep:
            confirmed.append(v)

    logger.info(
        f"分类过滤完成 | algo={algo_code} | 原始={len(rule_violations)} | 保留={len(confirmed)}"
    )
    return confirmed
