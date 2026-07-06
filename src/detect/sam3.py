import base64
from typing import List, Optional

import cv2
import numpy as np
import requests

from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
import time

logger = setup_logger("sam3")

# SAM3 默认 crop_config（与接口文档一致）
DEFAULT_CROP_CONFIG = {
    "max_size": 640,
    "padding": 20,
    "w_diou": 30,
    "w_expansion": 5,
    "count_penalty": 120,
    "nms_threshold": 0.2,
    "enable_ar_fix": True,
    "target_ar": 1,
}


def call_sam3(
    frame: np.ndarray,
    prompts: List[str],
    confidence_threshold: float = 0.5,
    return_mask: bool = False,
    pre_detect_labels: Optional[List[str]] = None,
    merge_results: bool = True,
    crop_config: Optional[dict] = None,
    url: Optional[str] = None,
) -> List[Box]:
    """
    调用 SAM3 推理服务，返回检测框列表

    :param frame: BGR 格式的 numpy 数组
    :param prompts: 检测目标文本提示词列表，如 ['person', 'head', 'helmet']
    :param confidence_threshold: 置信度阈值，默认 0.5
    :param return_mask: 是否返回掩码
    :param pre_detect_labels: 预检测标签列表，默认从 prompts 推断（取第一个）
    :param merge_results: 是否合并结果，默认 True
    :param crop_config: 裁剪配置字典，默认使用 DEFAULT_CROP_CONFIG
    :param url: 自定义 SAM3 接口地址；为空时使用 config.SAM3_URL_OBJ
    :return: Box 对象列表
    """
    if not prompts:
        return []

    try:
        # 1. 图片编码为 base64 JPEG
        success, encoded = cv2.imencode(".jpg", frame)
        # cv2.imwrite("debug_sam3_input.jpg", frame)  # 调试：保存输入图片
        if not success:
            logger.error("JPEG 编码失败")
            return []
        image_b64 = base64.b64encode(encoded).decode("utf-8")

        # 2. 构造 prompts 列表（忽略 boxes 字段）
        prompt_list = [{"text": p, "boxes": []} for p in prompts]

        # 3. 确定 pre_detect_labels（新接口字段，替代旧的 text 字段）
        if pre_detect_labels is None:
            # 默认取 prompts 中的第一个作为预检测标签，若不存在则回退到 person
            pre_detect_labels = [prompts[0]] if prompts else ["person"]

        # 4. 确定 crop_config
        effective_crop_config = crop_config if crop_config is not None else DEFAULT_CROP_CONFIG

        # 5. 组装请求体（适配新接口格式）
        payload = {
            "image_base64": image_b64,
            "confidence_threshold": confidence_threshold,
            "pre_detect_labels": pre_detect_labels,
            "prompts": prompt_list,
            "return_mask": return_mask,
            "merge_results": merge_results,
            "crop_config": effective_crop_config,
        }

        # 6. 发送请求
        effective_url = url if url else config.SAM3_URL_OBJ
        t_http_start = time.time()
        resp = requests.post(effective_url, json=payload, timeout=30)
        t_http_end = time.time()
        logger.info(f"SAM3 HTTP 请求耗时: {(t_http_end-t_http_start)*1000:.1f}ms | URL={effective_url}")
        resp.raise_for_status()
        data = resp.json()
        return parse_sam3_response(data)

    except requests.exceptions.RequestException as e:
        logger.error(f"SAM3 请求异常: {e}")
        return []
    except Exception as e:
        logger.error(f"SAM3 调用未知异常: {e}", exc_info=True)
        return []


def parse_sam3_response(data) -> List[Box]:
    """
    解析 SAM3 返回的 JSON 为 Box 对象列表

    支持多种返回格式：
      - {"results": [...]}         ← 你的 SAM3 实际格式
      - {"code": 0, "data": [...]}  ← 常见封装格式
      - [...]                       ← 直接列表格式
    """
    boxes = []

    if isinstance(data, dict):
        # 检查错误码
        code = data.get("code")
        if code is not None and code != 0:
            logger.warning(f"SAM3 返回错误码: {code}, msg={data.get('msg')}")
            return []

        # 依次尝试 "results"、"data" 等常见 key
        items = None
        for key in ("results", "data", "predictions", "output"):
            if key in data:
                items = data[key]
                break
        if items is None:
            logger.warning(f"SAM3 返回字典中未找到结果列表，keys={list(data.keys())}")
            return []
    elif isinstance(data, list):
        items = data
    else:
        logger.warning(f"SAM3 返回格式异常，类型={type(data)}")
        return []

    if not isinstance(items, list):
        logger.warning(f"SAM3 结果字段类型异常，期望 list，实际 {type(items)}")
        return []

    for item in items:
        try:
            if not isinstance(item, dict):
                continue
            label = item.get("label", "")
            score = float(item.get("score", 0))
            box_coords = item.get("box", [0, 0, 0, 0])
            mask = item.get("mask")
            boxes.append(Box(label=label, score=score, box=box_coords, mask=mask))
        except Exception as e:
            logger.error(f"解析 SAM3 结果项失败: {item}, 错误: {e}")
            continue

    logger.info(f"SAM3 解析完成: 原始结果数={len(items)}, 有效框数={len(boxes)}")
    return boxes
