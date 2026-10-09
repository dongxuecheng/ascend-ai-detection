import base64
import time
from typing import List, Optional
from urllib.parse import urlsplit

import cv2
import numpy as np
import requests

from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
from utils.rle import binary_mask_to_rle

logger = setup_logger("sam3")

# 恢复原有裁剪参数；max_crops 等预算继续由服务端默认值约束。
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
    confidence_threshold: float = 0.3,
    return_mask: bool = False,
    url: Optional[str] = None,
    *,
    pre_detect_labels: Optional[List[str]] = None,
    merge_results: bool = True,
    crop_config: Optional[dict] = None,
) -> List[Box]:
    """
    调用 ascend-sam3，按 URL 选择普通预测或目标精细检测协议。

    :param frame: BGR 格式的 numpy 数组
    :param prompts: 检测目标文本提示词列表，如 ['person', 'head', 'helmet']
    :param confidence_threshold: 置信度阈值，默认 0.3
    :param return_mask: 是否返回掩码
    :param url: 自定义 SAM3 接口地址；为空时使用 config.SAM3_URL_OBJ
    :param pre_detect_labels: 精细检测的主体预检测标签，默认 ['person']
    :param merge_results: 精细检测是否合并原图结果，默认 True
    :param crop_config: 精细检测裁剪参数；未传入时使用 DEFAULT_CROP_CONFIG
    :return: Box 对象列表
    """
    if not prompts:
        return []
    try:
        # 1. 图片编码为 base64 JPEG
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("JPEG 编码失败")
            return []
        image_b64 = base64.b64encode(encoded).decode("utf-8")

        effective_url = url if url else config.SAM3_URL_OBJ
        if urlsplit(effective_url).path.rstrip("/").endswith("/predict-obj-refine"):
            # 合并后的 prompts 来自 set，不能用第一个标签决定裁剪主体。
            payload = {
                "image_base64": image_b64,
                "confidence_threshold": confidence_threshold,
                "pre_detect_labels": pre_detect_labels if pre_detect_labels is not None else ["person"],
                "prompts": [{"text": p, "boxes": []} for p in prompts],
                "return_mask": return_mask,
                "merge_results": merge_results,
                "crop_config": dict(crop_config if crop_config is not None else DEFAULT_CROP_CONFIG),
            }
        else:
            payload = {
                "image": image_b64,
                "class_names": prompts,
                "confidence": confidence_threshold,
                "return_mask": return_mask,
            }

        t_http_start = time.time()
        resp = requests.post(effective_url, json=payload, timeout=config.SAM3_TIMEOUT_SECONDS)
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


def _parse_mask(item: dict) -> Optional[dict]:
    """将 Ascend 的行优先、1-based 起点/长度 RLE 转为 Box 的 COCO RLE。"""
    mask = item.get("mask")
    if mask is None or isinstance(mask, dict):
        return mask
    try:
        width, height = item["mask_width"], item["mask_height"]
        if (type(width) is not int or type(height) is not int
                or width <= 0 or height <= 0):
            raise ValueError("mask_width/mask_height 必须为正整数")
        x1, y1, x2, y2 = map(int, item["box"])
        if (height, width) != (y2 - y1, x2 - x1):
            raise ValueError("局部 mask 尺寸与检测框的整数 ROI 不一致")
        if not isinstance(mask, list) or len(mask) % 2:
            raise ValueError("mask 必须为起点/长度成对的列表")
        if any(type(value) is not int for value in mask):
            raise ValueError("mask 行程必须为整数")

        total = width * height
        previous_end = 0
        for start, length in zip(mask[0::2], mask[1::2]):
            offset = start - 1
            if offset < previous_end or length <= 0 or offset + length > total:
                raise ValueError("mask 行程越界、重叠或长度无效")
            previous_end = offset + length

        flat = np.zeros(total, dtype=np.uint8)
        for start, length in zip(mask[0::2], mask[1::2]):
            flat[start - 1:start - 1 + length] = 1
        return binary_mask_to_rle(flat.reshape((height, width)))
    except (KeyError, TypeError, ValueError, OverflowError) as e:
        # 损坏的 mask 不应导致有效检测框一并丢失。
        logger.warning("SAM3 mask 解析失败，保留检测框但忽略掩码: %s", e)
        return None


def parse_sam3_response(data) -> List[Box]:
    """
    解析 SAM3 返回的 JSON 为 Box 对象列表

    支持多种返回格式：
      - {"results": [...]}，ascend-sam3 实际格式
      - {"code": 0, "data": [...]}，兼容旧封装
      - [...]，兼容直接列表
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
            box_coords = [float(value) for value in item["box"]]
            if (len(box_coords) != 4 or not np.isfinite(box_coords).all()
                    or not np.isfinite(score) or not label
                    or box_coords[2] <= box_coords[0] or box_coords[3] <= box_coords[1]):
                raise ValueError("检测框标签、分数或坐标无效")
            mask = _parse_mask(item)
            boxes.append(Box(label=label, score=score, box=box_coords, mask=mask, source="SAM3"))
        except Exception as e:
            logger.error("解析 SAM3 结果项失败: %s", e)
            continue

    logger.info(f"SAM3 解析完成: 原始结果数={len(items)}, 有效框数={len(boxes)}")
    return boxes
