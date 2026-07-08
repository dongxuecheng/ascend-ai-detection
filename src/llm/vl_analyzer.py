"""
VL（视觉大模型）二次确认模块。

负责：
  - VL 客户端的延迟初始化与单例管理
  - 根据任务算法码和违规标签选取 prompt
  - 对规则引擎检出的违规目标裁剪后调用 VL 复核

core/analyzer.py 只负责传统规则分析，VL 相关逻辑统一迁移至此。
"""

import json
import threading
import time
from typing import List

import numpy as np

from config.config import config
from llm.prompts import VL_PROMPTS, VL_SYSTEM_PROMPT, select_vl_prompt
from utils.logger import setup_logger
from utils.obj import Box

logger = setup_logger("vl_analyzer")

# VL 大模型客户端（延迟初始化，第一次调用时才创建）
_vl_client = None
_vl_client_lock = threading.Lock()


def _get_vl_client():
    """延迟初始化并返回 VL 大模型客户端（线程安全）"""
    global _vl_client
    if _vl_client is not None:
        return _vl_client
    with _vl_client_lock:
        if _vl_client is not None:
            return _vl_client
        if not config.VL_ENABLED:
            logger.info("VL 大模型已关闭 (VL_ENABLED=false)，跳过 VL 分析")
            return None
        if not config.VL_API_URL:
            logger.warning("VL 大模型未配置调用地址 (VL_API_URL)，跳过 VL 分析")
            return None
        try:
            from llm.llm import VisionAPIClient
            _vl_client = VisionAPIClient(
                api_key=config.VL_API_KEY or None,
                base_url=config.VL_API_URL or None,
                model=config.VL_MODEL,
                enable_thinking=config.VL_ENABLE_THINKING,
            )
            logger.info(f"VL 大模型客户端初始化成功 | url={config.VL_API_URL} | model={config.VL_MODEL}")
        except Exception as e:
            logger.warning(f"VL 大模型客户端初始化失败: {e}")
    return _vl_client


def vl_analyze_for_task(
    frame: np.ndarray,
    task,
    rule_violations: List[Box],
    image_width: int = 0,
    image_height: int = 0
) -> List[Box]:
    """
    VL 二次确认：直接遍历规则引擎的 violations，对每个违规目标裁剪后交给 VL 判断。
    VL 确认违规则保留，否认则过滤。

    :param frame: 当前视频帧 (numpy array)
    :param task: 任务对象
    :param rule_violations: 规则引擎检出的违规目标列表
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :return: VL 确认后的违规目标 Box 列表
    """
    algo_code = str(task.algorithmCode)

    # 只有 algorithms.yaml 中显式启用 VL 的算法码才进行大模型二次复核
    vl_cfg = config.ALGORITHM_VL_CONFIG.get(algo_code, {})
    if not vl_cfg.get("enabled", False):
        logger.info(f"VL 复核已关闭 | algo={algo_code}")
        return rule_violations

    prompts = VL_PROMPTS.get(algo_code)
    if not prompts:
        logger.warning(f"VL 未配置 prompt 模块 | algo={algo_code}，跳过 VL 复核")
        return rule_violations

    client = _get_vl_client()
    if client is None:
        return rule_violations

    confirmed: List[Box] = []
    for idx, v in enumerate(rule_violations):
        # 根据违规标签名直接选取 prompt
        user_prompt = select_vl_prompt(prompts, v.label)
        if not user_prompt:
            logger.warning(f"VL 未找到匹配 prompt | algo={algo_code} | label={v.label}，默认保留")
            confirmed.append(v)
            continue

        try:
            # 裁剪违规目标区域（上下左右各扩 50%，让大模型看到周围上下文）
            x1, y1, x2, y2 = map(int, v.box)
            w, h = x2 - x1, y2 - y1
            pad_x = int(w * 0.5)
            pad_y = int(h * 0.3)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(image_width, x2 + pad_x)
            cy2 = min(image_height, y2 + pad_y)

            cy1 -= pad_y
            cy2 += pad_y
            cx1 -= pad_x
            cx2 += pad_x
            # 越界处理
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(image_width, cx2)
            cy2 = min(image_height, cy2)
            v.box = [cx1, cy1, cx2, cy2]
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                confirmed.append(v)
                continue

            t_start = time.time()
            result_str = client.analyze_image(
                image_source=crop,
                system_prompt=VL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                require_json=True,
            )
            t_end = time.time()
            logger.info(
                f"VL 分析完成 | algo={algo_code} | idx={idx} | label={v.label} | "
                f"crop=({cx1},{cy1},{cx2},{cy2}) | 耗时={(t_end - t_start) * 1000:.1f}ms"
            )

            result_json = json.loads(result_str)
            if result_json.get("has_violation", False):
                logger.warning(f"VL 确认违规 | algo={algo_code} | idx={idx} | label={v.label}")
                confirmed.append(v)
            else:
                logger.info(f"VL 否认违规 | algo={algo_code} | idx={idx} | label={v.label}，过滤")
        except json.JSONDecodeError as e:
            logger.error(f"VL JSON 解析失败 | idx={idx}: {e} | raw={result_str[:200]}")
            confirmed.append(v)
        except Exception as e:
            logger.error(f"VL 分析异常 | idx={idx}: {e}")
            confirmed.append(v)

    return confirmed
