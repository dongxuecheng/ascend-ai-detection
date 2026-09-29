"""
VL（视觉大模型）二次确认模块。

负责：
  - VL 客户端的延迟初始化与单例管理
  - 根据任务算法码和违规标签选取 prompt
  - 对规则引擎检出的违规目标裁剪后调用 VL 复核

core/analyzer.py 只负责传统规则分析，VL 相关逻辑统一迁移至此。
"""

import json
import time
import threading
import time
from typing import List

import cv2
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


def _enhance_contrast(crop: np.ndarray) -> np.ndarray:
    """
    对 ROI 裁剪图做局部对比度增强（CLAHE）。

    在 LAB 色彩空间的 L（亮度）通道上做 CLAHE，保持颜色不失真，
    使银灰色反光条在深色衣物背景中更明显（“亮”出来），提升小模型识别率。
    """
    if crop is None or crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        return crop
    try:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.warning(f"对比度增强失败，使用原始裁剪图: {e}")
        return crop


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
            # 裁剪违规目标区域（上下左右各扩，让大模型看到周围上下文）
            x1, y1, x2, y2 = map(int, v.box)
            w, h = x2 - x1, y2 - y1
            if algo_code == "14":
                # 香烟目标较小，左右/上下拓展更多，让大模型看到面部上下文
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.2)
            elif algo_code == "10":
                pad_x = 0
                pad_y = 0
            elif algo_code == "32":
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
            else:
                pad_x = int(w * 0.5)
                pad_y = int(h * 0.3)

            # 先按 pad 扩展，再以扩展框中心做 1:1 正方形裁剪
            ex1 = x1 - pad_x
            ey1 = y1 - pad_y
            ex2 = x2 + pad_x
            ey2 = y2 + pad_y
            ecx = (ex1 + ex2) / 2.0
            ecy = (ey1 + ey2) / 2.0
            side = max(ex2 - ex1, ey2 - ey1)
            half = side / 2.0
            cx1 = int(round(ecx - half))
            cy1 = int(round(ecy - half))
            cx2 = cx1 + int(round(side))
            cy2 = cy1 + int(round(side))

            # 越界处理（保持 1:1，贴边时整体平移）
            if cx1 < 0:
                cx2 -= cx1
                cx1 = 0
            if cy1 < 0:
                cy2 -= cy1
                cy1 = 0
            if cx2 > image_width:
                cx1 -= cx2 - image_width
                cx2 = image_width
            if cy2 > image_height:
                cy1 -= cy2 - image_height
                cy2 = image_height
            cx1 = max(0, min(cx1, image_width))
            cy1 = max(0, min(cy1, image_height))
            cx2 = max(0, min(cx2, image_width))
            cy2 = max(0, min(cy2, image_height))

            v.box = [cx1, cy1, cx2, cy2]
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                confirmed.append(v)
                continue
            if algo_code == "32":
                name = "sleep/" + str(time.time()) + ".jpg"
                cv2.imwrite(name, crop)

            # 反光衣：送入 VL 前先做对比度增强，凸显银灰色反光条
            # if algo_code == "10":
            #     crop = _enhance_contrast(crop)

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
                f"crop=({cx1},{cy1},{cx2},{cy2}) | 耗时={(t_end - t_start) * 1000:.1f}ms, result={result_str}")
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
