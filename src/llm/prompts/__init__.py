"""
大模型（VL）视觉复核提示词集合。

按算法/检测器模块分文件组织 prompt，每个文件独立完整，不抽取公共部分，便于单独维护。

算法码与模块名的映射关系在 config/algorithms.yaml 的 `algorithm_vl_config` 中配置，
本包根据配置动态加载对应的 prompt 模块。
"""

import importlib
from typing import Optional

from config.config import config
from utils.logger import setup_logger

logger = setup_logger("vl_prompts")


VL_SYSTEM_PROMPT = (
    "你是工业安全视觉复核专家，专门从施工现场图像中识别个人防护装备（PPE）的缺失情况。"
    "你的回答必须严格遵循以下 JSON 格式，不要输出任何额外文字、解释或 markdown 代码块：\n"
    "{\n"
    '  "has_violation": boolean,  // true=存在违规（缺少指定装备），false=合规（装备已佩戴）\n'
    '  "reason": "string",        // 简述判定理由；若合规请固定输出 "无违规"\n'
    '  "details": ["string"]      // 违规类型标识数组，无违规则必须为空列表 []\n'
    "}\n"
    "注意：boolean 必须是 true 或 false 小写，不要使用字符串。"
)


def _load_vl_prompts() -> dict[str, str | dict]:
    """
    根据 config/algorithms.yaml 中的 `algorithm_vl_config` 配置，动态加载各算法码对应的 prompt 模块。
    :return: {算法码: prompt 字典或字符串}
    """
    result: dict[str, str | dict] = {}
    for code, cfg in config.ALGORITHM_VL_CONFIG.items():
        module_name = cfg.get("module", "")
        if not module_name:
            logger.warning(f"VL 配置缺少 module | code={code}")
            continue
        try:
            module = importlib.import_module(f"llm.prompts.{module_name}")
            prompts = getattr(module, "PROMPTS", None)
            if prompts is None:
                logger.error(f"VL prompt 模块缺少 PROMPTS 变量 | code={code}, module={module_name}")
                continue
            result[code] = prompts
            logger.info(f"VL prompt 加载成功 | code={code}, module={module_name}")
        except Exception as e:
            logger.error(f"VL prompt 模块加载失败 | code={code}, module={module_name}: {e}")
    return result


# 视觉大模型（VL）按算法代码组织的提示词
VL_PROMPTS: dict[str, str | dict] = _load_vl_prompts()


def select_vl_prompt(prompts: str | dict, label: str) -> Optional[str]:
    """
    根据违规标签名直接选取对应的 VL prompt。
    :param prompts: 字符串（单一 prompt）或字典（多场景 prompt）
    :param label: 违规标签名（如 no_safety_rope, alarm-no-safety-belt）
    :return: 选中的 prompt 字符串，无匹配返回 None
    """
    if isinstance(prompts, str):
        return prompts
    if not isinstance(prompts, dict):
        return None

    # 直接用 label 作为 key 查找
    if label in prompts:
        return prompts[label]

    # 兜底：default
    return prompts.get("default")
