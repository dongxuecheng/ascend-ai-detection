import os
import sys
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from llm.vl_analyzer import _get_vl_client

VL_SYSTEM_PROMPT = (
    "你是工业安全视觉复核专家"
    "你的回答必须严格遵循以下 JSON 格式，不要输出任何额外文字、解释或 markdown 代码块：\n"
    "{\n"
    '  "has_violation": boolean,  // true=存在违规，false=合规\n'
    '  "reason": "string",        // 简述判定理由；若合规请固定输出 "无违规"\n'
    '  "details": ["string"]      // 违规类型标识数组，无违规则必须为空列表 []\n'
    "}\n"
    "注意：boolean 必须是 true 或 false 小写，不要使用字符串。"
)

prompt = (
    "检查画面中是否存在皮带，有皮带就违规"
)

if __name__ == "__main__":
    client = _get_vl_client()
    image = cv2.imread("asserts/images/test.jpg")
    result_str = client.analyze_image(
                image_source=image,
                system_prompt=VL_SYSTEM_PROMPT,
                user_prompt=prompt,
                require_json=True,
            )
    print(result_str)