import base64
import os
import cv2
import numpy as np
from openai import OpenAI

class VisionAPIClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o", enable_thinking: bool = False):
        """
        初始化视觉大模型客户端
        :param api_key: API 密钥，默认从环境变量 OPENAI_API_KEY 读取
        :param base_url: 自定义 API 基础地址（如 vLLM/Qwen 自部署服务）
        :param model: 模型名称
        :param enable_thinking: 是否开启模型思考链（reasoning/thinking），默认关闭
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.enable_thinking = enable_thinking

    def _encode_local_image(self, image_path: str) -> str:
        """将本地图片文件编码为 Base64 格式"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"本地图片未找到: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _encode_opencv_image(self, cv2_img: np.ndarray) -> str:
        """将 OpenCV 图像对象 (numpy array) 直接在内存中编码为 Base64 格式"""
        # 将 OpenCV 图像编码为 JPEG 格式的内存字节流
        success, buffer = cv2.imencode('.jpg', cv2_img)
        if not success:
            raise ValueError("OpenCV 图像编码为 JPEG 失败")
        # 将字节流转换为 Base64 字符串
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_image(
        self, 
        image_source, 
        system_prompt: str, 
        user_prompt: str, 
        require_json: bool = False,
        detail: str = "high"
    ) -> str:
        """
        发送图片及双重提示词进行分析
        :param image_source: 可以是 OpenCV 图像 (np.ndarray)、网页URL 或 本地图片路径
        :param system_prompt: 系统提示词（定义角色、限制返回格式等）
        :param user_prompt: 用户提示词（具体问题）
        :param require_json: 是否强制模型返回严格的 JSON 格式
        :param detail: 图片分析精度，"high" 或 "low"
        """
        
        # 1. 智能判断输入类型并获取 Base64 或 URL
        if isinstance(image_source, np.ndarray):
            # 处理 OpenCV 传入的图片 (内存直传)
            base64_image = self._encode_opencv_image(image_source)
            image_url_dict = {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}
            
        elif isinstance(image_source, str):
            if image_source.startswith("http://") or image_source.startswith("https://"):
                # 处理网络 URL
                image_url_dict = {"url": image_source, "detail": detail}
            else:
                # 处理本地文件路径
                base64_image = self._encode_local_image(image_source)
                image_url_dict = {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}
        else:
            raise TypeError("不支持的 image_source 类型。请传入 np.ndarray(OpenCV)、URL 或 本地路径")

        # 2. 构建消息体 (分离 System 和 User)
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": image_url_dict}
                ]
            }
        ]

        # 3. 配置 API 请求参数
        api_params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.1 # 低温度保证稳定性
        }

        if require_json:
            api_params["response_format"] = { "type": "json_object" }

        # Qwen3 等推理模型通过 extra_body 控制是否输出思考链
        if not self.enable_thinking:
            api_params["extra_body"] = {"enable_thinking": False}

        # 4. 发送请求并返回
        try:
            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        except Exception as e:
            return f"{{\"error\": \"API 调用失败: {str(e)}\"}}"