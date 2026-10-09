from utils.filter import label_filter
from utils.obj import Box


class FireDetector:
    """火焰识别：返回置信度严格大于 0.6 的 fire 检测框。"""

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        return [box for box in label_filter(predictions, ["fire"]) if box.score > 0.6]
