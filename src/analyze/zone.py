# 人员闯入判断
from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("zone")


class ZoneDetector:
    """
    区域入侵检测器，按设备维度保存历史状态
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

    @staticmethod
    def _to_polygon_coords(fence):
        """
        将 fence 数据统一转换为 Polygon 可接受的坐标列表。
        支持 numpy 数组、列表等多种格式。
        """
        import numpy as np
        if hasattr(fence, 'tolist'):
            fence = fence.tolist()
        # 确保每个点都是 tuple (x, y)
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断人员是否闯入指定区域
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，支持 numpy 数组或列表格式
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :return: 闯入区域的人员 Box 列表
        """
        if fences is None or fences == []:
            return []

        result: list[Box] = []
        fence_polygons = []
        for i, fence in enumerate(fences):
            coords = self._to_polygon_coords(fence)
            if len(coords) < 3:
                logger.warning(f"[device={device_id}] 围栏-{i} 坐标点不足3个，无法构成多边形，跳过: {coords}")
                continue
            try:
                fence_polygons.append(Polygon(coords))
            except Exception as e:
                logger.error(f"[device={device_id}] 围栏-{i} 创建 Polygon 失败: {e}, coords={coords}")
                continue

        if not fence_polygons:
            logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过入侵检测")
            return []

        person_boxes = label_filter(predictions, ['person'])
        person_boxes = area_filter(person_boxes, 1000.0)
        person_boxes = score_filter(person_boxes, 0.8)

        for person in person_boxes:
            for fence in fence_polygons:
                if person.fence_iom(fence) > 0.5:
                    result.append(person)
                    break

        # 只有指定了设备ID才保存历史状态，便于后续做时序分析（如连续多帧确认、火焰跳动检测等）
        if device_id:
            self._history[device_id] = result
        return result
