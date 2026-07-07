# 堆煤检测
import numpy as np
import cv2
from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("coal")


class CoalDetector:
    """
    堆煤检测器：检测煤堆掩膜是否与电子围栏相交，若相交则判定为违规。
    """

    def __init__(self, min_score: float = 0.5):
        """
        :param min_score: coal / coal pile 框最低置信度
        """
        self.min_score = min_score

    @staticmethod
    def _to_polygon_coords(fence) -> list[tuple[float, float]]:
        """把围栏数据统一转成 Shapely Polygon 可接受的 (x, y) 列表"""
        if hasattr(fence, "tolist"):
            fence = fence.tolist()
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        堆煤检测入口。

        逻辑：
        1. 过滤 coal / coal pile 目标。
        2. 将围栏转为 Shapely Polygon。
        3. 对每个煤堆目标，解码 RLE mask 得到二值图并提取轮廓。
        4. 将轮廓转换为全局坐标多边形，检查是否与任一围栏相交。
        5. 相交则返回该煤堆 Box 作为违规目标。

        :param predictions: Box 对象列表
        :param fences: 电子围栏坐标列表，支持 numpy 数组或列表格式
        :param device_id: 设备ID
        :param image_width: 原图宽度，用于 mask 解码
        :param image_height: 原图高度，用于 mask 解码
        :return: 违规目标 Box 列表
        """
        if fences is None or fences == []:
            return []

        if image_width <= 0 or image_height <= 0:
            logger.warning(f"[device={device_id}] 无法确定图像尺寸，跳过堆煤检测")
            return []

        # 构建围栏多边形
        fence_polys = []
        for i, fence in enumerate(fences):
            coords = self._to_polygon_coords(fence)
            if len(coords) < 3:
                logger.warning(f"[device={device_id}] 围栏-{i} 坐标点不足3个，跳过: {coords}")
                continue
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                fence_polys.append(poly)
            except Exception as e:
                logger.error(f"[device={device_id}] 围栏-{i} 创建 Polygon 失败: {e}")
                continue

        if not fence_polys:
            logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过堆煤检测")
            return []

        # 过滤煤堆目标
        coal_boxes = label_filter(predictions, ["coal", "coal pile"])
        coal_boxes = score_filter(coal_boxes, self.min_score)

        result = []
        used_coals = set()

        for coal in coal_boxes:
            cid = id(coal)
            if cid in used_coals:
                continue

            # 解码 mask
            mask_array = coal.compute_mask_array(image_width, image_height)
            if mask_array is None:
                # 无 mask 时退化为 bounding box 与围栏相交判断
                box_poly = Polygon([
                    (coal.box[0], coal.box[1]),
                    (coal.box[2], coal.box[1]),
                    (coal.box[2], coal.box[3]),
                    (coal.box[0], coal.box[3]),
                ])
                if not box_poly.is_valid:
                    box_poly = box_poly.buffer(0)

                for fence_poly in fence_polys:
                    if box_poly.intersects(fence_poly) and box_poly.intersection(fence_poly).area > 0:
                        result.append(coal)
                        used_coals.add(cid)
                        break
                continue

            # 从二值 mask 中提取外轮廓
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            is_violation = False
            for contour in contours:
                if contour.shape[0] < 3:
                    continue

                # 轮廓点转换为全局坐标
                contour_pts = contour.reshape(-1, 2)
                try:
                    mask_poly = Polygon(contour_pts)
                    if not mask_poly.is_valid:
                        mask_poly = mask_poly.buffer(0)
                except Exception:
                    continue

                # 检查 mask 多边形是否与任一围栏相交
                for fence_poly in fence_polys:
                    if mask_poly.intersects(fence_poly) and mask_poly.intersection(fence_poly).area > 0:
                        is_violation = True
                        break

                if is_violation:
                    break

            if is_violation:
                logger.warning(
                    f"[device={device_id}] 堆煤违规触发 | "
                    f"label={coal.label}, box={coal.box}, score={coal.score:.3f}"
                )
                result.append(coal)
                used_coals.add(cid)

        return result
