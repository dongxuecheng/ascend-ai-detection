# 车辆闯入区域检测（算法码 62）
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("car_zone")


class CarZoneDetector:
    """
    车辆闯入区域检测器（算法码 62）。

    逻辑（参考皮带跑偏 belt_deviation 的 mask 相交判定）：
      1. 过滤出 truck 目标（需要在 algorithms.yaml 中为 62 配置 return_mask: true）。
      2. 把 truck 的 RLE mask 解码成二值图，提取轮廓后取凸包，得到一个能完整包含 mask 的单个多边形
         （mask 断开成多块时同样适用）。
      3. 判断车辆 mask 多边形与围栏多边形是否相交，
         相交面积占“两者中较小面积”的比例超过阈值即判定为闯入。
      4. 返回闯入的车辆 Box，并把车辆 mask 光栅化挂回 Box，便于上层直接绘制。
    """

    def __init__(
        self,
        truck_min_score: float = 0.75,
        truck_min_area: float = 10000.0,
        overlap_ratio_thresh: float = 0.01,
    ):
        """
        :param truck_min_score: truck 最低置信度
        :param truck_min_area: truck 最小面积（像素），过滤小目标误报
        :param overlap_ratio_thresh: 相交面积 / min(车辆 mask 面积, 围栏面积) 的阈值，
                                     超过即判定为闯入（默认 0.01 与原来的 bbox fence_iom 口径保持一致，偏灵敏）
        """
        self.truck_min_score = truck_min_score
        self.truck_min_area = truck_min_area
        self.overlap_ratio_thresh = overlap_ratio_thresh

    @staticmethod
    def _to_polygon_coords(fence):
        """
        将 fence 数据统一转换为 Polygon 可接受的坐标列表。
        支持 numpy 数组、列表等多种格式。
        """
        if hasattr(fence, 'tolist'):
            fence = fence.tolist()
        # 确保每个点都是 tuple (x, y)
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    def _mask_to_polygon(self, box: Box, image_width: int, image_height: int):
        """
        将 Box 的 mask 解码并转为 Shapely 多边形，返回 (polygon, area)。

        对 mask 提取全部外轮廓，合并后取凸包，保证返回的是单个多边形，
        且无论 mask 是否断开、形状多不规则，该多边形都能完整包含 mask。
        """
        mask_array = box.compute_mask_array(image_width, image_height)
        if mask_array is None:
            return None, 0.0

        # 统一为 0/1
        mask = (mask_array > 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        polys = []
        for cnt in contours:
            contour = np.squeeze(cnt)
            if len(contour) < 3:      # 少于 3 个点构不成多边形
                continue
            poly = Polygon(contour)
            poly = poly.buffer(0) if not poly.is_valid else poly
            polys.append(poly)

        if not polys:
            return None, 0.0

        # 合并后取凸包：无论轮廓是否互不相连，结果都是单个多边形且完整包住 mask
        truck_poly = unary_union(polys).convex_hull

        return truck_poly, float(truck_poly.area)

    @staticmethod
    def _iter_polygon_parts(geom):
        """
        把 Polygon / MultiPolygon / GeometryCollection 统一展开为单个 Polygon 迭代器。

        凸包与外接多边形通常已经是单个 Polygon，这里做兼容处理：
        遇到 MultiPolygon / GeometryCollection 时先拆分再访问 `exterior`。
        """
        if geom is None or getattr(geom, "is_empty", True):
            return
        if hasattr(geom, "geoms"):
            for sub in geom.geoms:
                yield from CarZoneDetector._iter_polygon_parts(sub)
        elif hasattr(geom, "exterior"):
            yield geom

    @staticmethod
    def _polygon_to_mask(poly, image_width: int, image_height: int) -> np.ndarray:
        """把 Shapely Polygon / MultiPolygon 光栅化为与图像同尺寸的二值 mask，便于上层直接画 mask。"""
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        for part in CarZoneDetector._iter_polygon_parts(poly):
            ext = np.array(part.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            if len(ext) < 3:
                continue
            cv2.fillPoly(mask, [ext], 1)
            for interior in part.interiors:
                hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                if len(hole) < 3:
                    continue
                cv2.fillPoly(mask, [hole], 0)
        return mask

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断车辆是否闯入指定区域
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，支持 numpy 数组或列表格式
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :return: 闯入区域的车辆 Box 列表
        """
        if fences is None or fences == []:
            return []

        if image_width <= 0 or image_height <= 0:
            logger.warning(f"[device={device_id}] 无法确定图像尺寸，跳过车辆闯区域检测")
            return []

        result: list[Box] = []
        fence_polygons = []
        for i, fence in enumerate(fences):
            coords = self._to_polygon_coords(fence)
            if len(coords) < 3:
                logger.warning(f"[device={device_id}] 围栏-{i} 坐标点不足3个，无法构成多边形，跳过: {coords}")
                continue
            try:
                poly = Polygon(coords)
                poly = poly.buffer(0) if not poly.is_valid else poly
                fence_polygons.append(poly)
            except Exception as e:
                logger.error(f"[device={device_id}] 围栏-{i} 创建 Polygon 失败: {e}, coords={coords}")
                continue

        if not fence_polygons:
            logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过车辆闯区域检测")
            return []

        truck_boxes = label_filter(predictions, ['truck'])
        truck_boxes = [p for p in truck_boxes if p.source == "SAM3"]
        truck_boxes = area_filter(truck_boxes, self.truck_min_area)
        truck_boxes = score_filter(truck_boxes, self.truck_min_score)

        # 车辆 mask 多边形与围栏做相交判断
        used_ids = set()
        for truck in truck_boxes:
            tid = id(truck)
            if tid in used_ids:
                continue

            truck_poly, truck_area = self._mask_to_polygon(truck, image_width, image_height)
            if truck_poly is None or truck_area <= 0:
                logger.debug(f"[device={device_id}] 车辆-{tid} mask 无法转为多边形，跳过")
                continue

            for f_idx, fence_poly in enumerate(fence_polygons):
                if not truck_poly.intersects(fence_poly):
                    continue

                # 相交面积占“两者中较小面积”的比例：
                # 围栏比车大时衡量“车有多少进了区域”，围栏比车小时衡量“区域被车占了多少”
                intersection_area = float(truck_poly.intersection(fence_poly).area)
                fence_area = float(fence_poly.area)
                min_area = min(truck_area, fence_area)
                overlap_ratio = intersection_area / min_area if min_area > 0 else 0.0

                logger.info(
                    f"[device={device_id}] 车辆-{tid} 与围栏-{f_idx} 相交 | "
                    f"交集={intersection_area:.0f}, 车辆mask={truck_area:.0f}, 围栏={fence_area:.0f}, "
                    f"占比={overlap_ratio:.2%}, 阈值={self.overlap_ratio_thresh:.2%}"
                )

                if overlap_ratio >= self.overlap_ratio_thresh:
                    # 把闯入车辆的 mask 光栅化后挂到返回的 Box 上，便于上层直接绘制 mask
                    truck.mask_array = self._polygon_to_mask(truck_poly, image_width, image_height)
                    logger.warning(
                        f"[device={device_id}] 车辆闯入区域触发 | 围栏-{f_idx}, "
                        f"box={truck.box}, score={truck.score:.3f}, 占比={overlap_ratio:.2%}"
                    )
                    result.append(truck)
                    used_ids.add(tid)
                    break  # 命中任一围栏即报出

        return result
