import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("empty_truck")


class EmptyTruckDetector:
    """
    空车检查检测器：基于多边形（Mask）检测 truck bed 内是否存在煤块残留。
    要求：
      1. 车厢面积 > 图像面积的 85%（避免误检小目标）
      2. 煤块与车厢的 IoF（交集面积 / 煤块面积）超过阈值，认为车厢非空
    """

    def __init__(
        self,
        iof_threshold: float = 0.99,
        min_score: float = 0.7,
        min_contour_area: float = 1000.0,
        bed_area_ratio_threshold: float = 0.5,
    ):
        """
        :param iof_threshold: 煤块在车厢内的面积占比阈值，超过此值认为该煤块属于该车厢
        :param min_score: 目标最低置信度
        :param min_contour_area: 最小轮廓面积（过滤噪点）
        :param bed_area_ratio_threshold: 车厢面积占整图面积的最小比例，低于此值不触发检测
        """
        self.iof_threshold = iof_threshold
        self.min_score = min_score
        self.min_contour_area = min_contour_area
        self.bed_area_ratio_threshold = bed_area_ratio_threshold
    
    @staticmethod
    def _draw_polygons(
        image: np.ndarray,
        fence_polys: list[Polygon],
        belt_poly: Polygon,
        outside_ratio: float,
        is_violation: bool,
        color_fence=(0, 255, 0),
        color_belt=(255, 0, 0),
        color_violation=(0, 0, 255),
    ):
        """在图像上绘制围栏和皮带多边形，并显示越界比例。"""
        img_copy = image.copy()
        # 绘制围栏（绿色）
        for poly in fence_polys:
            pts = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], isClosed=True, color=color_fence, thickness=2)

        # 绘制皮带多边形（蓝色或红色）
        if belt_poly is not None:
            pts = np.array(belt_poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            color = color_violation if is_violation else color_belt
            # cv2.fillPoly(img_copy, [pts], color=color, lineType=cv2.LINE_AA)
            cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=5)
            # 显示越界比例
            text = f"outside_ratio: {outside_ratio:.2%}"
            cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img_copy

    @staticmethod
    def _to_polygon_coords(fence) -> list[tuple[float, float]]:
        """把围栏坐标统一转成 Shapely Polygon 可接受的 (x, y) 列表（保留未用）"""
        if hasattr(fence, "tolist"):
            fence = fence.tolist()
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    def _mask_to_polygon(self, box: Box, image_width: int, image_height: int):
        """
        将 Box 的 mask 解码并转为 Shapely 多边形，返回 (polygon, area)。
        对 U 型/C 型缺口使用凸包补全。
        """
        mask_array = box.compute_mask_array(image_width, image_height)
        if mask_array is None:
            return None, 0.0

        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        polys = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue

            # 轮廓点转为 (x, y) 列表
            contour = np.squeeze(cnt)
            if len(contour) < 3:
                continue

            poly = Polygon(contour)
            poly = poly.buffer(0) if not poly.is_valid else poly
            polys.append(poly)

        if not polys:
            return None, 0.0

        mask_poly = unary_union(polys)
        # 凸包补全，处理 U 型/C 型断裂
        mask_poly = mask_poly.convex_hull

        return mask_poly, float(mask_poly.area)

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
        frame: np.ndarray = None,
    ) -> list[Box]:
        """
        空车检查入口（基于多边形 IoF）。

        :param predictions: Box 对象列表
        :param fences: 区域坐标列表（预留）
        :param device_id: 设备ID
        :param image_width: 原图宽度（必须提供，用于 mask 解码）
        :param image_height: 原图高度（必须提供）
        :return: 违规目标 Box 列表，label 为 "coal residue"
        """
        if image_width <= 0 or image_height <= 0:
            logger.error(f"[device={device_id}] 无效的图像尺寸，无法解码 mask")
            return []

        # 过滤出车厢和煤块
        truck_beds = label_filter(predictions, ["truck"])
        truck_beds = score_filter(truck_beds, 0.75)

        coals = label_filter(predictions, ["dark coal residue", "coal residue"])
        coals = score_filter(coals, self.min_score)

        if not truck_beds or not coals:
            return []

        # 预计算所有煤块的多边形和面积（只算一次）
        coal_polys = []
        for coal in coals:
            poly, area = self._mask_to_polygon(coal, image_width, image_height)
            if poly is not None and area > 0:
                coal_polys.append((coal, poly, area))
        if not coal_polys:
            return []
        result = []
        used_coals = set()
        image_area = image_width * image_height
        # img_draw = frame
        for truck in truck_beds:
            # 1. 计算车厢多边形及面积
            bed_poly, bed_area = self._mask_to_polygon(truck, image_width, image_height)
            if bed_poly is None or bed_area <= 0:
                continue

            # 2. 检查车厢面积是否占图像面积 ≥ 阈值
            if bed_area / image_area < self.bed_area_ratio_threshold:
                logger.info(
                    f"[device={device_id}] 车厢面积占比 {bed_area/image_area:.2%} < "
                    f"{self.bed_area_ratio_threshold:.0%}，跳过该车厢"
                )
                continue
            
            # img_draw = self._draw_polygons(img_draw, [], bed_poly, 0.1, True)
            

            # 3. 遍历煤块，计算 IoF
            for coal, coal_poly, coal_area in coal_polys:
                if coal_area < 140*140:
                    continue
                # img_draw = self._draw_polygons(img_draw, [], coal_poly, 0.1, False)
                cid = id(coal)
                if cid in used_coals:
                    continue

                # 计算交集面积
                inter_poly = bed_poly.intersection(coal_poly)
                inter_area = inter_poly.area

                # IoF = 交集面积 / 煤块面积
                iof = inter_area / coal_area if coal_area > 0 else 0.0
                # print(iof)

                if iof > self.iof_threshold:
                    # 构造违规 Box（复制原煤块信息）
                    violation = Box(
                        label="coal residue",
                        score=coal.score,
                        box=list(coal.box),
                        mask=coal.mask,
                    )
                    logger.warning(
                        f"[device={device_id}] 空车检查违规触发 | 车厢内发现煤块 | "
                        f"iof={iof:.2%}, bed_area_ratio={bed_area/image_area:.2%}, "
                        f"truck_box={truck.box}, coal_box={coal.box}"
                    )
                    result.append(violation)
                    used_coals.add(cid)

        # 若存在多个违规煤块，只返回面积最大的一个（根据注释需求）
        if result:
            result = sorted(
                result,
                key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]),
                reverse=True
            )
            result = result[:1]
        # cv2.imwrite("result.jpg", img_draw)
        return result
