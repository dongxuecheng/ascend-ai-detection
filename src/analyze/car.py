# 车辆违规检测：检测人员是否在货车车厢内
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union

from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("car")


class CarDetector:
    """
    车辆违规检测器：检测人员是否在货车车厢内
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

    @staticmethod
    def _to_polygon_coords(fence) -> list[tuple[float, float]]:
        """
        将 fence 数据统一转换为 Polygon 可接受的坐标列表。
        支持 numpy 数组、列表等多种格式。
        """
        if hasattr(fence, 'tolist'):
            fence = fence.tolist()
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    @staticmethod
    def _mask_to_polygon(box: Box, image_width: int, image_height: int, use_convex_hull: bool = False):
        """
        将 Box 的 RLE mask 解码并转为 Shapely 几何体。

        参数:
        - box: 包含 mask 的 Box 对象
        - image_width: 原图宽度
        - image_height: 原图高度
        - use_convex_hull: 是否使用凸包。开启后会将 U型/断裂的 Mask 包裹成一个整体多边形，
                           适合“车厢把人抠掉”的场景，确保人能在车厢多边形内部。
        """
        try:
            mask_array = box.compute_mask_array(image_width, image_height)
            if mask_array is None:
                return None

            # 提取外轮廓，忽略内部空洞
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            polys = []
            for cnt in contours:
                # 过滤掉噪点
                if cv2.contourArea(cnt) < 10:
                    continue

                # reshape 保证形状为 (N, 2)，避免 squeeze 在 N 较小时产生标量/一维数组
                contour = cnt.reshape(-1, 2)
                if len(contour) >= 3:
                    poly = Polygon(contour)
                    # 修复可能自交的无效多边形
                    poly = poly.buffer(0) if not poly.is_valid else poly
                    polys.append(poly)

            if not polys:
                return None

            # 合并所有碎片（断裂 Mask 会生成 MultiPolygon）
            final_geom = unary_union(polys)

            # 对 U型/C型 缺口使用凸包补全
            if use_convex_hull:
                final_geom = final_geom.convex_hull

            return final_geom

        except Exception as e:
            logger.warning(f"解析多边形失败: {e}")
            return None

    @staticmethod
    def _compute_image_size(predictions: list[Box], image_width: int, image_height: int) -> tuple[int, int]:
        """
        如果外部未传入图像尺寸，则根据所有 Box 的最大坐标估算。
        """
        if image_width > 0 and image_height > 0:
            return image_width, image_height
        if not predictions:
            return image_width, image_height
        max_x = max(int(b.box[2]) for b in predictions)
        max_y = max(int(b.box[3]) for b in predictions)
        return max(image_width, max_x), max(image_height, max_y)

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0
    ) -> list[Box]:
        """
        检测人员是否在围栏内的货车车厢中。

        逻辑：
        1. 围栏内存在 truck bed 才触发。
        2. 通过 mask 获取 truck bed 的轮廓，判断人员是否在其边界内。
        3. 只有包含 leg 的完整 person 才参与判定，避免半身/截断目标误报。
        4. 返回违规人员，并附带现场相关的 truck bed（方便前端画框或后续查看）。

        :param predictions: Box 对象列表，包含 label、score、box、mask 等信息
        :param fences: 区域坐标列表，支持 numpy 数组或列表格式
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :return: 违规目标 Box 列表（人员 + 相关 truck bed）
        """
        if fences is None or fences == []:
            return []

        image_width, image_height = self._compute_image_size(predictions, image_width, image_height)
        if image_width <= 0 or image_height <= 0:
            logger.warning(f"[device={device_id}] 无法确定图像尺寸，跳过车辆违规检测")
            return []

        # 构建围栏多边形
        fence_polys = []
        for i, fence in enumerate(fences):
            coords = self._to_polygon_coords(fence)
            if len(coords) < 3:
                logger.warning(f"[device={device_id}] 围栏-{i} 坐标点不足3个，跳过: {coords}")
                continue
            try:
                fence_polys.append(Polygon(coords).buffer(0))
            except Exception as e:
                logger.error(f"[device={device_id}] 围栏-{i} 创建 Polygon 失败: {e}, coords={coords}")
                continue

        if not fence_polys:
            logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过车辆违规检测")
            return []

        # 按类别过滤
        car_boxes = label_filter(predictions, ['truck bed'])
        person_boxes = label_filter(predictions, ['person'])
        leg_boxes = label_filter(predictions, ['leg'])

        # 基础过滤
        car_boxes = score_filter(car_boxes, 0.5)
        person_boxes = score_filter(person_boxes, 0.5)
        leg_boxes = score_filter(leg_boxes, 0.5)

        # 解析车辆 mask，筛选围栏内的 truck bed
        valid_truck_beds = []
        for car in car_boxes:
            car_poly = self._mask_to_polygon(car, image_width, image_height, use_convex_hull=True)
            if car_poly is None:
                continue

            for fence in fence_polys:
                if not car_poly.intersects(fence):
                    continue
                inter_area = car_poly.intersection(fence).area
                overlap_ratio = inter_area / car_poly.area if car_poly.area > 0 else 0.0
                if overlap_ratio > 0.7:
                    valid_truck_beds.append({'box': car, 'poly': car_poly})
                    break  # 命中任一围栏即视为有效

        # 围栏内没有 truck bed，不触发后续判定
        if not valid_truck_beds:
            return []

        # 解析人员 mask
        persons_detected = []
        for person in person_boxes:
            person_poly = self._mask_to_polygon(person, image_width, image_height, use_convex_hull=False)
            if person_poly is not None:
                persons_detected.append({'box': person, 'poly': person_poly})

        # 解析腿部 mask
        legs_detected = []
        for leg in leg_boxes:
            leg_poly = self._mask_to_polygon(leg, image_width, image_height, use_convex_hull=False)
            if leg_poly is not None:
                legs_detected.append({'box': leg, 'poly': leg_poly})

        # 核心判定：判断完整人员是否在 truck bed 内
        final_violators = []
        for p_obj in persons_detected:
            p_poly = p_obj['poly']
            p_box = p_obj['box']

            # 只保留完整的人：要求存在 leg 与其高度重叠
            is_full_person = False
            for leg_obj in legs_detected:
                leg_poly = leg_obj['poly']
                if not p_poly.intersects(leg_poly):
                    continue
                inter_area = p_poly.intersection(leg_poly).area
                leg_overlap_ioa = inter_area / leg_poly.area if leg_poly.area > 0 else 0.0
                if leg_overlap_ioa >= 0.9:
                    is_full_person = True
                    break
            if not is_full_person:
                continue

            # 判断该人员是否在任一有效 truck bed 内
            is_in_truck_bed = False
            for tb_obj in valid_truck_beds:
                tb_poly = tb_obj['poly']
                if not p_poly.intersects(tb_poly):
                    continue
                inter_area = p_poly.intersection(tb_poly).area
                overlap_ioa = inter_area / p_poly.area if p_poly.area > 0 else 0.0
                if overlap_ioa > 0.6:
                    logger.info(
                        f"[device={device_id}] 发现违规：person 在 truck bed 内部，"
                        f"重合度 {overlap_ioa:.2%}"
                    )
                    is_in_truck_bed = True
                    break

            if is_in_truck_bed:
                final_violators.append(p_box)

        # 保存历史状态
        if device_id:
            self._history[device_id] = final_violators

        # 返回违规人员 + 相关 truck bed，方便前端画框
        if final_violators:
            return final_violators + [tb['box'] for tb in valid_truck_beds]

        return []
