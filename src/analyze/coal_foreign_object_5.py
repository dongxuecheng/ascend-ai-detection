# 煤流异物检测
from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import score_filter, label_filter, area_filter
from utils.logger import setup_logger

logger = setup_logger("coal_foreign_object_5")


class CoalForeignObjectDetector5:
    """
    煤流异物检测器（算法码 50）。

    逻辑：
      1. 过滤出 SAM3 检测到的异物候选框（stone / plastic bag 等）。
      2. 将电子围栏转换为 Shapely 多边形。
      3. 判断异物框是否完全位于电子围栏内。
      4. 只要电子围栏内存在任一异物，即以该异物框作为违规返回。

    返回的 Box 为违规异物框（label 统一为 foreign）。
    """

    def __init__(
        self,
        foreign_min_score: float = 0.4,
        foreign_min_area: float = 1000.0,
    ):
        """
        :param foreign_object_labels: 视为异物候选的 SAM3 标签，默认 ["light-grey rock", "plastic bag", "wooden stick"]
        :param foreign_min_score: 异物候选框最低置信度
        :param foreign_min_area: 异物候选框最小面积（像素）
        """
        self.foreign_object_labels = ["light-grey rock", "plastic bag", "wooden stick"]
        self.foreign_min_score = foreign_min_score
        self.foreign_min_area = foreign_min_area

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
        煤流异物检测入口。

        :param predictions: 当前帧所有检测框（SAM3 返回的 stone / plastic bag 等）
        :param fences: 电子围栏坐标列表，支持 numpy 数组或列表格式
        :param device_id: 设备ID
        :param image_width: 原图宽度（当前不使用 mask，保留以保持接口统一）
        :param image_height: 原图高度（当前不使用 mask，保留以保持接口统一）
        :return: 落在电子围栏内的违规异物 Box 列表
        """
        if fences is None or fences == []:
            return []

        # 1. 构建围栏多边形
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
            logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过煤流异物检测")
            return []

        # 2. 过滤异物候选框（SAM3 检测的 stone / plastic bag 等）
        # print(predictions)
        foreign_boxes = label_filter(predictions, self.foreign_object_labels)
        foreign_boxes = score_filter(foreign_boxes, self.foreign_min_score)
        foreign_boxes = area_filter(foreign_boxes, self.foreign_min_area)
        if not foreign_boxes:
            logger.info(f"[device={device_id}] 煤流异物检测 | 未检测到异物候选框 {self.foreign_object_labels}")
            return []

        # 3. 判断异物框是否完全位于电子围栏内
        violations = []
        for foreign in foreign_boxes:
            x1, y1, x2, y2 = foreign.box
            box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            for fence_poly in fence_polys:
                if not fence_poly.contains(box_poly):
                    continue
                logger.warning(
                    f"[device={device_id}] 煤流异物 | {foreign.label}, "
                    f"box={foreign.box}, score={foreign.score:.3f}"
                )
                foreign.label = "foreign"
                violations.append(foreign)
                break

        return violations
