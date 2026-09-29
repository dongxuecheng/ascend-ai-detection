### 登高无安全带检测（专门用于算法码 58）
import numpy as np
import cv2
from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("height_work")


class HeightWorkDetector:
    """
    登高无安全带检测器（专门用于算法码 58）：

    无围栏时：检测到 person-on-ladder / person-on-scaffolding 与梯子/脚手架相交，
    且未佩戴 safety harness / harness 时，判定为违规。

    传入围栏时：不再依赖梯子/脚手架，只要围栏内的 person 未佩戴安全带即判定为违规。

    安全带置信度阈值单独设置（默认 0.3），便于在低置信度下也能识别到安全带。
    """

    def __init__(
        self,
        person_score_thresh: float = 0.6,
        equipment_score_thresh: float = 0.6,
        belt_score_thresh: float = 0.3,
        expand_ratio: float = 0.0,
        dark_thresh: int = 60,
        backlit_dark_ratio: float = 0.8,
        bright_thresh: int = 200,
        backlit_bright_ratio: float = 0.8,
        enable_backlight_filter: bool = True,
        grid_size: int = 3,
        grid_min_backlit_cells: int = 5,
        min_person_height_ratio: float = 0.25,
        fence_iom_thresh: float = 0.3,
    ):
        """
        :param person_score_thresh: 登高人员框最低置信度
        :param equipment_score_thresh: 梯子/脚手架框最低置信度
        :param belt_score_thresh: 安全带置信度阈值（设置较低，便于检出微弱安全带）
        :param expand_ratio: 报警框四边扩展比例（相对人员框宽高），默认 0.5，即上下左右各扩展 50%
        :param dark_thresh: 判定“暗像素”的灰度阈值（0-255），低于该值的像素视为暗像素
        :param backlit_dark_ratio: 判定区域内暗像素占比达到该值即判定为逆光（过暗）
        :param bright_thresh: 判定“亮像素”的灰度阈值（0-255），高于该值的像素视为亮像素
        :param backlit_bright_ratio: 判定区域内亮像素占比达到该值即判定为过曝（过亮）
        :param enable_backlight_filter: 是否启用光照异常区域排除（关闭则保持原逻辑）
        :param grid_size: 光照异常判断的网格边长，默认 3，即把区域拆分成 3x3 九宫格
        :param grid_min_backlit_cells: 判定为光照异常所需的最少异常格子数，默认 5（九宫格的一半）
        :param min_person_height_ratio: 登高人员框高度占画面高度的最小比例，低于该比例视为低处误检直接排除，默认 0.25
        :param fence_iom_thresh: 围栏模式下判断人员是否在围栏内的 IoM 阈值，默认 0.3
        """
        self.person_score_thresh = person_score_thresh
        self.equipment_score_thresh = equipment_score_thresh
        self.belt_score_thresh = belt_score_thresh
        self.expand_ratio = expand_ratio

        # 光照异常排除参数（传统图像算法）
        self.dark_thresh = dark_thresh
        self.backlit_dark_ratio = backlit_dark_ratio
        self.bright_thresh = bright_thresh
        self.backlit_bright_ratio = backlit_bright_ratio
        # self.enable_backlight_filter = enable_backlight_filter
        self.enable_backlight_filter = False 

        # 九宫格光照异常判断参数
        self.grid_size = grid_size
        self.grid_min_backlit_cells = grid_min_backlit_cells

        # 登高人员高度过滤参数
        self.min_person_height_ratio = min_person_height_ratio

        # 围栏模式人员包含判断阈值
        self.fence_iom_thresh = fence_iom_thresh

    @staticmethod
    def _intersection_area(box1: list, box2: list) -> float:
        """计算两个框的交集面积"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _to_polygon_coords(fence) -> list[tuple[float, float]]:
        """把围栏坐标统一转成 Shapely Polygon 可接受的 (x, y) 列表。"""
        if hasattr(fence, "tolist"):
            fence = fence.tolist()
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    # ---------- 光照异常区域判断（传统图像算法） ----------
    def _is_backlit_region(self, frame: np.ndarray, box: Box) -> bool:
        """
        用传统图像算法判断某目标区域是否光照异常（过暗或过亮）。
        思路：把目标区域（直接使用目标框，不外扩）转灰度，拆分为 grid_size x grid_size 九宫格；
        对每个格子统计暗像素占比与亮像素占比，
        若过半格子（默认 5/9）过暗（逆光/阴影）或过亮（过曝），
        说明该区域整体光照异常，检测结果不可靠，应排除。
        """
        if frame is None:
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        rh, rw = gray.shape[:2]
        if rh == 0 or rw == 0:
            return False

        # 拆分为 grid_size x grid_size 九宫格（np.array_split 自动处理不能整除的行列）
        row_groups = np.array_split(np.arange(rh), self.grid_size)
        col_groups = np.array_split(np.arange(rw), self.grid_size)

        dark_cells = 0
        bright_cells = 0
        for row_idx in row_groups:
            for col_idx in col_groups:
                cell = gray[np.ix_(row_idx, col_idx)]
                if cell.size == 0:
                    continue
                if float((cell < self.dark_thresh).mean()) >= self.backlit_dark_ratio:
                    dark_cells += 1
                elif float((cell > self.bright_thresh).mean()) >= self.backlit_bright_ratio:
                    bright_cells += 1

        backlit_cells = dark_cells + bright_cells
        if backlit_cells >= self.grid_min_backlit_cells:
            logger.info(
                f"[登高] 光照异常区域 | box={box.box} | "
                f"九宫格异常格子={backlit_cells}/{self.grid_size * self.grid_size} "
                f"(过暗={dark_cells}, 过亮={bright_cells}) "
                f">= 阈值 {self.grid_min_backlit_cells}"
            )
            return True
        return False

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
        登高无安全带检测入口。

        逻辑：
        1. 过滤 safety harness / harness（安全带，使用较低阈值）。
        2. 若传入围栏：
           - 过滤普通 person，只要 person 位于围栏内且未佩戴安全带即判定为违规；
           - 不再依赖 person-on-ladder / ladder / scaffolding。
        3. 若未传入围栏：
           - 过滤 person-on-ladder / person-on-scaffolding（登高人员）与 ladder / scaffolding（梯子/脚手架）；
           - 判断登高人员是否与梯子/脚手架有交集。
        4. 对登高人员，判断其高度是否小于画面高度的 1/4，小于则直接排除（低处误检）。
        5. 对目标人员，检查是否有安全带与其有交集。
        6. 无安全带时，返回人员框上下左右各扩展后的报警框，label 为 "alarm-no-safety-belt"。

        :param predictions: Box 对象列表
        :param fences: 区域坐标列表；传入时启用围栏模式（围栏内 person 无安全带即报警）
        :param device_id: 设备ID
        :param image_width: 原图宽度，用于扩展框边界裁剪
        :param image_height: 原图高度，用于扩展框边界裁剪
        :param frame: 当前帧图像（BGR），用于光照异常区域判断；为 None 时不排除
        :return: 违规目标 Box 列表
        """
        belt_boxes = label_filter(
            predictions,
            ["safety harness", "harness"],
        )
        belt_boxes = score_filter(belt_boxes, self.belt_score_thresh)

        # 是否传入围栏：传入则按“围栏内人员无安全带即报警”处理，不再依赖梯子/脚手架
        has_fences = fences is not None and len(fences) > 0

        if has_fences:
            fence_polygons = []
            for i, fence in enumerate(fences):
                coords = self._to_polygon_coords(fence)
                if len(coords) < 3:
                    logger.warning(f"[device={device_id}] 围栏-{i} 坐标点不足3个，跳过")
                    continue
                try:
                    poly = Polygon(coords)
                    poly = poly.buffer(0) if not poly.is_valid else poly
                    fence_polygons.append(poly)
                except Exception as e:
                    logger.error(f"[device={device_id}] 围栏-{i} 创建 Polygon 失败: {e}")
                    continue

            if not fence_polygons:
                logger.warning(f"[device={device_id}] 没有有效的围栏多边形，跳过登高检测")
                return []

            person_boxes = label_filter(predictions, ["person"])
            person_boxes = score_filter(person_boxes, self.person_score_thresh)
            equipment_boxes = []
            if not person_boxes:
                return []
        else:
            person_boxes = label_filter(
                predictions,
                ["person-on-ladder", "person-on-scaffolding"],
            )
            person_boxes = score_filter(person_boxes, self.person_score_thresh)

            equipment_boxes = label_filter(
                predictions,
                ["ladder", "scaffolding"],
            )
            equipment_boxes = score_filter(equipment_boxes, self.equipment_score_thresh)

            if not person_boxes or not equipment_boxes:
                return []
            fence_polygons = []

        # 图像边界，用于扩展框裁剪
        max_x = float(image_width - 1) if image_width > 0 else float("inf")
        max_y = float(image_height - 1) if image_height > 0 else float("inf")

        result = []
        used_persons = set()

        for person in person_boxes:
            pid = id(person)
            if pid in used_persons:
                continue

            # A0. 高度过滤（仅无围栏模式）：登高人员框高度小于画面高度的 1/4 时，视为低处误检，直接排除
            if not has_fences and image_height > 0:
                person_h = person.box[3] - person.box[1]
                min_height = image_height * self.min_person_height_ratio
                if person_h < min_height:
                    logger.info(
                        f"[device={device_id}] 登高人员高度不足排除 | box={person.box} | "
                        f"人员框高度={person_h:.1f} < {min_height:.1f} "
                        f"(画面高度={image_height})"
                    )
                    continue

            # A. 判断人员是否位于检测区域：
            #    围栏模式看人员是否在围栏内；否则看是否与梯子/脚手架相交
            if has_fences:
                if not any(person.fence_iom(poly) > self.fence_iom_thresh for poly in fence_polygons):
                    continue
            else:
                target_equipment = None
                for equipment in equipment_boxes:
                    if self._intersection_area(person.box, equipment.box) > 0:
                        target_equipment = equipment
                        break  # 只要在一个设备上就算

                if target_equipment is None:
                    continue

            # B. 判断该人员是否佩戴安全带
            has_belt = False
            for belt in belt_boxes:
                if person.iom(belt) > 0.9:
                    has_belt = True
                    break

            # C. 在高处且没有安全带，触发报警
            if not has_belt:
                # 报警框：仅对登高人员框上下左右各扩展 expand_ratio，不合并设备框
                expanded = person.expand_box("all", self.expand_ratio)
                ex_x1 = max(0.0, expanded.box[0])
                ex_y1 = max(0.0, expanded.box[1])
                ex_x2 = min(max_x, expanded.box[2])
                ex_y2 = min(max_y, expanded.box[3])

                violation = Box(
                    label="alarm-no-safety-belt",
                    score=person.score,
                    box=[ex_x1, ex_y1, ex_x2, ex_y2],
                    mask=person.mask,
                )

                # 光照异常区域排除：传统算法判断报警区域是否过暗/过亮，是则跳过，避免误报
                if (
                    self.enable_backlight_filter
                    and frame is not None
                    and self._is_backlit_region(frame, violation)
                ):
                    logger.info(
                        f"[device={device_id}] 登高无安全带 光照异常区域排除 | "
                        f"box={violation.box}"
                    )
                    used_persons.add(pid)
                    continue

                logger.warning(
                    f"[device={device_id}] 登高无安全带违规触发 | "
                    f"box={violation.box}, score={violation.score:.3f}"
                )
                result.append(violation)
                used_persons.add(pid)

        return result
