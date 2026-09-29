from __future__ import annotations
from shapely.geometry import Point, Polygon
from utils.rle import rle_to_binary_mask
import numpy as np


class Box(object):
    # 类常量：位置名 -> 坐标计算函数（接收 box 元组/列表，返回 (x, y)）
    _POSITION_FUNCS = {
        'top_left':      lambda box: (box[0], box[1]),
        'top_right':     lambda box: (box[2], box[1]),
        'bottom_left':   lambda box: (box[0], box[3]),
        'bottom_right':  lambda box: (box[2], box[3]),
        'center':        lambda box: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        'top_center':    lambda box: ((box[0] + box[2]) / 2, box[1]),
        'bottom_center': lambda box: ((box[0] + box[2]) / 2, box[3]),
        'left_center':   lambda box: (box[0], (box[1] + box[3]) / 2),
        'right_center':  lambda box: (box[2], (box[1] + box[3]) / 2),
    }
    _DEFAULT_POS = 'bottom_center'   # 默认位置名

    def __init__(self, label: str, score: float, box: list, mask: dict | None = None, mask_array: np.ndarray | None = None, source="SAM3"):
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
        self.mask_array = mask_array  # 延迟计算的二值掩码数组
        self.source = source

    def compute_mask_array(self, image_width: int, image_height: int) -> np.ndarray | None:
        if self.mask_array is not None:
            return self.mask_array
        if self.mask is None:
            return None
        # 如果 mask 已经是 numpy 数组（如来自 intersection_mask / union_mask / complement_mask 的结果），直接使用
        if isinstance(self.mask, np.ndarray):
            self.mask_array = self.mask.astype(np.uint8)
            return self.mask_array
        # 防御性检查：如果不是预期的 RLE 字典，记录并返回 None
        if not isinstance(self.mask, dict):
            return None
        # 否则按 RLE 字典解码为局部掩码，再嵌入全局画布
        mask_arr = self.rle_to_mask()
        self.mask_array = np.zeros((image_height, image_width), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, self.box)
        self.mask_array[y1:y2, x1:x2] = mask_arr
        return self.mask_array
    
    # 计算mask交集
    def mask_intersection(self, other: 'Box', label: str, image_width: int, image_height: int) -> Box| None:
        mask1 = self.compute_mask_array(image_width, image_height)
        mask2 = other.compute_mask_array(image_width, image_height)
        if mask1 is None or mask2 is None:
            return None
        inter = np.logical_and(mask1, mask2).astype(np.uint8)
        ys, xs = np.where(inter > 0)
        if len(xs) == 0:
            return None
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return Box(label, 1.0, [x1, y1, x2, y2], mask=None, mask_array=inter)

    # 计算mask并集
    def mask_union(self, other: 'Box', label: str, image_width: int, image_height: int) -> Box | None:
        mask1 = self.compute_mask_array(image_width, image_height)
        mask2 = other.compute_mask_array(image_width, image_height)
        if mask1 is None and mask2 is None:
            return None
        union = np.logical_or(mask1, mask2).astype(np.uint8) if mask1 is not None and mask2 is not None else (mask1 if mask1 is not None else mask2)
        ys, xs = np.where(union > 0)
        if len(xs) == 0:
            return None
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return Box(label, 1.0, [x1, y1, x2, y2], mask=None, mask_array=union)

    # 计算mask差集（当前box减去另一个box）
    def mask_complement(self, other: 'Box', label: str, image_width: int, image_height: int) -> Box| None:
        mask1 = self.compute_mask_array(image_width, image_height)
        mask2 = other.compute_mask_array(image_width, image_height)
        if mask1 is None:
            return None
        if mask2 is None:
            ys, xs = np.where(mask1 > 0)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            return Box(label, 1.0, [x1, y1, x2, y2], mask=None, mask_array=mask1.copy())
        comp = np.logical_and(mask1, np.logical_not(mask2)).astype(np.uint8)
        ys, xs = np.where(comp > 0)
        if len(xs) == 0:
            return None
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return Box(label, 1.0, [x1, y1, x2, y2], mask=None, mask_array=comp)
    

    # 计算宽高比
    def whration(self):
        x1, y1, x2, y2 = self.box
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            return float('inf')
        return width / height

    def mask_iou(self, other: 'Box', image_width: int, image_height: int) -> float:
        mask1 = self.compute_mask_array(image_width, image_height)
        mask2 = other.compute_mask_array(image_width, image_height)
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def mask_iom(self, other: 'Box', image_width: int, image_height: int) -> float:
        mask1 = self.compute_mask_array(image_width, image_height)
        mask2 = other.compute_mask_array(image_width, image_height)
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2).sum()
        min_area = min(mask1.sum(), mask2.sum())
        if min_area == 0:
            return 0.0
        return intersection / min_area


    def point_in_fence(self, position: str, fence: Polygon) -> bool:
        # 获取对应的计算函数，若不存在则使用默认函数
        func = self._POSITION_FUNCS.get(position, self._POSITION_FUNCS[self._DEFAULT_POS])
        # 动态计算当前 box 下的坐标
        point = func(self.box)
        return fence.contains(Point(point))

    # top : 切出上半部分，bottom : 切出下半部分，left : 切出左半部分，right : 切出右半部分
    def cut_box(self, position: str, ratio: float = 0.5) -> 'Box':
        x1, y1, x2, y2 = self.box
        width = x2 - x1
        height = y2 - y1
        if position == 'top':
            # 切出上半部分：从底部往上切，保留顶部区域
            ny2 = y2 - height * ratio
            return Box(self.label, self.score, [x1, y1, x2, ny2], self.mask)
        elif position == 'bottom':
            # 切出下半部分：从顶部往下切，保留底部区域
            ny1 = y1 + height * ratio
            return Box(self.label, self.score, [x1, ny1, x2, y2], self.mask)
        elif position == 'left':
            nx2 = x1 + width * ratio
            return Box(self.label, self.score, [x1, y1, nx2, y2], self.mask)
        elif position == 'right':
            nx1 = x2 - width * ratio
            return Box(self.label, self.score, [nx1, y1, x2, y2], self.mask)
        else:
            raise ValueError(f"cut_box: 不支持的位置 '{position}'，请使用 'top'/'bottom'/'left'/'right'")

    # top : 向上扩展，bottom : 向下扩展，left : 向左扩展，right : 向右扩展，all : 四边同时扩展
    def expand_box(self, position: str, expand_ratio: float = 0.2) -> 'Box':
        x1, y1, x2, y2 = self.box
        width = x2 - x1
        height = y2 - y1
        if position == 'all':
            expand_x = width * expand_ratio
            expand_y = height * expand_ratio
            return Box(self.label, self.score, [x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y], self.mask)
        elif position == 'top':
            ny1 = y1 - height * expand_ratio
            return Box(self.label, self.score, [x1, ny1, x2, y2], self.mask)
        elif position == 'bottom':
            ny2 = y2 + height * expand_ratio
            return Box(self.label, self.score, [x1, y1, x2, ny2], self.mask)
        elif position == 'left':
            nx1 = x1 - width * expand_ratio
            return Box(self.label, self.score, [nx1, y1, x2, y2], self.mask)
        elif position == 'right':
            nx2 = x2 + width * expand_ratio
            return Box(self.label, self.score, [x1, y1, nx2, y2], self.mask)
        else:
            raise ValueError(f"expand_box: 不支持的位置 '{position}'，请使用 'top'/'bottom'/'left'/'right'/'all'")

    def area(self):
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)

    # 计算两个 Box 的交集面积
    def intersection(self, other: 'Box') -> float:
        x1, y1, x2, y2 = self.box
        x3, y3, x4, y4 = other.box

        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
            return 0

        return (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

    def iou(self, other: 'Box') -> float:
        intersection_area = self.intersection(other)
        union_area = self.area() + other.area() - intersection_area
        if union_area == 0:
            return 0
        return intersection_area / union_area

    def iom(self, other: 'Box') -> float:
        intersection_area = self.intersection(other)
        min_area = min(self.area(), other.area())
        if min_area == 0:
            return 0
        return intersection_area / min_area

    def fence_iou(self, fence: Polygon) -> float:
        x1, y1, x2, y2 = self.box
        box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        intersection_area = box_polygon.intersection(fence).area
        union_area = box_polygon.union(fence).area
        if union_area == 0:
            return 0
        return intersection_area / union_area

    def fence_iom(self, fence: Polygon) -> float:
        x1, y1, x2, y2 = self.box
        box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        intersection_area = box_polygon.intersection(fence).area
        min_area = min(box_polygon.area, fence.area)
        if min_area == 0:
            return 0
        return intersection_area / min_area

    
    def rle_to_mask(self) -> np.ndarray | None:
        if self.mask is None:
            return None
        # 这里可以调用之前实现的 rle_to_binary_mask 函数
        # 需要传入 mask 中的 RLE 数据以及 box 的坐标和尺寸信息
        return rle_to_binary_mask(self.mask)


    def __repr__(self):
        return f"Box(label={self.label}, score={self.score}, box={self.box})"

    def __str__(self):
        return f"Box(label={self.label}, score={self.score}, box={self.box})"
