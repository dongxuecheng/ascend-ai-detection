
from typing import List, Tuple
from utils.obj import Box


def area_filter(boxes: list, min_area: float) -> list:
    """过滤掉面积小于阈值的框"""
    return [box for box in boxes if box.area() >= min_area]


def score_filter(boxes: list, min_score: float) -> list:
    """过滤掉置信度低于阈值的框"""
    return [box for box in boxes if box.score >= min_score]


def label_filter(boxes: list, allowed_labels: list) -> list:
    """只保留指定标签的框"""
    return [box for box in boxes if box.label in allowed_labels]


def size_range_filter(boxes: list, min_area: float = 0, max_area: float = float('inf')) -> list:
    """
    按面积范围过滤框
    :param min_area: 最小面积（默认 0，不过滤）
    :param max_area: 最大面积（默认 inf，不过滤）
    """
    return [box for box in boxes if min_area <= box.area() <= max_area]


def aspect_ratio_filter(boxes: list, min_ratio: float = 0.3, max_ratio: float = 3.0) -> list:
    """
    按宽高比过滤框，过滤掉太扁或太瘦的异常框
    :param min_ratio: 最小宽高比（宽/高），默认 0.3
    :param max_ratio: 最大宽高比（宽/高），默认 3.0
    """
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box.box
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            continue
        ratio = w / h
        if min_ratio <= ratio <= max_ratio:
            result.append(box)
    return result


def edge_filter(boxes: list, img_width: int, img_height: int, margin_ratio: float = 0.05) -> list:
    """
    过滤掉紧贴图像边缘的框（通常是误识别或截断目标）
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param margin_ratio: 距离边缘的最小留白（像素）比例，默认 0.05
    """
    margin_height = int(img_height * margin_ratio)
    margin_width  = int(img_width * margin_ratio)
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box.box
        if x1 < margin_width or y1 < margin_height or x2 > (img_width - margin_width) or y2 > (img_height - margin_height):
            continue
        result.append(box)
    return result


def top_k_filter(boxes: list, k: int = 10) -> list:
    """
    只保留分数最高的前 K 个框
    :param k: 保留的最大数量
    """
    if len(boxes) <= k:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    return sorted_boxes[:k]


def nms_filter(boxes: list, iou_thresh: float = 0.5) -> list:
    """
    对 Box 列表做非极大值抑制（NMS），去除高度重叠的重复检测框
    :param iou_thresh: IoU 阈值，超过则认为重叠
    """
    if not boxes:
        return []

    # 按分数降序排列
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep = []

    while sorted_boxes:
        current = sorted_boxes.pop(0)
        keep.append(current)
        # 过滤掉与 current 重叠度高的框
        sorted_boxes = [
            b for b in sorted_boxes
            if current.iou(b) <= iou_thresh
        ]

    return keep


def parent_child_filter(
    boxes: list,
    parent_labels: List[str],
    child_labels: List[str],
    min_iom: float = 0.8
) -> list:
    """
    父子框包含过滤：过滤掉被父框完全包含的子框（通常是误识别）
    例如：person 框内部有很小的 hand 框，且 hand 几乎完全在 person 内，可能是误识别

    :param parent_labels: 父框标签列表，如 ['person']
    :param child_labels: 子框标签列表，如 ['hand', 'head']
    :param min_iom: 最小包含度（子框与父框的 IoM），默认 0.8
    :return: 过滤后的 Box 列表（保留父框，移除被包含的子框）
    """
    parents = [b for b in boxes if b.label in parent_labels]
    children = [b for b in boxes if b.label in child_labels]
    others = [b for b in boxes if b.label not in parent_labels and b.label not in child_labels]

    valid_children = []
    for child in children:
        contained = False
        for parent in parents:
            if child.iom(parent) >= min_iom:
                contained = True
                break
        if not contained:
            valid_children.append(child)

    return parents + valid_children + others


def center_zone_filter(boxes: list, zone: Tuple[int, int, int, int]) -> list:
    """
    只保留中心点落在指定区域内的框
    :param zone: (x1, y1, x2, y2) 区域坐标
    """
    zx1, zy1, zx2, zy2 = zone
    result = []
    for box in boxes:
        cx, cy = box.center
        if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
            result.append(box)
    return result
