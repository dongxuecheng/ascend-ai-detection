"""
OSD (On-Screen Display) 绘图工具

在视频帧上绘制检测框、标签、围栏、告警信息等，用于生成上传平台的报警截图。
标签直接固定在目标框右上角，不启用复杂布局计算。
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional

from utils.obj import Box

FONT = cv2.FONT_HERSHEY_SIMPLEX
VIOLATION_COLOR = (0, 0, 255)   # 红色：仅用于违规框/文字
FENCE_COLOR     = (0, 165, 255) # 橙色：围栏线

# 固定字号与 padding
FONT_SIZE = 16
PADDING_X = 4
PADDING_Y = 2


# ---------- 动态颜色生成（排除红色系） ----------
def _generate_color(label: str) -> tuple:
    """根据 label 哈希动态生成 BGR 颜色，避开红色（留给违规框）"""
    seed = hash(label) & 0xFFFFFFFF
    # HSV 中 H 范围 20-340，避开 0-15 和 345-360 的红色系
    hue = 20 + (seed % 320)
    sat = 0.7 + (seed % 20) / 100.0   # 0.70-0.89
    val = 0.8 + (seed % 15) / 100.0   # 0.80-0.94
    hsv = np.float32([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(x * 255) for x in bgr)


# ---------- 绘制单个标签（右上角固定位置） ----------
def _draw_label_top_right(
    canvas: np.ndarray,
    x2: int, y1: int,
    text: str,
    color: tuple,
) -> None:
    """
    在目标框右上角绘制标签背景 + 文字。
    左上角坐标固定为 (x2, y1)。
    """
    font_scale = FONT_SIZE / 22.0
    thickness = max(1, int(FONT_SIZE / 16))
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)

    label_w = text_w + PADDING_X * 2
    label_h = text_h + baseline + PADDING_Y * 2
    label_x = x2
    label_y = y1

    # 若超出右边界，向左对齐
    img_h, img_w = canvas.shape[:2]
    if label_x + label_w > img_w:
        label_x = max(0, img_w - label_w)
    # 若超出上边界，画在框内顶部
    if label_y < 0:
        label_y = 0

    # 绘制背景
    '''
    cv2.rectangle(
        canvas,
        (label_x, label_y),
        (label_x + label_w, label_y + label_h),
        color,
        -1,
    )
    '''

    # 绘制文字（白色）
    text_x = label_x + PADDING_X
    text_y = label_y + PADDING_Y + text_h
    cv2.putText(
        canvas, text, (text_x, text_y),
        FONT, font_scale, (255, 255, 255),
        thickness, cv2.LINE_AA,
    )


# ---------- 围栏绘制 ----------
def draw_fences(frame: np.ndarray, fences: List, color: tuple = FENCE_COLOR,
                thickness: int = 2) -> np.ndarray:
    if not fences:
        return frame
    for fence in fences:
        try:
            pts = np.array(fence, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        except Exception:
            continue
    return frame


# ---------- 主入口：生成报警截图 ----------
def render_alert_frame(
    frame: np.ndarray,
    all_boxes: List[Box],
    violations: List[Box],
    task,
    fences: List = None,
    draw_all_boxes: bool = True,
    draw_violation_boxes: bool = True,
) -> np.ndarray:
    """一站式生成报警截图。每个标签动态分配颜色（非红），文字与边框同色，白色描边。"""
    canvas = frame.copy()

    # 1. 围栏（最底层）
    if fences:
        draw_fences(canvas, fences)

    # 2. 收集所有框和标签信息
    violation_ids = {id(v) for v in violations}

    # 普通检测框
    if draw_all_boxes and all_boxes:
        for b in all_boxes:
            if id(b) not in violation_ids:
                x1, y1, x2, y2 = map(int, b.box)
                color = _generate_color(b.label)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                _draw_label_top_right(
                    canvas, x2, y1,
                    f"{b.label} {b.score:.2f}",
                    color,
                )

    # 违规框（红色），如有 mask 则先叠加半透明红色高亮
    if draw_violation_boxes and violations:
        img_h, img_w = canvas.shape[:2]
        for b in violations:
            '''
            try:
                if b.label == "conveyor belt":
                    mask = b.compute_mask_array(img_w, img_h)
                    if mask is not None and mask.any():
                        overlay = canvas.copy()
                        overlay[mask > 0] = VIOLATION_COLOR
                        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
            except Exception:
                pass
            '''
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), VIOLATION_COLOR, 3)
            # _draw_label_top_right(
            #     canvas, x2, y1,
            #     f"{b.label}",
            #     VIOLATION_COLOR,
            # )

    return canvas
