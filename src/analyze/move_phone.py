### 移动中使用手机检测
import threading
import time
from collections import deque
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from config.config import config
from utils.obj import Box
from utils.filter import label_filter, score_filter, area_filter, nms_filter
from utils.logger import setup_logger
from obj_track.trackers import create_tracker
from detect.sam3 import call_sam3

logger = setup_logger("move_phone")


class MoveUsePhoneDetector:
    """
    移动中使用手机检测器。

    算法流程：
      1. 使用 Triton/YOLO 返回的 person 框作为输入，通过 ByteTrack/OCSort 跟踪人员。
      2. 缓存每个 track 最近一段时间的帧和 bbox。
      3. 当某人员持续移动超过阈值时，从历史缓存中抽取中间三帧。
      4. 对每帧裁剪出人员区域并调用 SAM3 检测 phone / person / hand。
      5. 若手机与手相交，且与人员上半身相交，则判定为移动中使用手机违章。

    注意：本检测器内部维护跟踪状态和历史帧缓存，需要在 config/algorithms.yaml 的
    thread_local_detectors 中配置为线程隔离，避免多路视频并发时状态串扰。
    """

    def __init__(
        self,
        person_min_score: float = 0.5,
        person_min_area: float = 1000.0,
        tracker_type: str = "bytetrack",
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 25,
        speed_check_seconds: float = 0.5,
        iou_threshold: float = 0.75,
        min_moving_duration_seconds: float = 1.0,
        min_total_iou: float = 0.3,
        cache_duration_seconds: float = 3.0,
        phone_hand_iom: float = 0.05,
        phone_upper_body_iom: float = 0.1,
        disappearance_threshold_seconds: float = 2.0,
        crop_padding_ratio: float = 0.2,
        sam3_confidence_threshold: float = 0.65,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param tracker_type: 跟踪器类型，可选 "bytetrack" / "ocsort"
        :param track_thresh: 跟踪器检测阈值
        :param track_buffer: 跟踪器丢失缓冲帧数
        :param match_thresh: 跟踪器匹配阈值
        :param frame_rate: 视频帧率，用于跟踪器时间推算
        :param speed_check_seconds: 移动判定窗口（秒），低于此窗口 IoU 阈值认为在移动
        :param iou_threshold: 移动判定 IoM 阈值，低于此值认为在移动
        :param min_moving_duration_seconds: 持续移动多少秒后触发手机判定
        :param min_total_iou: 缓存起终点 IoM 阈值，高于此值认为位移过小，过滤原地摇晃
        :param cache_duration_seconds: 历史帧缓存时长（秒）
        :param phone_hand_iom: 手机与 hand 的最小 IoM
        :param phone_upper_body_iom: 手机与人员上半身的最小 IoM
        :param disappearance_threshold_seconds: track 消失多少秒后清理状态
        :param crop_padding_ratio: 裁剪人员区域时的 padding 比例
        :param sam3_confidence_threshold: SAM3 手机检测置信度阈值
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area

        self.speed_check_seconds = speed_check_seconds
        self.iou_threshold = iou_threshold
        self.min_moving_duration_seconds = min_moving_duration_seconds
        self.min_total_iou = min_total_iou
        self.cache_duration_seconds = cache_duration_seconds
        self.phone_hand_iom = phone_hand_iom
        self.phone_upper_body_iom = phone_upper_body_iom
        self.disappearance_threshold_seconds = disappearance_threshold_seconds
        self.crop_padding_ratio = crop_padding_ratio
        self.sam3_confidence_threshold = sam3_confidence_threshold

        # 人员跟踪器（每个线程一个实例）
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

        # track_id -> 状态
        self._states: Dict[int, dict] = {}
        self._lock = threading.Lock()

        # SAM3 接口地址
        self.sam3_url = config.SAM3_URL

    @staticmethod
    def _tlwh_to_xyxy(tlwh) -> List[float]:
        """将 [top_left_x, top_left_y, width, height] 转为 [x1, y1, x2, y2]"""
        x, y, w, h = tlwh
        return [float(x), float(y), float(x + w), float(y + h)]

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox: List[float], padding_ratio: float = 0.2) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        裁剪出人员区域，并添加 padding。

        :param frame: 原始帧
        :param bbox: 边界框 [x1, y1, x2, y2]
        :param padding_ratio: padding 比例（相对于边界框宽高）
        :return: (裁剪后的图像, 裁剪区域左上角坐标 (crop_x1, crop_y1))
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        pad_x = int(bbox_width * padding_ratio)
        pad_y = int(bbox_height * padding_ratio)

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(w, x2 + pad_x)
        y2_padded = min(h, y2 + pad_y)

        cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        return cropped, (x1_padded, y1_padded)

    def _check_phone_in_use(self, phone_boxes: List[Box], hand_boxes: List[Box], person_upper_body: Box) -> bool:
        """
        检查手机是否与手相交，且与人员上半身相交。

        :param phone_boxes: 手机框列表（裁剪图坐标）
        :param hand_boxes: 手框列表（裁剪图坐标）
        :param person_upper_body: 人员上半身框（裁剪图坐标）
        :return: 是否存在符合要求的手机
        """
        if not phone_boxes:
            return False

        for phone in phone_boxes:
            # 手机必须与人员上半身相交
            if phone.iom(person_upper_body) <= self.phone_upper_body_iom:
                continue

            # 手机必须与手相交
            for hand in hand_boxes:
                if hand.iom(phone) > self.phone_hand_iom:
                    return True

        return False

    def _check_phone_in_frames(
        self,
        frames_with_bbox: List[Tuple[np.ndarray, Tuple[int, int], List[float]]],
        device_id: str = "",
    ) -> bool:
        """
        顺序检查三帧中是否有手机，验证手机是否与人员上半身及手相交。

        :param frames_with_bbox: [(cropped_frame, crop_offset, original_bbox), ...]
        :return: 是否检测到使用手机
        """
        for idx, (person_crop, crop_offset, bbox) in enumerate(frames_with_bbox):
            if person_crop.size == 0:
                continue

            # 计算人员上半身（裁剪图坐标）
            x1, y1, x2, y2 = bbox
            crop_x, crop_y = crop_offset
            person_height = y2 - y1
            person_y2_upper = y1 + int(person_height * 2 / 5)
            person_upper_body = Box(
                label="person",
                score=1.0,
                box=[
                    x1 - crop_x,
                    y1 - crop_y,
                    x2 - crop_x,
                    person_y2_upper - crop_y,
                ],
            )

            sam3_boxes = call_sam3(
                person_crop,
                prompts=["phone", "person", "hand"],
                confidence_threshold=self.sam3_confidence_threshold,
                return_mask=False,
                url=self.sam3_url,
            )
            phone_boxes = [b for b in sam3_boxes if b.label in ("phone", "mobile phone", "cell phone")]
            if not phone_boxes:
                continue

            hand_boxes = [b for b in sam3_boxes if b.label == "hand"]

            if self._check_phone_in_use(phone_boxes, hand_boxes, person_upper_body):
                logger.info(f"[device={device_id}] 第 {idx + 1} 帧历史帧判定为移动中使用手机")
                return True

        return False

    def _sample_frames(
        self,
        frame_cache: deque,
    ) -> List[Tuple[np.ndarray, Tuple[int, int], List[float]]]:
        """从历史帧缓存中抽取中间三帧（已裁剪）。"""
        cache_len = len(frame_cache)
        if cache_len == 0:
            return []

        if cache_len < 3:
            # 缓存不足 3 帧时取全部
            return [(item[1], item[2], item[3]) for item in frame_cache]

        mid_idx = cache_len // 2
        gap = max(1, cache_len // 15)
        indices = [
            max(0, mid_idx - gap),
            mid_idx,
            min(cache_len - 1, mid_idx + gap),
        ]
        # 去重并保持顺序
        seen = set()
        result = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                item = frame_cache[idx]
                result.append((item[1], item[2], item[3]))
        return result

    def detect(
        self,
        predictions: List[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
        frame: Optional[np.ndarray] = None,
    ) -> List[Box]:
        """
        移动中使用手机检测入口。

        :param predictions: 当前帧的所有检测框，需包含 person
        :param fences: 区域坐标列表，当前检测不使用，预留以保持接口统一
        :param device_id: 设备 ID，用于日志区分
        :param image_width: 原图宽度（用于跟踪器）
        :param image_height: 原图高度（用于跟踪器）
        :param frame: 当前帧图像（BGR numpy 数组），必须传入，否则无法缓存历史帧
        :return: 违规目标 Box 列表（label=moving-use-phone）
        """
        now = time.time()

        if frame is None:
            logger.warning(
                f"[device={device_id}] MoveUsePhoneDetector 未收到 frame，无法缓存历史帧，跳过检测"
            )
            return []

        # 1. 过滤 person
        person_boxes = label_filter(predictions, ["person"])
        person_boxes = score_filter(person_boxes, self.person_min_score)
        person_boxes = area_filter(person_boxes, self.person_min_area)
        person_boxes = nms_filter(person_boxes, 0.5)

        # 2. 用 tracker 跟踪人员
        if person_boxes:
            dets = np.array(
                [[b.box[0], b.box[1], b.box[2], b.box[3], b.score, 0] for b in person_boxes],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        frame_shape = (image_height, image_width, 3) if image_width and image_height else None
        tracks = self.tracker.update(dets, frame_shape=frame_shape)

        # 3. 更新跟踪状态并判定违章
        result = []
        current_track_ids = set()

        with self._lock:
            for track in tracks:
                tid = track.track_id
                current_track_ids.add(tid)

                track_xyxy = self._tlwh_to_xyxy(track.tlwh)

                if tid not in self._states:
                    self._states[tid] = {
                        "bbox_history": deque(),  # (timestamp, bbox)
                        "frame_cache": deque(),   # (timestamp, cropped_frame, crop_offset, bbox)
                        "last_time": now,
                        "is_moving": False,
                        "alerted": False,
                        "moving_start_time": None,
                    }

                state = self._states[tid]
                state["last_time"] = now

                # 缓存当前 bbox 用于移动判定
                state["bbox_history"].append((now, track_xyxy))
                while state["bbox_history"] and state["bbox_history"][0][0] < now - self.speed_check_seconds:
                    state["bbox_history"].popleft()

                # 缓存当前帧的人员裁剪图，减少内存占用
                if frame is not None:
                    cropped, crop_offset = self._crop_person(frame, track_xyxy, self.crop_padding_ratio)
                    state["frame_cache"].append((now, cropped, crop_offset, track_xyxy))
                while state["frame_cache"] and state["frame_cache"][0][0] < now - self.cache_duration_seconds:
                    state["frame_cache"].popleft()

                # 移动判定：历史框与当前框的 IoM 低于阈值认为在移动
                history = state["bbox_history"]
                if len(history) >= 2:
                    first_time, first_bbox = history[0]
                    current_box = Box("person", 1.0, track_xyxy)
                    first_box = Box("person", 1.0, first_bbox)
                    iom = current_box.iom(first_box)
                    if iom < self.iou_threshold:
                        if not state["is_moving"]:
                            state["is_moving"] = True
                            state["moving_start_time"] = now
                    else:
                        if state["is_moving"]:
                            logger.info(
                                f"[device={device_id}] track {tid} 停止移动"
                            )
                        state["is_moving"] = False
                        state["alerted"] = False
                        state["moving_start_time"] = None
                else:
                    state["is_moving"] = False
                    state["moving_start_time"] = None

                # 判定：持续移动超过阈值，且未报警
                if state["is_moving"] and not state["alerted"]:
                    moving_start = state["moving_start_time"] or now
                    moving_duration = now - moving_start
                    if moving_duration >= self.min_moving_duration_seconds:
                        # 额外检查：缓存起终点位移，过滤原地小幅摇晃
                        if len(state["frame_cache"]) >= 2:
                            first_frame_bbox = state["frame_cache"][0][3]
                            last_frame_bbox = state["frame_cache"][-1][3]
                            first_box = Box("person", 1.0, first_frame_bbox)
                            last_box = Box("person", 1.0, last_frame_bbox)
                            total_iom = last_box.iom(first_box)
                            if total_iom > self.min_total_iou:
                                logger.debug(
                                    f"[device={device_id}] track {tid} 总位移过小 "
                                    f"(iom={total_iom:.3f})，跳过"
                                )
                                continue

                        sampled = self._sample_frames(state["frame_cache"])
                        if sampled and self._check_phone_in_frames(sampled, device_id=device_id):
                            logger.info(
                                f"[device={device_id}] track {tid} 移动中使用手机，触发报警"
                            )
                            alert_box = Box(
                                label="moving-use-phone",
                                score=1.0,
                                box=track_xyxy,
                            )
                            result.append(alert_box)
                            state["alerted"] = True

            # 清理过期跟踪
            expired = [
                tid
                for tid, state in self._states.items()
                if now - state["last_time"] > self.disappearance_threshold_seconds
            ]
            for tid in expired:
                del self._states[tid]

        return result
