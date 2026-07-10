### 睡岗检测
import threading
import time
from typing import List, Dict, Optional

import cv2
import numpy as np

from config.config import config
from utils.obj import Box
from utils.filter import label_filter, score_filter, area_filter
from utils.logger import setup_logger

try:
    from llm.llm import VisionAPIClient
    _vlm_import_error = None
except ImportError as e:
    VisionAPIClient = None
    _vlm_import_error = str(e)

logger = setup_logger("sleep_duty")
if _vlm_import_error:
    logger.warning(f"llm.llm 导入失败，睡岗 VLM 二次确认将不可用: {_vlm_import_error}")

logger = setup_logger("sleep_duty")


class SleepDutyDetector:
    """
    睡岗检测器。

    逻辑：
      1. 从 SAM3 返回的 Box 中过滤出 person、head、hand（需开启 return_mask）。
      2. 按 IoU 为每个 person 分配跟踪 ID，维护每个 ID 的静止积分。
      3. 通过对比相邻帧中 person / head / hand 的 mask 重叠度，判断人员是否静止。
      4. 当静止时长超过阈值时，裁剪人员区域并调用 VLM 进行二次确认。
      5. VLM 确认睡觉则返回 sleeping 违规 Box。

    说明：
      - 本检测器使用简单的 IoU 进行 ID 关联，不依赖 ByteTrack 等外部跟踪器。
      - 必须在 config/algorithms.yaml 中为算法码 32 配置 return_mask: true。
    """

    def __init__(
        self,
        person_min_score: float = 0.5,
        person_min_area: float = 1000.0,
        part_min_score: float = 0.5,
        bind_iou_threshold: float = 0.9,
        still_iom_threshold: float = 0.95,
        sleep_threshold_seconds: float = 60.0,
        cleanup_timeout_seconds: float = 600.0,
        vlm_prompt: Optional[str] = None,
        crop_padding_ratio: float = 0.3,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param part_min_score: head / hand 最低置信度
        :param bind_iou_threshold: 判定部位（head/hand）属于某 person 的最小 IoU
        :param still_iom_threshold: 判定为静止的最小 mask IoM（重叠度高于此值认为静止）
        :param sleep_threshold_seconds: 触发 VLM 确认的静止时长阈值（秒）
        :param cleanup_timeout_seconds: 长时间未出现则清理 ID 的超时时间（秒）
        :param vlm_prompt: VLM 判断 prompt，为空时使用默认睡岗 prompt
        :param crop_padding_ratio: 裁剪人员区域交给 VLM 时的 padding 比例
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
        self.part_min_score = part_min_score
        self.bind_iou_threshold = bind_iou_threshold
        self.still_iom_threshold = still_iom_threshold
        self.sleep_threshold_seconds = sleep_threshold_seconds
        self.cleanup_timeout_seconds = cleanup_timeout_seconds
        self.crop_padding_ratio = crop_padding_ratio

        self.vlm_prompt = vlm_prompt or (
            "检测图像中是否有人员在睡觉。\n"
            "判定标准：\n"
            "- 如果发现人员处于睡眠状态（如闭眼、头部下垂、躺卧等姿势），判定为睡岗，返回 yes\n"
            "- 如果人员保持正常工作状态、姿势端正、保持警觉，判定为未睡岗，返回 no\n"
        )

        # 简单的 IoU ID 关联状态
        self._trackers: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._next_id = 0

        # VLM 客户端
        self._vlm_client: Optional[VisionAPIClient] = None
        self._init_vlm_client()

    def _init_vlm_client(self):
        """初始化 VLM 客户端"""
        try:
            self._vlm_client = VisionAPIClient(
                api_key=getattr(config, "VL_API_KEY", None) or None,
                base_url=getattr(config, "VL_API_URL", None) or None,
                model=getattr(config, "VL_MODEL", "/models/qwen3-vl-4b"),
                enable_thinking=getattr(config, "VL_ENABLE_THINKING", False),
            )
            logger.info("睡岗 VLM 客户端初始化成功")
        except Exception as e:
            logger.error(f"睡岗 VLM 客户端初始化失败: {e}")
            self._vlm_client = None

    @staticmethod
    def _crop_person(frame: np.ndarray, bbox: List[float], padding_ratio: float = 0.3) -> np.ndarray:
        """裁剪出人员区域，并添加 padding，方便 VLM 聚焦判定。"""
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

        return frame[y1_padded:y2_padded, x1_padded:x2_padded]

    def _request_vlm_analysis(self, person_crop: np.ndarray, device_id: str = "") -> str:
        """调用 VLM 判断人员是否在睡觉。"""
        if self._vlm_client is None:
            logger.warning(f"[device={device_id}] VLM 客户端未初始化，跳过睡岗二次确认")
            return "error"

        try:
            start_time = time.time()
            result = self._vlm_client.analyze_image(
                image_source=person_crop,
                system_prompt="你是一名工业安全监控专家，只回答 yes 或 no。",
                user_prompt=self.vlm_prompt,
            )
            duration = time.time() - start_time
            result_lower = str(result).strip().lower()
            logger.info(f"[device={device_id}] VLM 睡岗判定耗时: {duration:.2f}s | 结果: {result_lower}")
            return result_lower
        except Exception as e:
            logger.error(f"[device={device_id}] VLM 睡岗请求异常: {e}")
            return "error"

    def _associate_id(self, person_box: Box) -> str:
        """按 IoU 为当前 person 匹配已有 ID，未匹配则分配新 ID。"""
        best_tid = None
        best_iou = 0.4
        with self._lock:
            for tid, tdata in self._trackers.items():
                iou = person_box.iou(Box("person", 1.0, tdata["last_box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is None:
                best_tid = f"sleep_{self._next_id}"
                self._next_id += 1
                self._trackers[best_tid] = {
                    "last_masks": {"person": None, "head": None, "hands": []},
                    "score": 0.0,
                    "last_box": person_box.box,
                    "last_seen": time.time(),
                    "last_process_time": time.time(),
                    "alerted": False,
                }

            return best_tid

    def _detect_stillness(
        self,
        person_box: Box,
        head_box: Optional[Box],
        hand_boxes: List[Box],
        last_state: dict,
        image_width: int,
        image_height: int,
    ) -> bool:
        """
        对比当前帧与上一帧的 mask 重叠度，判断人员是否静止。

        :return: True=静止，False=移动
        """
        is_moving = False
        last_masks = last_state["last_masks"]

        # A. 人体整体判定
        if last_masks["person"] is not None:
            iom = person_box.mask_iom(last_masks["person"], image_width, image_height)
            if iom < self.still_iom_threshold:
                is_moving = True

        # B. 头部判定
        if not is_moving:
            last_head = last_masks["head"]
            if (last_head is None) != (head_box is None):
                if last_masks["person"] is not None:
                    is_moving = True
            elif last_head is not None and head_box is not None:
                iom = head_box.mask_iom(last_head, image_width, image_height)
                if iom < self.still_iom_threshold:
                    is_moving = True

        # C. 手部逻辑判定
        if not is_moving:
            prev_hands = last_masks["hands"]
            if len(prev_hands) != len(hand_boxes):
                if len(prev_hands) > 0 or len(hand_boxes) > 0:
                    if last_masks["person"] is not None:
                        is_moving = True
            elif len(hand_boxes) > 0:
                matched_count = 0
                for c_hand in hand_boxes:
                    best_iom = 0.0
                    for p_hand in prev_hands:
                        iom = c_hand.mask_iom(p_hand, image_width, image_height)
                        best_iom = max(best_iom, iom)
                    if best_iom >= self.still_iom_threshold:
                        matched_count += 1
                if matched_count != len(hand_boxes):
                    is_moving = True

        return not is_moving

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
        睡岗检测入口。

        :param predictions: SAM3 返回的检测框，需包含 person / head / hand 且带 mask
        :param fences: 区域坐标列表，当前检测不使用，预留以保持接口统一
        :param device_id: 设备 ID，用于日志区分
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :param frame: 当前帧图像（BGR numpy 数组），VLM 二次确认需要
        :return: 违规目标 Box 列表（label=sleeping）
        """
        now = time.time()
        result = []

        if frame is None:
            logger.warning(f"[device={device_id}] SleepDutyDetector 未收到 frame，跳过检测")
            return result

        if image_width <= 0 or image_height <= 0:
            image_height, image_width = frame.shape[:2]

        # 1. 过滤 person / head / hand
        person_boxes = label_filter(predictions, ["person"])
        person_boxes = score_filter(person_boxes, self.person_min_score)
        person_boxes = area_filter(person_boxes, self.person_min_area)

        head_boxes = label_filter(predictions, ["head"])
        head_boxes = score_filter(head_boxes, self.part_min_score)

        hand_boxes = label_filter(predictions, ["hand"])
        hand_boxes = score_filter(hand_boxes, self.part_min_score)

        # 2. 清理过期 ID
        with self._lock:
            expired = [
                tid for tid, tdata in self._trackers.items()
                if now - tdata["last_seen"] > self.cleanup_timeout_seconds
            ]
            for tid in expired:
                del self._trackers[tid]

        # 3. 按 person 进行睡岗判定
        for person_box in person_boxes:
            # 关联 ID
            tid = self._associate_id(person_box)

            with self._lock:
                tdata = self._trackers[tid]
                time_delta = now - tdata["last_process_time"]

                # 找出属于该人员的 head / hands（按 IoU 绑定，mask 优先）
                matched_head = None
                for head in head_boxes:
                    iou = head.iou(person_box)
                    if iou > self.bind_iou_threshold:
                        matched_head = head
                        break

                matched_hands = []
                for hand in hand_boxes:
                    iou = hand.iou(person_box)
                    if iou > self.bind_iou_threshold:
                        matched_hands.append(hand)

                # 判定是否静止
                is_still = self._detect_stillness(
                    person_box,
                    matched_head,
                    matched_hands,
                    tdata,
                    image_width,
                    image_height,
                )

                if is_still:
                    tdata["score"] += time_delta
                    logger.debug(
                        f"[device={device_id}] ID:{tid} 静止中，累计 {tdata['score']:.2f}s"
                    )
                else:
                    if tdata["score"] > 0:
                        logger.info(
                            f"[device={device_id}] ID:{tid} 发生移动，静止积分清零 "
                            f"(原 {tdata['score']:.2f}s)"
                        )
                    tdata["score"] = 0.0
                    tdata["alerted"] = False

                # 更新状态
                tdata["last_masks"] = {
                    "person": person_box,
                    "head": matched_head,
                    "hands": matched_hands,
                }
                tdata["last_box"] = person_box.box
                tdata["last_seen"] = now
                tdata["last_process_time"] = now

                # 触发 VLM 二次确认
                if tdata["score"] >= self.sleep_threshold_seconds and not tdata["alerted"]:
                    logger.info(
                        f"[device={device_id}] ID:{tid} 静止时长 {tdata['score']:.2f}s，"
                        f"触发 VLM 睡岗二次确认"
                    )

                    person_crop = self._crop_person(frame, person_box.box, self.crop_padding_ratio)
                    vlm_result = self._request_vlm_analysis(person_crop, device_id=device_id)

                    if "yes" in vlm_result:
                        logger.warning(
                            f"[device={device_id}] 睡岗报警触发! ID:{tid} (VLM 确认)"
                        )
                        alert_box = Box(
                            label="sleeping",
                            score=min(1.0, tdata["score"] / self.sleep_threshold_seconds),
                            box=person_box.box,
                        )
                        result.append(alert_box)
                        tdata["alerted"] = True
                    else:
                        logger.info(
                            f"[device={device_id}] VLM 判定 ID:{tid} 未睡岗，忽略本次嫌疑"
                        )

                    # 无论是否报警，确认后清零积分，防止重复发包
                    tdata["score"] = 0.0

        return result
