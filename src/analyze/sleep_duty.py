### 睡岗检测
import time
from typing import List, Dict, Optional

import numpy as np

from utils.obj import Box
from utils.filter import label_filter, score_filter, area_filter, nms_filter
from utils.logger import setup_logger

logger = setup_logger("sleep_duty")


class SleepDutyDetector:
    """
    睡岗检测器。

    逻辑：
      1. 从 SAM3 返回的 Box 中过滤出 person、head、hand、face（需开启 return_mask）。
      2. 按 mask 重叠度（或 box IoU）为每个 person 分配跟踪 ID，维护每个 ID 的“连续静止次数”。
      3. 每次分析时对比当前帧与上一次观测的 person / head / hand 的 mask 重叠度：
         静止则次数 +1，出现移动（或看到脸）则次数清零。
      4. 连续静止次数达到 sleep_threshold_frames 时，输出 sleeping 违规框，
         由上层 VL 模块统一进行二次确认。

    说明：
      - 按“次数”计数，与检测频率无关：检测间隔 3s 时 50 次 ≈ 150s，间隔 10s 时 ≈ 500s，
        想看实际时长就调检测间隔或阈值次数。
      - 本检测器使用简单的 mask/IoU 进行 ID 关联，不依赖 ByteTrack 等外部跟踪器。
      - person 框先做 NMS 去重（person_nms_iou），避免同一人被重复检测后重复跟踪。
      - 同一帧内一个 ID 只分配给一个 person，避免两人共用一份计数互相串扰。
      - 漏检保护：连续 max_missing_frames 次没检测到该 ID，视为人员离开画面，计数清零。
      - 必须在 config/algorithms.yaml 中为算法码 32 配置 return_mask: true。
      - VL 二次确认统一走 llm/vl_analyzer.py，本检测器不再内部调用大模型，避免重复请求。
    """

    def __init__(
        self,
        person_min_score: float = 0.8,
        person_min_area: float = 5000.0,
        part_min_score: float = 0.65,
        face_min_score: float = 0.8,
        person_nms_iou: float = 0.6,
        bind_iou_threshold: float = 0.7,
        bind_mask_iom_threshold: float = 0.7,
        still_iom_threshold: float = 0.95,
        sleep_threshold_frames: int = 50,
        max_missing_frames: int = 5,
        cleanup_timeout_seconds: float = 1000.0,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param part_min_score: head / hand 最低置信度
        :param face_min_score: face 最低置信度
        :param person_nms_iou: person 框 NMS 去重的 IoU 阈值
        :param bind_iou_threshold: 用 box IoU 关联 ID 时的最小阈值（无 mask 时使用）
        :param bind_mask_iom_threshold: 用 mask IoM 绑定部位或关联 ID 时的最小阈值
        :param still_iom_threshold: 判定为静止的最小 mask IoM（重叠度高于此值认为静止）
        :param sleep_threshold_frames: 触发 sleeping 违规的连续静止次数阈值（次）
        :param max_missing_frames: 连续多少次没检测到该 ID 就视为人员离开画面，计数清零（次）
        :param cleanup_timeout_seconds: 长时间未出现则清理 ID 的超时时间（秒）
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
        self.part_min_score = part_min_score
        self.face_min_score = face_min_score
        self.person_nms_iou = person_nms_iou
        self.bind_iou_threshold = bind_iou_threshold
        self.bind_mask_iom_threshold = bind_mask_iom_threshold
        self.still_iom_threshold = still_iom_threshold
        self.sleep_threshold_frames = max(1, int(sleep_threshold_frames))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self.cleanup_timeout_seconds = cleanup_timeout_seconds

        # 简单的 ID 关联状态
        self._trackers: Dict[str, dict] = {}
        self._next_id = 0

    def _overlap_score(
        self,
        box_a: Box,
        box_b: Box,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        计算两个 Box 的重叠度。优先使用 mask IoM，任一没有 mask 时回退到 box IoU。
        """
        # 只要两者都有 mask（RLE 或 ndarray），就用 mask 计算
        if box_a.mask is not None and box_b.mask is not None:
            return box_a.mask_iom(box_b, image_width, image_height)
        # 其中一方有 mask 另一方没有时，mask_iom 会返回 0，不如 IoU 稳健
        return box_a.iou(box_b)

    def _associate_id(
        self,
        person_box: Box,
        image_width: int,
        image_height: int,
        used_ids: set,
    ) -> str:
        """
        按 mask 重叠度（或 box IoU）为当前 person 匹配已有 ID，未匹配则分配新 ID。

        :param used_ids: 本帧已被占用的 ID 集合；同一个 ID 在同一帧不允许重复分配，
                         否则多个人会共用一份静止积分、last_masks 互相覆盖导致状态串扰
        """
        best_tid = None
        best_score = 0.0

        for tid, tdata in self._trackers.items():
            if tid in used_ids:
                continue

            # 使用完整的上一帧 Box（含 mask），保证 mask IoM 能真正生效
            last_box = tdata.get("last_person_box")
            if last_box is None:
                continue

            score = self._overlap_score(person_box, last_box, image_width, image_height)
            # 根据实际使用的度量选择阈值，避免拿 box IoU 去比 mask IoM 的阈值
            has_mask = person_box.mask is not None and last_box.mask is not None
            threshold = self.bind_mask_iom_threshold if has_mask else self.bind_iou_threshold
            if score >= threshold and score > best_score:
                best_score = score
                best_tid = tid

        if best_tid is None:
            best_tid = f"sleep_{self._next_id}"
            self._next_id += 1
            self._trackers[best_tid] = {
                "last_masks": {"person": None, "head": None, "hands": []},
                "last_person_box": None,
                "still_count": 0,     # 连续静止次数
                "miss": 0,            # 连续漏检次数
                "last_seen": time.time(),
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
        没有 mask 时回退到 box IoM。

        :return: True=静止，False=移动
        """
        is_moving = False
        last_masks = last_state["last_masks"]
        last_person = last_masks["person"]

        # A. 人体整体判定
        if last_person is not None:
            if person_box.mask is not None and last_person.mask is not None:
                score = person_box.mask_iom(last_person, image_width, image_height)
            else:
                score = person_box.iom(last_person)
            if score < self.still_iom_threshold:
                is_moving = True
                logger.debug(f"睡岗 ID 移动触发：person 重叠度 {score:.3f} < {self.still_iom_threshold}")

        # B. 头部判定
        if not is_moving:
            last_head = last_masks["head"]
            if (last_head is None) != (head_box is None):
                # 头部出现或消失，认为有变化
                if last_person is not None:
                    is_moving = True
            elif last_head is not None and head_box is not None:
                if head_box.mask is not None and last_head.mask is not None:
                    score = head_box.mask_iom(last_head, image_width, image_height)
                else:
                    score = head_box.iom(last_head)
                if score < self.still_iom_threshold:
                    is_moving = True
                    logger.debug(f"睡岗 ID 移动触发：head 重叠度 {score:.3f} < {self.still_iom_threshold}")

        # C. 手部逻辑判定
        if not is_moving:
            prev_hands = last_masks["hands"]
            if len(prev_hands) != len(hand_boxes):
                if len(prev_hands) > 0 or len(hand_boxes) > 0:
                    if last_person is not None:
                        is_moving = True
            elif len(hand_boxes) > 0:
                matched_count = 0
                for c_hand in hand_boxes:
                    best_score = 0.0
                    for p_hand in prev_hands:
                        if c_hand.mask is not None and p_hand.mask is not None:
                            score = c_hand.mask_iom(p_hand, image_width, image_height)
                        else:
                            score = c_hand.iom(p_hand)
                        best_score = max(best_score, score)
                    if best_score >= self.still_iom_threshold:
                        matched_count += 1
                    else:
                        logger.debug(f"睡岗 hand 移动：best_iom={best_score:.3f}")
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

        :param predictions: SAM3 返回的检测框，需包含 person / head / hand / face 且带 mask
        :param fences: 区域坐标列表，当前检测不使用，预留以保持接口统一
        :param device_id: 设备 ID，用于日志区分
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :param frame: 当前帧图像（BGR numpy 数组），本检测器不再内部使用，仅保持接口兼容
        :return: 违规目标 Box 列表（label=sleeping）
        """
        now = time.time()
        result = []

        if image_width <= 0 or image_height <= 0:
            if frame is not None:
                image_height, image_width = frame.shape[:2]
            else:
                logger.warning(f"[device={device_id}] SleepDutyDetector 未获取到图像宽高，跳过检测")
                return result

        # 1. 过滤 person / head / hand / face
        person_boxes = label_filter(predictions, ["person"])
        person_boxes = [p for p in person_boxes if p.source == "SAM3"]
        person_boxes = score_filter(person_boxes, self.person_min_score)
        person_boxes = area_filter(person_boxes, self.person_min_area)
        # 同一人被重复检测成多个重叠框时会导致重复跟踪，先做 NMS 去重
        person_boxes = nms_filter(person_boxes, self.person_nms_iou)

        head_boxes = label_filter(predictions, ["head"])
        head_boxes = score_filter(head_boxes, self.part_min_score)

        hand_boxes = label_filter(predictions, ["hand"])
        hand_boxes = score_filter(hand_boxes, self.part_min_score)

        face_boxes = label_filter(predictions, ["face"])
        face_boxes = score_filter(face_boxes, self.face_min_score)

        # 2. 清理过期 ID
        expired = [
            tid for tid, tdata in self._trackers.items()
            if now - tdata["last_seen"] > self.cleanup_timeout_seconds
        ]
        for tid in expired:
            del self._trackers[tid]

        # 3. 按 person 进行睡岗判定
        used_ids = set()   # 本帧已分配的 ID，保证一帧内一个 ID 只对应一个 person
        for person_box in person_boxes:
            # 关联 ID
            tid = self._associate_id(person_box, image_width, image_height, used_ids)
            used_ids.add(tid)
            tdata = self._trackers[tid]
            tdata["miss"] = 0   # 本帧观测到了

            # 找出属于该人员的 head / hands（优先使用 mask IoM，无 mask 时回退 IoU）
            matched_head = None
            for head in head_boxes:
                score = self._overlap_score(head, person_box, image_width, image_height)
                if score > self.bind_mask_iom_threshold:
                    matched_head = head
                    break

            matched_hands = []
            for hand in hand_boxes:
                score = self._overlap_score(hand, person_box, image_width, image_height)
                if score > self.bind_mask_iom_threshold:
                    matched_hands.append(hand)

            # 脸部绑定：只要能看到脸部，就认为不是睡岗
            matched_face = None
            for face in face_boxes:
                score = self._overlap_score(face, person_box, image_width, image_height)
                if score > self.bind_mask_iom_threshold:
                    matched_face = face
                    break

            # 判定本帧是否静止并更新连续静止次数
            if tdata["last_person_box"] is None:
                # 新建 ID 的第一次观测没有比较基准，本帧不计数
                logger.debug(f"[device={device_id}] ID:{tid} 首次观测，建立比较基准")
                is_still = False
                reason = "首次观测"
            elif matched_face is not None:
                # 看到脸：认为人员清醒
                is_still = False
                reason = "检测到可见脸部"
            else:
                is_still = self._detect_stillness(
                    person_box,
                    matched_head,
                    matched_hands,
                    tdata,
                    image_width,
                    image_height,
                )
                reason = "与上一帧重叠度不足"

            if is_still:
                tdata["still_count"] += 1
                if tdata["still_count"] % 10 == 0:
                    logger.info(
                        f"[device={device_id}] ID:{tid} 连续静止 "
                        f"{tdata['still_count']} 次 / 阈值 {self.sleep_threshold_frames} 次"
                    )
                else:
                    logger.debug(
                        f"[device={device_id}] ID:{tid} 静止中，第 {tdata['still_count']} 次"
                    )
            else:
                if tdata["still_count"] > 0:
                    logger.info(
                        f"[device={device_id}] ID:{tid} 连续静止被打断（{reason}），"
                        f"计数清零 (原 {tdata['still_count']} 次)"
                    )
                tdata["still_count"] = 0
                tdata["alerted"] = False

            # 更新状态
            tdata["last_masks"] = {
                "person": person_box,
                "head": matched_head,
                "hands": matched_hands,
            }
            tdata["last_person_box"] = person_box
            tdata["last_seen"] = now

            # 触发 sleeping 违规输出（VL 二次确认统一由 vl_analyzer 处理）
            if tdata["still_count"] >= self.sleep_threshold_frames and not tdata["alerted"]:
                logger.info(
                    f"[device={device_id}] ID:{tid} 连续静止 {tdata['still_count']} 次，"
                    f"输出睡岗嫌疑目标，等待 VL 复核"
                )
                alert_box = Box(
                    label="sleeping",
                    score=min(1.0, tdata["still_count"] / self.sleep_threshold_frames),
                    box=person_box.box,
                )
                result.append(alert_box)
                tdata["alerted"] = True
                # 输出一次后清零，防止同一静止周期内连续重复输出；
                # 若人员继续静止，后续会重新计数并再次触发
                tdata["still_count"] = 0

        # 4. 本帧没被观测到的 ID：累计漏检次数，连续多次没看到就认为人已离开画面，计数清零
        for tid, tdata in self._trackers.items():
            if tid in used_ids:
                continue
            tdata["miss"] = tdata.get("miss", 0) + 1
            if tdata["miss"] > self.max_missing_frames and tdata["still_count"] > 0:
                logger.info(
                    f"[device={device_id}] ID:{tid} 连续 {tdata['miss']} 次未检测到，"
                    f"视为人员离开画面，静止计数清零 (原 {tdata['still_count']} 次)"
                )
                tdata["still_count"] = 0
                tdata["alerted"] = False

        return result

