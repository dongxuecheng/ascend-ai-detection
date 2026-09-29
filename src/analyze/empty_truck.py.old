# 空车检查
from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("empty_truck")


class EmptyTruckDetector:
    """
    空车检查检测器：检测 truck bed 内是否存在煤块残留（dark coal residue）。
    若煤块与车厢的 IoF（Intersection over Foreground）超过阈值，认为车厢非空。
    """

    def __init__(self, iof_threshold: float = 0.99, min_score: float = 0.5):
        """
        :param iof_threshold: 煤块在车厢内的面积占比阈值，超过此值认为该煤块属于该车厢
        :param min_score: 目标最低置信度
        """
        self.iof_threshold = iof_threshold
        self.min_score = min_score

    @staticmethod
    def _intersection_area(box1: list, box2: list) -> float:
        """计算两个框的交集面积"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        空车检查入口。

        逻辑：
        1. 过滤 truck bed 和 dark coal residue 目标。
        2. 对每个煤块，计算其与车厢的交集面积。
        3. 计算 IoF = 交集面积 / 煤块面积。
        4. IoF 超过阈值时，认为车厢非空，返回该煤块 Box 作为违规目标。

        :param predictions: Box 对象列表
        :param fences: 区域坐标列表，当前空车检查不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度（预留）
        :param image_height: 原图高度（预留）
        :return: 违规目标 Box 列表，label 为 "dark coal residue"
        """
        truck_beds = label_filter(predictions, ["truck bed"])
        truck_beds = score_filter(truck_beds, self.min_score)

        coals = label_filter(predictions, ["dark coal residue"])
        coals = score_filter(coals, self.min_score)

        if not truck_beds or not coals:
            return []

        result = []
        used_coals = set()

        image_area = image_width * image_height

        # 遍历每个车厢和每个煤块的组合
        for truck in truck_beds:
            truck_area = (truck.box[2] - truck.box[0]) * (truck.box[3] - truck.box[1])
            if truck_area / image_area < 0.8:
                logger.warning(
                    f"[device={device_id}] 车厢面积过小，可能为误检，跳过: {truck.box}"
                )
                continue
            t_box = truck.box

            for coal in coals:
                cid = id(coal)
                if cid in used_coals:
                    continue

                c_box = coal.box
                coal_area = (c_box[2] - c_box[0]) * (c_box[3] - c_box[1])
                if coal_area <= 0:
                    continue

                inter_area = self._intersection_area(t_box, c_box)
                iof = inter_area / coal_area

                if iof > self.iof_threshold:
                    violation = Box(
                        label="dark coal residue",
                        score=coal.score,
                        box=list(coal.box),
                        mask=coal.mask,
                    )
                    logger.warning(
                        f"[device={device_id}] 空车检查违规触发 | 车厢内发现煤块 | "
                        f"iof={iof:.2%}, truck={t_box}, coal={c_box}"
                    )
                    result.append(violation)
                    used_coals.add(cid)
        # 返回最大的那一块
        result = sorted(result, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]), reverse=True)
        result = result[:1]  # 只保留最大的那一块
        return result
