import os
import threading
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
from stream.remote_capture import RTSPClient, STATUS_CONNECTED, STATUS_DISCONNECTED, STATUS_NOT_FOUND
from stream.remote_capture import DECODER_CPU_FFMPEG, DECODER_GPU_NVCUVID
from task.upload import EventUploader
from task.getFence import TaskFence
from detect.sam3 import call_sam3
from detect.triton_client_fast import YOLOTritonFast

# 全局单例：电子围栏管理器
task_fence = TaskFence()
from core.analyzer import analyze_for_task, merge_prompts_for_tasks
from core.classifier import classify_for_task
from llm.vl_analyzer import vl_analyze_for_task
from utils.osd import render_alert_frame
from utils.alert_dedup import AlertDedup

logger = setup_logger("stream_worker")

# 全局异步上传器，所有 StreamWorker 共享
uploader = EventUploader()

# 全局报警去重器，所有 StreamWorker 共享
alert_dedup = AlertDedup(
    cooldown_seconds=30.0,
    iou_thresh=0.5,
    per_label=True,
)

# 线程本地 YOLO 客户端缓存：tritonclient.http 底层使用 gevent，
# greenlet 不能跨 OS 线程切换，因此每个线程必须持有独立的客户端实例
_yolo_local = threading.local()


def _get_yolo_client(model_name: str) -> Optional[YOLOTritonFast]:
    """获取或创建 YOLO Triton 客户端（线程本地单例缓存）"""
    if not hasattr(_yolo_local, "clients"):
        _yolo_local.clients = {}

    if model_name not in _yolo_local.clients:
        try:
            model_cfg = config.YOLO_MODEL_CONFIGS.get(model_name, {})
            client = YOLOTritonFast(
                url=config.TRITON_YOLO_URL,
                model_name=model_name,
                label_map=model_cfg.get("label_map"),
                input_name=model_cfg.get("input_name", "images"),
                output_name=model_cfg.get("output_name", "output0"),
                input_size=model_cfg.get("input_size", 640),
                output_format=model_cfg.get("output_format", "yolo_v8_v11"),
                conf_thresh=model_cfg.get("conf_thresh", 0.5),
                iou_thresh=model_cfg.get("iou_thresh", 0.45),
                warmup=True
            )
            _yolo_local.clients[model_name] = client
            logger.info(f"YOLO 客户端创建成功: {model_name} @ {config.TRITON_YOLO_URL} "
                        f"(classes={client.num_classes})")
        except Exception as e:
            logger.error(f"YOLO 客户端创建失败 {model_name}: {e}")
            return None
    return _yolo_local.clients.get(model_name)


class StreamWorker(threading.Thread):
    """
    每个唯一 RTSP URL 对应一个 StreamWorker 线程。
    负责：拉流 → 一次 SAM3 推理 → 分发给关联的多个 Task 各自分析 → 各自上报
    """

    def __init__(self, rtsp_url: str, tasks: List, gpu_code: int = 0):
        super().__init__(daemon=False)
        self.rtsp_url = rtsp_url
        self.tasks = tasks  # 该流关联的所有 Task
        self.stopped = False
        self.capture: Optional[RTSPClient] = None
        self._stream_id: Optional[str] = None
        self.lock = threading.Lock()
        self.gpu_code = gpu_code

        # gRPC 服务端地址与启动参数（用于故障恢复时重建流）
        self._server_address = getattr(config, "STREAM_SERVER_ADDRESS", "192.168.100.74:50051")

        # 根据该流关联的任务算法码，判断是否需要拉取全部帧
        # 只要任一任务属于 full_frame_algorithms，就不能只拉关键帧
        full_frame_codes = set(getattr(config, "FULL_FRAME_ALGORITHMS", []))
        need_full_frame = any(
            str(getattr(task, "algorithmCode", "")) in full_frame_codes
            for task in tasks
        )
        if need_full_frame:
            logger.info(f"[StreamWorker] 流 {rtsp_url} 关联了全帧算法，only_key_frames 设为 False")

        self._stream_start_params = {
            "heartbeat_timeout_ms": 1000000,
            # "decode_interval_ms": 70,
            "use_shared_mem": True,
            "only_key_frames": not need_full_frame,
            "decoder_type": DECODER_CPU_FFMPEG,
            # "decoder_type": DECODER_GPU_NVCUVID,
            # "gpu_id": self.gpu_code,
        }

        # 读帧故障恢复状态
        self._consecutive_read_failures = 0      # 连续读帧失败次数（仅用于日志/诊断）
        self._last_frame_at = 0.0                # 上次成功读到帧的 wall-clock 时间
        self._last_recovery_at = 0.0             # 上次执行恢复的时间
        self._recovery_attempts = 0              # 本轮连续恢复失败次数
        self._max_recovery_attempts = 5          # 超过此次数后退出 worker，由 orchestrator 重建
        self._recovery_cooldown_sec = getattr(config, "STREAM_RECOVERY_COOLDOWN_SEC", 10.0)
        self._shm_gone_immediate_recover = True  # 共享内存消失时立即恢复

        # 每个任务的上次分析时间戳（用于 sleepJudgeTime 控制频率）
        self.task_last_run: Dict[str, float] = {}
        self._last_frame_ts = 0  # 用于计算帧间隔
        for task in tasks:
            self.task_last_run[task.id] = 0.0

    def _get_fences_for_task(self, task):
        """获取任务对应的电子围栏数据"""
        algo_code = str(task.algorithmCode)
        # 只在配置了 FENCE_ALGORITHMS 的算法才走 TaskFence
        if algo_code in config.FENCE_ALGORITHMS and task.deviceAlgorithmIp and task.deviceChannel:
            raw_fences = task_fence.get_fence(algo_code, task.deviceAlgorithmIp, task.deviceChannel)
            if raw_fences:
                try:
                    return [f.tolist() for f in raw_fences]
                except Exception as e:
                    logger.warning(f"围栏格式转换失败: {e}")
        return None

    def update_tasks(self, tasks: List):
        """
        热更新该流关联的任务列表（新增/删除任务时调用）
        """
        with self.lock:
            old_ids = {t.id for t in self.tasks}
            new_ids = {t.id for t in tasks}

            # 保留还在的任务的上次运行时间
            new_last_run = {}
            for task in tasks:
                new_last_run[task.id] = self.task_last_run.get(task.id, 0.0)

            self.tasks = tasks
            self.task_last_run = new_last_run

            # 重新计算 only_key_frames 需求：只要关联了任一全帧算法，就必须拉全部帧
            full_frame_codes = set(getattr(config, "FULL_FRAME_ALGORITHMS", []))
            need_full_frame = any(
                str(getattr(task, "algorithmCode", "")) in full_frame_codes
                for task in tasks
            )
            new_only_key_frames = not need_full_frame
            old_only_key_frames = self._stream_start_params.get("only_key_frames", True)
            if new_only_key_frames != old_only_key_frames:
                self._stream_start_params["only_key_frames"] = new_only_key_frames
                logger.info(
                    f"StreamWorker 任务列表更新导致 only_key_frames 变更: "
                    f"{self.rtsp_url} {old_only_key_frames} -> {new_only_key_frames}，"
                    f"下次流重建时生效"
                )

            logger.info(f"StreamWorker 任务列表更新: {self.rtsp_url}, "
                        f"新增={len(new_ids - old_ids)}, 移除={len(old_ids - new_ids)}")

    def update_rtsp_url(self, new_rtsp_url: str) -> bool:
        """
        通过 gRPC 通知后端更新流的 RTSP URL，无需关闭重开
        :param new_rtsp_url: 新的 RTSP 地址
        :return: 是否更新成功
        """
        if not self.capture or not self._stream_id:
            logger.warning(f"无法更新 RTSP URL，流尚未打开: {self.rtsp_url}")
            return False
        try:
            ok = self.capture.update_stream_url(self._stream_id, new_rtsp_url)
            if ok:
                logger.info(f"RTSP URL 已更新: {self.rtsp_url} -> {new_rtsp_url}")
                self.rtsp_url = new_rtsp_url
            else:
                logger.warning(f"RTSP URL 更新失败: {self.rtsp_url} -> {new_rtsp_url}")
            return ok
        except Exception as e:
            logger.error(f"更新 RTSP URL 异常: {e}")
            return False

    def _shm_file_exists(self) -> bool:
        """检查当前流对应的共享内存文件是否还存在"""
        if not self._stream_id:
            return False
        if not self._stream_start_params.get("use_shared_mem"):
            return True
        shm_path = f"/dev/shm/{self._stream_id}"
        return os.path.exists(shm_path)

    def _query_stream_status(self) -> Optional[int]:
        """查询服务端当前流状态，失败返回 None"""
        if not self.capture or not self._stream_id:
            return None
        try:
            return self.capture.get_stream_status(self._stream_id)
        except Exception as e:
            logger.error(f"[拉流] 查询服务端流状态失败: {self.rtsp_url}, {e}")
            return None

    def _recover_stream(self) -> bool:
        """
        强制恢复视频流：关闭旧流、重连 gRPC、用相同参数重新 start_stream。
        用于 gRPC 服务重启或共享内存消失后的自愈。
        """
        now = time.time()
        if now - self._last_recovery_at < self._recovery_cooldown_sec:
            logger.debug(f"[恢复] 冷却中，跳过: {self.rtsp_url}")
            return False
        self._last_recovery_at = now

        logger.warning(
            f"[恢复] 开始重建视频流: {self.rtsp_url}, "
            f"历史恢复尝试={self._recovery_attempts}"
        )

        try:
            # 1. 如果还连着服务端，尝试通知它停止旧流（忽略失败）
            if self.capture and self._stream_id:
                try:
                    self.capture.stop_stream(self._stream_id)
                except Exception as e:
                    logger.error(f"[恢复] 停止旧流时异常（可忽略）: {e}")

            # 2. 重建 gRPC 连接
            if not self.capture or not self.capture.reconnect():
                logger.error(f"[恢复] gRPC 重连失败: {self.rtsp_url}")
                return False

            # 3. 用原参数重新启动流
            new_stream_id = self.capture.start_stream(
                self.rtsp_url, **self._stream_start_params
            )
            if not new_stream_id:
                logger.error(f"[恢复] 重新启动流失败: {self.rtsp_url}")
                return False

            self._stream_id = new_stream_id
            self._consecutive_read_failures = 0
            self._recovery_attempts = 0
            self._last_frame_at = time.time()
            logger.info(f"[恢复] 视频流重建成功: {self.rtsp_url}, new_stream_id={new_stream_id}")
            return True
        except Exception as e:
            logger.error(f"[恢复] 重建视频流异常: {self.rtsp_url}, {e}", exc_info=True)
            return False

    def _pre_detect_with_yolo(self, frame, tasks: List) -> Tuple[List, List[Box], set]:
        """
        使用 YOLO 模型对任务进行预检测。

        注意：即使 YOLO 没有检测到任务配置的类别，也**不会**跳过该任务，而是仅跳过 SAM3 调用，
        让任务继续进入分析流程。因为部分算法（如单人作业/单人滞留）依赖时间累计，空输入时
        也需要走一遍逻辑推理以更新内部状态。

        :param frame: 当前视频帧
        :param tasks: 待检测的任务列表
        :return: (通过预检测的任务列表, YOLO 检测到的 Box 列表[格式同 SAM3], 需跳过 SAM3 的任务 ID 集合)
        """
        if not tasks:
            return [], [], set()

        # 收集每个任务所需的 YOLO 预检测配置
        # 结构: [(task, model_name, required_classes), ...]
        task_configs = []
        for task in tasks:
            algo_code = str(task.algorithmCode)
            pre_config = config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES.get(algo_code, {})
            if not pre_config:
                # 没有配置预检测，直接通过
                task_configs.append((task, None, None))
                continue
            # 取该算法配置的第一个模型（当前只支持单模型）
            for model_name, required_classes in pre_config.items():
                task_configs.append((task, model_name, required_classes))
                break

        # 如果没有任务需要预检测，全部通过
        yolo_required = [(t, m, c) for t, m, c in task_configs if m is not None]
        if not yolo_required:
            return tasks, [], set()

        # 收集每个模型需要的类别（按模型去重，合并所有任务的类别需求）
        # model_name -> set(类别名)
        model_required_classes: Dict[str, set] = {}
        for _, model_name, required_classes in yolo_required:
            if model_name not in model_required_classes:
                model_required_classes[model_name] = set()
            model_required_classes[model_name].update(required_classes)

        # 执行 YOLO 推理，收集所有检测到的 Box（格式同 SAM3）
        yolo_results: Dict[str, Optional[List[Box]]] = {}
        all_yolo_boxes: List[Box] = []
        for model_name, class_names in model_required_classes.items():
            client = _get_yolo_client(model_name)
            if client is None:
                logger.warning(f"YOLO 客户端不可用 {model_name}，相关任务跳过预检测直接调用 SAM3")
                yolo_results[model_name] = None
                continue

            # 类别名 -> 类别 ID，类似 ultralytics 的 classes 参数
            class_ids = []
            unknown_classes = []
            for name in class_names:
                # 反向查找 label_map：name -> cls_id
                found_ids = [cid for cid, cname in client.label_map.items() if cname == name]
                if found_ids:
                    class_ids.extend(found_ids)
                else:
                    unknown_classes.append(name)
            if unknown_classes:
                logger.warning(f"YOLO 类别映射未找到: {unknown_classes}，将回退到全类别检测")
                class_ids = None  # 有未知类别时，不过滤，全量检测

            try:
                boxes = client.predict(frame, classes=class_ids)
                yolo_results[model_name] = boxes
                all_yolo_boxes.extend(boxes)
                label_counts = {}
                for b in boxes:
                    label_counts[b.label] = label_counts.get(b.label, 0) + 1
                logger.debug(f"YOLO 预检测 {model_name} | 框数={len(boxes)}, 类别分布={label_counts}")
            except Exception as e:
                logger.error(f"YOLO 预检测失败 {model_name}: {e}")
                yolo_results[model_name] = []

        # 判断每个任务是否通过预检测
        passed_tasks = []
        skip_sam3_task_ids = set()  # YOLO 未检测到目标、需要跳过 SAM3 的任务 ID
        now = time.time()
        for task, model_name, required_classes in task_configs:
            # 即使 YOLO 没有检测到目标，也要让任务进入后续分析流程（支持时间累计类算法）
            passed_tasks.append(task)

            if model_name is None:
                # 未配置预检测，直接调用 SAM3
                continue

            boxes = yolo_results.get(model_name)
            if boxes is None:
                # YOLO 客户端不可用，跳过预检测直接调用 SAM3
                logger.info(f"YOLO 客户端不可用，跳过预检测 task={task.id}, algo={task.algorithmCode}")
                continue

            detected_labels = {b.label for b in boxes}

            # 检查 required_classes 中是否有任何一个被检测到
            has_target = any(cls in detected_labels for cls in required_classes)

            if has_target:
                matched = [cls for cls in required_classes if cls in detected_labels]
                logger.info(f"YOLO 预检测通过 task={task.id}, algo={task.algorithmCode}, 匹配到: {matched}")
                # 更新运行时间，避免该任务立即再次进入 ready 状态
                with self.lock:
                    self.task_last_run[task.id] = now
            else:
                # YOLO 未检测到目标，跳过 SAM3 以节省资源，但仍进入分析流程更新时间累计状态
                skip_sam3_task_ids.add(task.id)
                logger.info(f"YOLO 预检测未通过 task={task.id}, algo={task.algorithmCode}, 跳过 SAM3 但仍继续分析")

        return passed_tasks, all_yolo_boxes, skip_sam3_task_ids

    def stop(self):
        """请求 worker 线程停止。

        注意：此方法**不**关闭本地 capture 资源（mmap/信号量/gRPC），
        以避免与仍在运行的 worker 线程发生竞争导致 segfault。
        本地资源由 worker 线程在 run() 退出时自行释放，或由 orchestrator
        在 join() 成功后做兜底清理。
        """
        self.stopped = True

    def run(self):
        logger.info(f"StreamWorker 启动: {self.rtsp_url}, 关联任务数={len(self.tasks)}")

        # 1. 打开视频流（使用 gRPC 生命周期管理 + 共享内存读帧）
        self.capture = RTSPClient(self._server_address)
        self._stream_id = self.capture.start_stream(
            self.rtsp_url, **self._stream_start_params
        )
        if not self._stream_id:
            logger.error(f"打开视频流失败: {self.rtsp_url}")
            return
        self._last_frame_at = time.time()

        # 2. 主循环
        while not self.stopped:
            try:
                frame_ts, frame = self.capture.read(
                    self._stream_id, blocking=True, timeout_ms=5000
                )
                if frame is None:
                    self._consecutive_read_failures += 1
                    now = time.time()
                    no_frame_duration = now - self._last_frame_at
                    shm_gone = (
                        self._stream_start_params.get("use_shared_mem")
                        and not self._shm_file_exists()
                    )

                    # 优先根据服务端状态决定是否需要重建
                    server_status = self._query_stream_status()
                    if shm_gone:
                        logger.error(
                            f"[拉流] 共享内存已消失: /dev/shm/{self._stream_id}, "
                            f"rtsp={self.rtsp_url}"
                        )
                        need_recover = True
                    elif server_status == STATUS_CONNECTED:
                        # 服务端认为流正常，只是本地暂时没有读到帧，不重建
                        logger.debug(
                            f"[拉流] 服务端流状态正常，继续等待: "
                            f"rtsp={self.rtsp_url}, 无帧时长={no_frame_duration:.1f}s"
                        )
                        need_recover = False
                    elif server_status in (STATUS_DISCONNECTED, STATUS_NOT_FOUND):
                        logger.warning(
                            f"[拉流] 服务端流状态异常: "
                            f"rtsp={self.rtsp_url}, status={server_status}"
                        )
                        need_recover = True
                    else:
                        # 查询不到状态（gRPC 异常等），保守等待，不盲目重建
                        logger.debug(
                            f"[拉流] 无法确认服务端状态，继续等待: "
                            f"rtsp={self.rtsp_url}, 无帧时长={no_frame_duration:.1f}s"
                        )
                        need_recover = False

                    in_cooldown = (now - self._last_recovery_at) < self._recovery_cooldown_sec

                    if need_recover and not in_cooldown:
                        logger.warning(
                            f"[拉流] 触发流恢复: rtsp={self.rtsp_url}, "
                            f"连续失败={self._consecutive_read_failures}, "
                            f"无帧时长={no_frame_duration:.1f}s, "
                            f"shm_gone={shm_gone}, server_status={server_status}"
                        )
                        if self._recover_stream():
                            continue
                        self._recovery_attempts += 1
                        logger.error(
                            f"[拉流] 流恢复失败: rtsp={self.rtsp_url}, "
                            f"本轮尝试={self._recovery_attempts}/{self._max_recovery_attempts}"
                        )
                        if self._recovery_attempts >= self._max_recovery_attempts:
                            logger.error(
                                f"[拉流] 流恢复连续失败 {self._max_recovery_attempts} 次，"
                                f"退出 worker 等待 orchestrator 重建: {self.rtsp_url}"
                            )
                            break
                        time.sleep(5)
                    elif need_recover and in_cooldown:
                        logger.debug(
                            f"[拉流] 满足恢复条件但处于冷却中，跳过: "
                            f"rtsp={self.rtsp_url}, 无帧时长={no_frame_duration:.1f}s"
                        )
                    continue

                # 读帧成功，重置故障计数
                self._consecutive_read_failures = 0
                self._recovery_attempts = 0
                self._last_frame_at = time.time()

                # 应用层日志：读取到图片，打印设备/任务信息（不改动 RTSPClient 封装）
                try:
                    if self.tasks:
                        first_task = self.tasks[0]
                        device_info = (
                            f"deviceAlgorithmIp={first_task.deviceAlgorithmIp}, "
                            f"deviceChannel={first_task.deviceChannel}, "
                            f"deviceId={first_task.deviceId}, "
                            f"deviceName={first_task.deviceName}"
                        )
                        task_info = [
                            {
                                "task_id": t.id,
                                "algorithmCode": t.algorithmCode,
                                "algorithmName": t.algorithmName,
                            }
                            for t in self.tasks
                        ]
                    else:
                        device_info = "deviceAlgorithmIp=None, deviceChannel=None, deviceId=None, deviceName=None"
                        task_info = []

                    # 避免日志刷屏：每 5 秒输出一次 INFO 级别汇总
                    now_log = time.time()
                    if not hasattr(self, "_last_frame_info_log_at") or now_log - self._last_frame_info_log_at >= 10.0:
                        self._last_frame_info_log_at = now_log
                        logger.info(
                            f"[拉流] 读取到图片 | rtsp={self.rtsp_url}, {device_info}, "
                            f"stream_id={self._stream_id}, frame_ts={frame_ts}, "
                            f"frame_shape={frame.shape if frame is not None else None}, "
                            f"tasks={task_info}"
                        )
                except Exception:
                    logger.debug("读取到图片日志打印失败", exc_info=True)

                frame_captured_time = time.time()
                # C++ 端 ts 是 steady_clock 的毫秒时间戳（从系统启动开始计数）
                # Python 的 time.monotonic() 也是 CLOCK_MONOTONIC（系统启动为 epoch），可以对应
                frame_ts_sec = frame_ts / 1000.0
                monotonic_now = time.monotonic()
                frame_delay_ms = (monotonic_now - frame_ts_sec) * 1000
                # 计算帧间隔
                if self._last_frame_ts > 0:
                    ts_interval = frame_ts - self._last_frame_ts
                else:
                    ts_interval = 0
                self._last_frame_ts = frame_ts
                now = frame_captured_time
                ready_tasks = []

                with self.lock:
                    for task in self.tasks:
                        # 使用 algorithms.yaml 中按算法码配置的检测间隔
                        algo_code = str(task.algorithmCode)
                        interval = config.ALGORITHM_INTERVALS.get(
                            algo_code, config.DEFAULT_ALGORITHM_INTERVAL
                        )
                        last_run = self.task_last_run.get(task.id, 0)
                        if now - last_run >= interval:
                            ready_tasks.append(task)

                t_ready = time.time()
                if not ready_tasks:
                    # 当前没有任务需要分析，短暂休眠后继续
                    time.sleep(0.05)
                    continue

                # 3. YOLO 预检测：未检测到目标则跳过 SAM3
                t_yolo_start = time.time()
                ready_tasks, yolo_boxes, skip_sam3_task_ids = self._pre_detect_with_yolo(frame, ready_tasks)
                logger.debug(f"预检测完成 | 通过预检测的任务数={len(ready_tasks)}, YOLO 检测到的框数={len(yolo_boxes)}, 跳过 SAM3 的任务数={len(skip_sam3_task_ids)}")
                t_yolo_end = time.time()
                if not ready_tasks:
                    continue

                # 4. 按 SAM3 URL 分组，每组分别请求 SAM3
                sam3_groups: Dict[str, List] = {}
                for task in ready_tasks:
                    if task.id in skip_sam3_task_ids:
                        continue
                    algo_code = str(task.algorithmCode)
                    url = config.ALGORITHM_SAM3_URL.get(algo_code, config.SAM3_URL_OBJ)
                    sam3_groups.setdefault(url, []).append(task)

                all_boxes: List[Box] = []
                has_sam3_prompt = False
                t_sam3_start = t_sam3_end = 0.0  # 当没有 SAM3 请求时避免 UnboundLocalError
                for url, group_tasks in sam3_groups.items():
                    merged_prompts, return_mask = merge_prompts_for_tasks(group_tasks)
                    if not merged_prompts:
                        logger.warning(
                            f"没有可用的 SAM3 prompt | URL={url}, "
                            f"tasks={[t.id for t in group_tasks]}"
                        )
                        continue

                    has_sam3_prompt = True
                    t_sam3_start = time.time()
                    logger.info(f"SAM3 请求 | URL={url} | Prompts: {merged_prompts}, return_mask={return_mask}")
                    boxes = call_sam3(
                        frame, merged_prompts,
                        confidence_threshold=0.5,
                        return_mask=return_mask,
                        url=url,
                    )
                    t_sam3_end = time.time()

                    # 统计 SAM3 返回的类别
                    label_counts = {}
                    for b in boxes:
                        label_counts[b.label] = label_counts.get(b.label, 0) + 1
                    logger.info(
                        f"SAM3 返回 | URL={url} | 总框数={len(boxes)}, "
                        f"类别分布={label_counts} | 耗时={(t_sam3_end-t_sam3_start)*1000:.1f}ms"
                    )
                    all_boxes.extend(boxes)

                if not has_sam3_prompt and yolo_boxes:
                    logger.warning(f"所有任务均无 SAM3 prompt, tasks={[t.id for t in ready_tasks]}, 使用yolo结果进行分析")
                    all_boxes = list(yolo_boxes)

                # 按需合并 YOLO 框（补充 SAM3 缺失的类别）
                needs_yolo_merge = any(str(t.algorithmCode) in config.USE_YOLO_BOXES for t in ready_tasks)
                if has_sam3_prompt and needs_yolo_merge and yolo_boxes:
                    sam_labels = {b.label for b in all_boxes}
                    merged = [yb for yb in yolo_boxes if yb.label not in sam_labels]
                    if merged:
                        all_boxes.extend(merged)
                        logger.info(f"YOLO 框合并 | 补充类别: {set([b.label for b in merged])}, 合并后总框数={len(all_boxes)}")

                # 即使 SAM3/YOLO 都没有检测到任何目标，也要对每个任务走一遍分析逻辑。
                # 部分算法（如单人作业/单人滞留）依赖时间累计，空输入时需要更新内部状态。
                if not all_boxes:
                    logger.debug(f"当前帧未检测到任何目标，仍继续分析 | ready_tasks={[t.id for t in ready_tasks]}")

                # 5. 各任务独立分析并上报
                t_analyze_start = time.time()
                total_violations = 0
                with self.lock:
                    for task in ready_tasks:
                        try:
                            algo_code = str(task.algorithmCode)
                            # 判断该任务是否需要 SAM3：配置了 prompt，或未配置 prompt 但任务带围栏（回退到 person）
                            sam3_prompts = config.ALGORITHM_SAM3_PROMPT.get(algo_code, [])
                            if algo_code not in config.ALGORITHM_SAM3_PROMPT and task.electricFence:
                                sam3_prompts = ["person"]

                            # 未配置 SAM3 的算法任务，直接使用 YOLO 预检测结果作为输入框
                            if not sam3_prompts and yolo_boxes:
                                task_boxes = list(yolo_boxes)
                                logger.info(
                                    f"任务未配置 SAM3 prompt，使用 YOLO 框分析 | "
                                    f"task={task.id}, algo={algo_code}, boxes={len(task_boxes)}"
                                )
                            else:
                                task_boxes = all_boxes

                            if len(task_boxes) == 0:
                                logger.debug(f"当前帧未检测到任何目标，仍继续分析 task={task.id}, algo={task.algorithmCode}")
                            else:
                                logger.info(f"开始分析 task={task.id}, algo={task.algorithmCode}, input_boxes={len(task_boxes)}")
                            fences = self._get_fences_for_task(task)
                            image_width = frame.shape[1]
                            image_height = frame.shape[0]
                            violations = analyze_for_task(task_boxes, task, fences=fences, image_width=image_width, image_height=image_height, frame=frame)
                            if len(task_boxes) > 0 and len(violations) > 0:
                                logger.info(f"分析完成 task={task.id}, 规则引擎违规数={len(violations)}")
                            if len(violations) == 0:
                                # 即使没有违规，也记录分析时间，避免频繁进入 ready 状态
                                self.task_last_run[task.id] = now
                                continue

                            # 分类器二次确认（如吸烟检测后调用 resnet_smoke 分类模型）
                            violations = classify_for_task(
                                frame, task, violations,
                                image_width=image_width, image_height=image_height
                            )
                            logger.info(f"分类过滤完成 task={task.id}, 剩余违规数={len(violations)}")
                            if len(violations) == 0:
                                self.task_last_run[task.id] = now
                                continue

                            # VL 大模型二次确认：遍历每个违规目标裁剪后分别判断，否认则过滤
                            violations = vl_analyze_for_task(
                                frame, task, violations,
                                image_width=image_width, image_height=image_height
                            )
                            logger.info(f"VL 二次确认后 task={task.id}, 最终违规数={len(violations)}")

                            if violations:
                                # 报警去重：按算法码单独配置
                                dedup_cfg = config.ALERT_DEDUP_CONFIG.get(str(task.algorithmCode), {})
                                if dedup_cfg.get("enabled", False):
                                    raw_count = len(violations)
                                    violations = alert_dedup.filter(
                                        violations,
                                        device_id=task.deviceId or "",
                                        algorithm_code=str(task.algorithmCode),
                                        cooldown_seconds=dedup_cfg.get("cooldown_seconds"),
                                        iou_thresh=dedup_cfg.get("iou_thresh"),
                                    )
                                    logger.info(
                                        f"报警去重 task={task.id} | 原始={raw_count}, 保留={len(violations)}, "
                                        f"cooldown={dedup_cfg.get('cooldown_seconds')}s, iou={dedup_cfg.get('iou_thresh')}"
                                    )

                                if violations:
                                    logger.warning(f"⚠️ 检测到违规! task={task.id}, violations={len(violations)}")
                                else:
                                    logger.info(f"去重后无新违规 task={task.id}")
                                    self.task_last_run[task.id] = now
                                    continue

                                total_violations += len(violations)

                                # 绘制 OSD（检测框、违规高亮、告警横幅、时间戳）
                                alert_frame = render_alert_frame(
                                    frame=frame,
                                    all_boxes=task_boxes,
                                    violations=violations,
                                    task=task,
                                    fences=fences,
                                    draw_all_boxes=False,
                                    draw_violation_boxes=True
                                )
                                alert_context = {
                                    'nvr_ip': task.deviceAlgorithmIp or '0.0.0.0',
                                    'channel': task.deviceChannel or '1',
                                    'task_id': task.id,
                                }
                                uploader.add_alert(alert_frame, frame, alert_context, task.algorithmCode)
                                logger.info(f"✅ 已加入上传队列 task={task.id}")
                            else:
                                logger.debug(f"无违规 task={task.id}")
                        except Exception as e:
                            logger.error(f"任务分析异常 task={task.id}: {e}", exc_info=True)
                        finally:
                            self.task_last_run[task.id] = now

                t_loop_end = time.time()
                # 输出端到端阶段耗时统计
                logger.debug(
                    f"阶段耗时 | "
                    f"帧延迟={frame_delay_ms:.1f}ms | "
                    f"帧间隔={ts_interval}ms | "
                    f"帧捕获→就绪={(t_ready-frame_captured_time)*1000:.1f}ms | "
                    f"YOLO预检={(t_yolo_end-t_yolo_start)*1000:.1f}ms | "
                    f"SAM3推理={(t_sam3_end-t_sam3_start)*1000:.1f}ms | "
                    f"分析+OSD+上报={(t_loop_end-t_analyze_start)*1000:.1f}ms | "
                    f"Python端到端={(t_loop_end-frame_captured_time)*1000:.1f}ms | "
                    f"违规数={total_violations}"
                )

            except Exception as e:
                logger.error(f"StreamWorker 主循环异常: {self.rtsp_url}, {e}", exc_info=True)
                time.sleep(1)

        # 3. 清理资源
        if self.capture:
            if self._stream_id:
                try:
                    self.capture.stop_stream(self._stream_id)
                except Exception:
                    pass
            try:
                self.capture.disconnect()
            except Exception:
                pass
        logger.info(f"StreamWorker 停止: {self.rtsp_url}")
