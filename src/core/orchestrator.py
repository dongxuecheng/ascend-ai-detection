import threading
import time
from typing import List, Dict, Optional, Tuple

from config.config import config
from utils.logger import setup_logger
from task.getTask import TaskManager
from core.stream_worker import StreamWorker

logger = setup_logger("orchestrator")


# 设备唯一标识: (deviceAlgorithmIp, deviceChannel)
DeviceKey = Tuple[str, str]


def _device_key(task) -> Optional[DeviceKey]:
    """从任务中提取稳定的设备标识"""
    ip = task.deviceAlgorithmIp
    ch = task.deviceChannel
    if ip and ch:
        return (str(ip), str(ch))
    return None


class Orchestrator:
    """
    编排器：管理所有 StreamWorker 的生命周期，与 TaskManager 同步任务列表
    使用 (deviceAlgorithmIp, deviceChannel) 作为 worker 的唯一 key，RTSP URL 变化不影响 key 稳定性
    """

    def __init__(self):
        self.tm = TaskManager(
            url=config.GET_TASK_URL,
            algorithm_code="",
            refresh_interval=config.REQUEST_INTERVAL
        )
        # (deviceAlgorithmIp, deviceChannel) -> StreamWorker
        self.workers: Dict[DeviceKey, StreamWorker] = {}
        self._stop = False
        self.lock = threading.Lock()

    def _build_rtsp_url(self, task) -> Optional[str]:
        """
        从任务对象构造 RTSP URL。
        优先使用 playbackAddress，否则尝试通过 API 获取。
        """
        if task.playbackAddress:
            return task.playbackAddress
        return None

    def _filter_supported_tasks(self, tasks: List) -> List:
        """
        根据 config.ALGORITHM_CODES 过滤掉不支持算法的任务
        """
        supported = set(str(code) for code in config.ALGORITHM_CODES)
        filtered = []
        for task in tasks:
            algo_code = str(task.algorithmCode)
            if algo_code in supported:
                filtered.append(task)
            else:
                logger.warning(f"剔除不支持的算法任务: task={task.id}, algorithmCode={algo_code}, "
                               f"支持列表={config.ALGORITHM_CODES}")
        if len(filtered) != len(tasks):
            logger.info(f"任务过滤完成: 原始={len(tasks)}, 保留={len(filtered)}, 剔除={len(tasks) - len(filtered)}")
        return filtered

    def _group_tasks_by_device(self, tasks: List) -> Dict[DeviceKey, List]:
        """
        按设备标识 (deviceAlgorithmIp, deviceChannel) 对任务分组
        """
        tasks = self._filter_supported_tasks(tasks)
        device_map: Dict[DeviceKey, List] = {}
        for task in tasks:
            key = _device_key(task)
            if not key:
                logger.warning(f"任务 {task.id} 缺少 deviceAlgorithmIp 或 deviceChannel，跳过")
                continue
            if key not in device_map:
                device_map[key] = []
            device_map[key].append(task)
        return device_map

    def start(self):
        """
        启动编排器主循环：等待 TaskManager 首次同步，然后定期 diff 任务列表
        """
        logger.info("Orchestrator 启动，等待 TaskManager 首次同步...")
        time.sleep(2)  # 给 TaskManager 足够时间完成首次拉取

        try:
            while not self._stop:
                try:
                    all_tasks = self.tm.get_all_tasks()
                    device_map = self._group_tasks_by_device(all_tasks)
                    current_keys = set(device_map.keys())

                    with self.lock:
                        existing_keys = set(self.workers.keys())

                        # ---- Step 1: 检测已有 worker 的 RTSP 地址变化（海康平台流地址会动态更新）----
                        for key in current_keys & existing_keys:
                            tasks = device_map[key]
                            new_url = self._build_rtsp_url(tasks[0])
                            old_url = self.workers[key].rtsp_url
                            if new_url and new_url != old_url:
                                logger.info(f"检测到 RTSP 地址变化 device={key}: {old_url} -> {new_url}")
                                self.workers[key].update_rtsp_url(new_url)

                        # ---- Step 2: 新增设备 ----
                        for key in current_keys - existing_keys:
                            tasks = device_map[key]
                            url = self._build_rtsp_url(tasks[0])
                            if not url:
                                logger.warning(f"设备 {key} 无法获取 RTSP 地址，跳过")
                                continue
                            worker = StreamWorker(url, tasks)
                            worker.start()
                            self.workers[key] = worker
                            logger.info(f"新增视频流 worker: device={key}, url={url}, 任务数={len(tasks)}")

                        # ---- Step 3: 移除设备 ----
                        for key in existing_keys - current_keys:
                            self.workers[key].stop()
                            del self.workers[key]
                            logger.info(f"移除视频流 worker: device={key}")

                        # ---- Step 4: 已有设备但任务列表可能变更 ----
                        for key in current_keys & existing_keys:
                            new_tasks = device_map[key]
                            old_task_ids = {t.id for t in self.workers[key].tasks}
                            new_task_ids = {t.id for t in new_tasks}
                            if old_task_ids != new_task_ids:
                                added = new_task_ids - old_task_ids
                                removed = old_task_ids - new_task_ids
                                logger.info(f"已有设备任务变更: device={key}, 新增={len(added)}, 移除={len(removed)}")
                                if added:
                                    logger.info(f"  新增任务 IDs: {sorted(added)}")
                                if removed:
                                    logger.info(f"  移除任务 IDs: {sorted(removed)}")
                                self.workers[key].update_tasks(new_tasks)
                            else:
                                logger.debug(f"已有设备任务无变化: device={key}, 任务数={len(new_tasks)}")

                        # ---- Step 5: 检测已退出 worker 并重建 ----
                        # StreamWorker 在流连续恢复失败后会退出，需要 orchestrator 重新拉起
                        for key in list(current_keys & existing_keys):
                            worker = self.workers[key]
                            if not worker.is_alive():
                                logger.warning(f"检测到 worker 已退出，准备重建: device={key}")
                                tasks = device_map[key]
                                url = self._build_rtsp_url(tasks[0])
                                if not url:
                                    logger.warning(f"设备 {key} 无法获取 RTSP 地址，跳过重建")
                                    continue
                                new_worker = StreamWorker(url, tasks)
                                new_worker.start()
                                self.workers[key] = new_worker
                                logger.info(f"已重建视频流 worker: device={key}, url={url}, 任务数={len(tasks)}")

                except Exception as e:
                    logger.error(f"Orchestrator 同步异常: {e}", exc_info=True)

                logger.info(f"本轮同步完成: 当前流数={len(self.workers)}, 等待 {config.REQUEST_INTERVAL}s 后下一轮...")
                time.sleep(config.REQUEST_INTERVAL)
        finally:
            # 任何情况下（正常退出、KeyboardInterrupt、异常）都执行清理
            self._cleanup_workers()

    def _cleanup_workers(self):
        """停止并回收所有 StreamWorker 线程（幂等，可多次调用）"""
        logger.info("Orchestrator 停止，清理所有 StreamWorker...")
        with self.lock:
            for worker in list(self.workers.values()):
                worker.stop()
            for worker in list(self.workers.values()):
                worker.join(timeout=10)
            # 兜底：对仍未释放本地资源的 capture 做强制清理
            for worker in list(self.workers.values()):
                if worker.capture and worker.capture.is_connected():
                    try:
                        if worker._stream_id:
                            worker.capture.stop_stream(worker._stream_id)
                    except Exception:
                        pass
                    try:
                        worker.capture.disconnect()
                    except Exception:
                        pass
            self.workers.clear()

    def shutdown(self):
        """请求停止编排器，并同步等待所有 worker 清理完成"""
        self._stop = True
        self._cleanup_workers()
