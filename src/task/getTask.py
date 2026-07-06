import threading
import time
import requests
from config.config import config
from utils.logger import setup_logger

logger = setup_logger("task_manager")

class Task:
    def __init__(self):
        self.fwqCode = None
        self.id = None
        self.alarmTaskName = None
        self.deviceId = None
        self.deviceName = None
        self.deviceAlgorithmIp = None
        self.cameraType = None
        self.deviceChannel = None
        self.videoName = None
        self.videoPassword = None
        self.algorithmCode = None
        self.algorithmName = None
        self.beforeDrawPhoto = None
        self.electricFence = None
        self.playbackAddress = None
        self.deviceCode = None
        self.sleepJudgeTime = None
        self.personLimitNum = None

    # 全部属性的字符串表示，方便调试和日志记录
    def __str__(self):
        return (f"Task(id={self.id}, alarmTaskName={self.alarmTaskName}, deviceId={self.deviceId}, "
                f"deviceName={self.deviceName}, deviceAlgorithmIp={self.deviceAlgorithmIp}, "
                f"cameraType={self.cameraType}, deviceChannel={self.deviceChannel}, videoName={self.videoName}, "
                f"videoPassword={self.videoPassword}, algorithmCode={self.algorithmCode}, "
                f"algorithmName={self.algorithmName}, beforeDrawPhoto={self.beforeDrawPhoto}, "
                f"electricFence={self.electricFence}, playbackAddress={self.playbackAddress}, "
                f"deviceCode={self.deviceCode}, sleepJudgeTime={self.sleepJudgeTime}, "
                f"personLimitNum={self.personLimitNum})")


# 获取同一个算法的所有的任务
class TaskManager: 
    def __init__(self, url: str, algorithm_code: str="", refresh_interval: int = 10):
        self.url = url
        self.algorithm_code = algorithm_code
        self.refresh_interval = refresh_interval
        self.tasks = {}
        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self.loop_sync, daemon=True)
        self.t.start()


    def loop_sync(self):
        while not self.stopped:
            self._sync_config()
            time.sleep(self.refresh_interval)

    def _get_stream_url(self, ip, channel):
        """Helper to fetch the stream address from the new API"""
        try:
            params = {
                "ip": ip,
                "channel": channel,
                "protocol": "" 
            }
            resp = requests.post(config.GET_RTSP_URL, params=params, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            if result.get("code") == 0:
                # Assuming the URL is inside 'data'. Adjust based on actual JSON structure.
                # logger.info(f"获取流地址成功 (IP: {ip}, Channel: {channel}): {result.get('msg')}")
                return result.get("msg")
            return None
        except Exception as e:
            # logger.warning(f"获取流地址失败 (IP: {ip}, Channel: {channel}): {e}")
            return None

    def _sync_config(self):
        with self.lock:
            old_count = len(self.tasks)
            old_ids = set(self.tasks.keys())
            try:
                self.tasks = {}
                payload = {"algorithmCode": self.algorithm_code}
                resp = requests.post(self.url, params=payload, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                new_tasks = {}
                for item in data.get("data", []):
                    task = Task()
                    task.fwqCode = item.get("fwqCode")
                    task.id = item.get("id")
                    task.alarmTaskName = item.get("alarmTaskName")
                    task.deviceId = item.get("deviceId")
                    task.deviceName = item.get("deviceName")
                    task.deviceAlgorithmIp = item.get("deviceAlgorithmIp")
                    task.cameraType = item.get("cameraType")
                    task.deviceChannel = item.get("deviceChannel")
                    task.videoName = item.get("videoName")
                    task.videoPassword = item.get("videoPassword")
                    task.algorithmCode = item.get("algorithmCode")
                    task.algorithmName = item.get("algorithmName")
                    task.beforeDrawPhoto = item.get("beforeDrawPhoto")
                    task.electricFence = item.get("electricFence")
                    task.playbackAddress = item.get("playbackAddress")
                    task.deviceCode = item.get("deviceCode")
                    task.sleepJudgeTime = item.get("sleepJudgeTime")
                    task.personLimitNum = item.get("personLimitNum")

                    if stream_url := self._get_stream_url(task.deviceAlgorithmIp, task.deviceChannel):
                        task.playbackAddress = stream_url
                    # if task.deviceAlgorithmIp and task.deviceChannel:
                    #     task.playbackAddress = self._get_stream_url(task.deviceAlgorithmIp, task.deviceChannel)

                    new_tasks[task.id] = task

                new_ids = set(new_tasks.keys())
                added = new_ids - old_ids
                removed = old_ids - new_ids
                self.tasks = new_tasks

                if added or removed or old_count != len(new_tasks):
                    logger.info(f"同步任务配置成功 | 任务数: {old_count} -> {len(new_tasks)} | "
                                f"新增={len(added)} 移除={len(removed)}")
                    if added:
                        logger.info(f"新增任务 IDs: {sorted(added)}")
                    if removed:
                        logger.info(f"移除任务 IDs: {sorted(removed)}")
                else:
                    logger.debug(f"同步任务配置成功，任务无变化，当前任务数: {len(new_tasks)}")
            except Exception as e:
                logger.error(f"同步任务配置失败: {e}", exc_info=True)

    def get_task(self, task_id) -> Task:
        with self.lock:
            return self.tasks.get(task_id, None)

    def get_tasks_by_device(self):
        with self.lock:
            device_map = {}
            for task in self.tasks.values():
                if task.deviceId not in device_map:
                    device_map[task.deviceId] = []
                device_map[task.deviceId].append(task)
            return device_map

    def get_all_tasks(self):
        with self.lock:
            return list(self.tasks.values())
    
    def task_alive(self, task_id) -> bool:
        with self.lock:
            return task_id in self.tasks

    def stop(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join()