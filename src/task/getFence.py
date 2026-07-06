import threading
import time
import requests
import numpy as np
from config.config import config
from utils.logger import setup_logger

logger = setup_logger("fence")

FENCE_API_URL = config.GET_FENCE_URL  # 电子围栏接口地址


class TaskFence:
    """
    电子围栏管理器（单例模式）。
    一个实例管理所有任务的围栏数据，后台线程定期同步。
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, sync_interval: int = 30):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # (algorithm_code, nvr_ip, channel) -> [np.array, ...]
        self._cache: dict = {}
        self._keys: set = set()  # 需要同步的 key 集合
        self._data_lock = threading.Lock()
        self._stop = False
        self._sync_interval = sync_interval

        self._t = threading.Thread(target=self._loop_sync, daemon=True)
        self._t.start()
        logger.info("TaskFence 单例初始化完成")

    def register(self, algorithm_code: str, nvr_ip: str, channel: str):
        """注册一个需要同步的围栏 key"""
        key = (str(algorithm_code), str(nvr_ip), str(channel))
        with self._data_lock:
            if key not in self._keys:
                self._keys.add(key)
                logger.info(f"TaskFence 注册 | algo={algorithm_code}, ip={nvr_ip}, ch={channel}")

    def get_fence(self, algorithm_code: str, nvr_ip: str, channel: str):
        """
        获取指定任务的围栏数据。
        如果该 key 未注册，会自动注册并尝试首次同步。
        """
        key = (str(algorithm_code), str(nvr_ip), str(channel))
        self.register(algorithm_code, nvr_ip, channel)
        with self._data_lock:
            return self._cache.get(key)

    def _loop_sync(self):
        """后台线程：定期同步所有已注册的围栏"""
        while not self._stop:
            keys = []
            with self._data_lock:
                keys = list(self._keys)
            for key in keys:
                self._sync_one(key)
            time.sleep(self._sync_interval)

    def _sync_one(self, key):
        """同步单个围栏 key"""
        algorithm_code, nvr_ip, channel = key
        try:
            fence_payload = {
                "algorithmCode": algorithm_code,
                "algorithmIp": nvr_ip,
                "channel": channel
            }
            resp = requests.post(FENCE_API_URL, params=fence_payload, timeout=5)
            resp.raise_for_status()
            fence_resp = resp.json()

            fence_data_obj = {}
            if fence_resp.get('code') == 0:
                fence_data_obj = fence_resp.get('data', {})
            else:
                logger.warning(f"获取围栏失败 | key={key}, msg={fence_resp.get('msg')}")

            fences = self._parse_fence_data(fence_data_obj)
            with self._data_lock:
                self._cache[key] = fences

            if fences:
                logger.info(f"围栏同步成功 | key={key}, 区域数={len(fences)}")
        except Exception as e:
            logger.error(f"同步电子围栏失败 | key={key}: {e}")

    def _parse_fence_data(self, data_obj):
        """
        解析 pointCollections 字符串
        格式示例: "531#719,733#75,||1341#542,1723#499,"
        """
        fences = []
        try:
            if not isinstance(data_obj, dict):
                return fences

            points_str = data_obj.get('pointCollections', "")
            if not points_str:
                return fences

            str_fence_list = points_str.split('||')

            for str_fence in str_fence_list:
                pts = []
                items = str_fence.split(',')
                for item in items:
                    if item.strip() and '#' in item:
                        try:
                            x_s, y_s = item.split('#')
                            pts.append([int(x_s), int(y_s)])
                        except ValueError:
                            continue
                if pts:
                    fences.append(np.array(pts, np.int32))
        except Exception as e:
            logger.error(f"围栏数据解析失败: {e}")

        return fences

    def stop(self):
        self._stop = True
