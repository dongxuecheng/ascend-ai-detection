from datetime import datetime
import threading
import queue
import requests
import numpy as np
import cv2
import time
import os

from config.config import config
from utils.logger import setup_logger

logger = setup_logger("upload")

UPLOAD_QUEUE_SIZE = 10
API_UPLOAD_ALERT = config.UPLOAD_URL  # 违章报警图片上传接口地址


class EventUploader:
    """
    负责将违章报警图片上传到服务器。
    独立线程运行，防止网络请求阻塞视频分析主流程。
    """
    def __init__(self):
        # 队列限制长度，防止断网时内存溢出
        self.queue = queue.Queue(maxsize=UPLOAD_QUEUE_SIZE)
        self.stopped = False
        
        # 启动后台上传线程
        self.t = threading.Thread(target=self.worker, daemon=True)
        self.t.start()
        logger.info("异步上传服务已启动")

    def add_alert(self, upload_frame: np.ndarray, original_frame: np.ndarray, alert_context: dict, algorithm_code: str):
        """
        主线程调用此方法提交报警任务。

        :param upload_frame: 用于上传的 OSD 绘制后图片 (numpy array)
        :param original_frame: 违章时刻的原图 (numpy array)，用于本地落盘
        :param alert_context: 包含报警详情的字典, 需包含:
               {
                   'nvr_ip': str,   # 必填: 接口要求的 NVR IP
                   'channel': str,  # 必填: 接口要求的 通道号
                   'task_id': str   # 可选: 任务 ID，用于文件名
               }
        :param algorithm_code: 触发报警的算法代码 (例如 "38")
        """
        if self.queue.full():
            try:
                # 队列满时丢弃最早的任务，优先处理新报警
                self.queue.get_nowait()
                logger.info("上传队列已满，丢弃旧报警任务")
            except queue.Empty:
                pass

        # 必须深拷贝图片，因为主线程后续会继续绘制修改原图
        # 将报警需要的所有信息放入队列（附带入队时间戳，用于计算队列等待延迟）
        self.queue.put((upload_frame.copy(), original_frame.copy(), alert_context, algorithm_code, time.time()))

    def worker(self):
        """后台线程循环"""
        while not self.stopped:
            try:
                # 阻塞等待任务，超时1秒以便检查 stopped 标志
                upload_frame, original_frame, context, algorithm_code, queued_time = self.queue.get(timeout=1)
                queue_wait_ms = (time.time() - queued_time) * 1000
                logger.info(f"报警出队 | 队列等待: {queue_wait_ms:.1f}ms | 当前队列深度: {self.queue.qsize()}")
                self._upload_implementation(upload_frame, original_frame, context, algorithm_code, queue_wait_ms)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"上传线程发生未知异常: {e}", exc_info=True)

    def _save_original_image(self, original_frame: np.ndarray, algorithm_code: str, task_id: str) -> str:
        """
        将违章原图按 类别/日期 保存到本地。

        :return: 保存路径；失败返回空字符串
        """
        try:
            category = config.CODE_DESCRIPTIONS.get(algorithm_code, algorithm_code) or algorithm_code
            # 去除不适合做目录名的字符
            category = category.replace("/", "_").replace("\\", "_").strip() or algorithm_code

            now = datetime.now()
            date_dir = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{time_str}_task{task_id}_algo{algorithm_code}.jpg"

            save_dir = os.path.join(config.VIOLATION_IMAGE_SAVE_DIR, category, date_dir)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, original_frame)
            logger.info(f"违章原图已保存 | path={save_path}")
            return save_path
        except Exception as e:
            logger.error(f"保存违章原图失败: {e}", exc_info=True)
            return ""

    def _upload_implementation(self, upload_frame: np.ndarray, original_frame: np.ndarray, context: dict, algorithm_code: str, queue_wait_ms: float = 0):
        """
        执行具体的 HTTP POST 请求，同时保存原图到本地
        """
        try:
            # 0. 先保存原图（即使上传失败也保留现场）
            task_id = context.get('task_id', 'unknown')
            self._save_original_image(original_frame, algorithm_code, task_id)

            # 1. 图片转码为 JPG 二进制
            _, img_encoded = cv2.imencode('.jpg', upload_frame)
            img_bytes = img_encoded.tobytes()

            # 2. 准备 Query 参数 (URL 后面的参数)
            current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 从 context 中提取 NVR IP 和 通道，如果没有则给默认值防止报错
            nvr_ip = context.get('nvr_ip', '0.0.0.0')
            channel = context.get('channel', '1')

            query_params = {
                "channel": channel,                         # 摄像机通道 (必填)
                "classIndex": algorithm_code,               # 算法类别 (必填, 动态传入)
                "ip": nvr_ip,                               # NVR IP (必填)
                "videoTime": current_time_str,              # 违章时间 (必填)
                "levelId": 1,                               # 报警级别
            }

            # 3. 准备 Multipart 文件 (表单数据)
            files = {
                'file': ('violation.jpg', img_bytes, 'image/jpeg')
            }

            logger.info(f"正在上传报警 | IP:{nvr_ip} CH:{channel} AlgoCode:{algorithm_code}")

            # 4. 发送 POST 请求
            t_upload_start = time.time()
            resp = requests.post(
                API_UPLOAD_ALERT,
                params=query_params,
                files=files,
                timeout=10
            )
            t_upload_end = time.time()
            http_ms = (t_upload_end - t_upload_start) * 1000
            total_ms = queue_wait_ms + http_ms
            logger.info(f"报警上传完成 | 队列等待={queue_wait_ms:.1f}ms | HTTP={http_ms:.1f}ms | 总延迟={total_ms:.1f}ms | IP:{nvr_ip} CH:{channel}")

            # 5. 处理响应
            if resp.status_code == 200:
                resp_json = resp.json()
                if resp_json.get('code') == 0:
                    logger.info(f"报警上传成功 | IP:{nvr_ip} CH:{channel}")
                else:
                    logger.error(f"上传业务失败: {resp_json.get('msg')} | IP:{nvr_ip} CH:{channel}")
            else:
                logger.error(f"上传 HTTP 失败: {resp.status_code} - {resp.text}")

        except requests.exceptions.Timeout:
            logger.error("上传请求超时")
        except requests.exceptions.ConnectionError:
            logger.error("上传连接失败(网络不通)")
        except Exception as e:
            logger.error(f"上传过程异常: {e}", exc_info=True)

    def stop(self):
        self.stopped = True
        if self.t.is_alive():
            self.t.join(timeout=2)