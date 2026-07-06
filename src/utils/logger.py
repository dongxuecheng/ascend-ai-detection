import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ==================== 环境变量配置 ====================
# AIDETECTION_LOG_LEVEL      : 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)，默认 INFO
# AIDETECTION_LOG_FILE       : 日志文件路径，默认 logs/app.log
# AIDETECTION_LOG_DISABLE_CONSOLE : 是否禁用控制台输出 (1/yes/true)，默认启用
# AIDETECTION_LOG_DISABLE_FILE    : 是否禁用文件输出 (1/yes/true)，默认启用


def _env_bool(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔值"""
    val = os.getenv(name, "").lower().strip()
    return val in ("1", "yes", "true", "on") if val else default


def _env_level(name: str, default: int = logging.INFO) -> int:
    """从环境变量读取日志级别"""
    val = os.getenv(name, "").upper().strip()
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(val, default)


# 全局默认配置
DEFAULT_LOG_LEVEL = _env_level("AIDETECTION_LOG_LEVEL", logging.INFO)
DEFAULT_LOG_FILE = os.getenv("AIDETECTION_LOG_FILE", os.path.join("logs", "app.log"))
DISABLE_CONSOLE = _env_bool("AIDETECTION_LOG_DISABLE_CONSOLE")
DISABLE_FILE = _env_bool("AIDETECTION_LOG_DISABLE_FILE")


def setup_logger(name: str = "AIDetection", level: int = None, log_file: str = None) -> logging.Logger:
    """
    配置项目日志：支持环境变量控制级别、文件路径、输出目标

    环境变量：
      - AIDETECTION_LOG_LEVEL=[DEBUG|INFO|WARNING|ERROR|CRITICAL]
      - AIDETECTION_LOG_FILE=/path/to/app.log
      - AIDETECTION_LOG_DISABLE_CONSOLE=1    (禁用控制台)
      - AIDETECTION_LOG_DISABLE_FILE=1       (禁用文件)

    参数优先级：传参 > 环境变量 > 默认值
    """
    logger = logging.getLogger(name)

    # 确定日志级别
    effective_level = level if level is not None else DEFAULT_LOG_LEVEL
    logger.setLevel(effective_level)

    # 如果已配置且级别一致，直接返回（避免重复添加 handler）
    if logger.handlers and logger.level == effective_level:
        return logger

    # 清除旧 handler（级别变化时重新配置）
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    if not DISABLE_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出（带轮转，单个文件 10MB，保留 5 个备份）
    if not DISABLE_FILE:
        file_path = log_file if log_file is not None else DEFAULT_LOG_FILE
        # 确保日志目录存在
        log_dir = os.path.dirname(file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError:
                pass
        try:
            file_handler = RotatingFileHandler(
                file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            file_handler.setLevel(effective_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # 文件写入失败时至少保证控制台能输出
            logger.warning(f"日志文件初始化失败: {e}")

    return logger
