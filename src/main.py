"""
AI 视频安监检测系统入口

核心模块拆分：
  - core/analyzer.py      : 分析函数（算法分发、prompt 合并）
  - core/stream_worker.py : 按 RTSP 流的视频拉取与推理线程
  - core/orchestrator.py  : 任务生命周期管理与 worker 调度
"""

from utils.logger import setup_logger
from core.orchestrator import Orchestrator

logger = setup_logger("main")


def main():
    orchestrator = Orchestrator()

    # 在主线程中运行编排器
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
