"""
本地图片违章检测测试脚本

支持三种推理模式：
  - yolo-sam3（默认）: YOLO 预检 + SAM3 精检，模拟生产流程
  - yolo-only        : 仅用 YOLO 快速检测
  - sam3-only        : 直接调用 SAM3，不走 YOLO 预检

用法示例：
    # YOLO预检+SAM3，危险区域闯入（算法8）
    python tests/local/test_local_image.py -i assets/images/test.jpg -a 8

    # 整个目录，单人作业（算法34）
    python tests/local/test_local_image.py -i ./assets/images/ -a 34 -o ./test_results/

    # 只用 YOLO 快速检测（适合验证 YOLO 本身）
    python tests/local/test_local_image.py -i assets/images/test.jpg -a 8 --mode yolo-only

    # 只用 SAM3
    python tests/local/test_local_image.py -i assets/images/test.jpg -a 8 --mode sam3-only

    # 电子围栏，指定围栏坐标（算法8）
    python tests/local/test_local_image.py -i assets/images/test.jpg -a 8 --fence "100#100,200#100,200#200,100#200"

    # 关闭 Triton 共享内存（回退 HTTP）
    python tests/local/test_local_image.py -i assets/images/test.jpg -a 8 --no-shm
"""

import argparse
import os
import sys
import time
from typing import List, Optional

# 把 src/ 加入 sys.path，确保能 import config/ detect/ utils/ 等模块
# 当前文件位于 tests/local/，向上回退两级才是项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
# stream/ 下的 gRPC 生成代码使用绝对导入，需把 stream/ 目录也加入路径
sys.path.insert(0, os.path.join(SRC_DIR, "stream"))

import cv2
import numpy as np

from config.config import config
from utils.logger import setup_logger
from utils.obj import Box
from core.analyzer import analyze_for_task
from utils.osd import render_alert_frame

logger = setup_logger("test_local")

# YOLO 客户端缓存（测试脚本内单例）
_yolo_client_cache = {}
_use_shm = True  # 是否对 Triton 使用共享内存


class MockTask:
    """本地测试用的模拟 Task 对象"""
    def __init__(self, algorithm_code: str, algorithm_name: str = "",
                 electric_fence=None):
        self.id = "test"
        self.algorithmCode = algorithm_code
        self.algorithmName = algorithm_name
        self.deviceId = ""
        self.deviceAlgorithmIp = ""
        self.deviceChannel = ""
        self.electricFence = electric_fence
        self.alarmTaskName = f"测试任务-{algorithm_name or algorithm_code}"
        self.beforeDrawPhoto = None
        self.personLimitNum = None


def parse_fence_string(fence_str: str):
    """
    解析围栏坐标字符串为 zone.py 需要的格式
    格式: "x1#y1,x2#y2,x3#y3||x1#y1,x2#y2" （支持多区域，用 || 分隔）
    """
    if not fence_str:
        return None
    fences = []
    for region_str in fence_str.split("||"):
        pts = []
        for item in region_str.split(","):
            item = item.strip()
            if not item or "#" not in item:
                continue
            try:
                x, y = item.split("#")
                pts.append([int(x), int(y)])
            except ValueError:
                continue
        if pts:
            fences.append(pts)
    return fences if fences else None


def _get_yolo_client(model_name: str):
    """获取或创建 YOLO 客户端（测试脚本内单例），自动从配置读取模型参数"""
    global _yolo_client_cache
    if model_name not in _yolo_client_cache:
        from detect.triton_client_fast import YOLOTritonFast
        model_cfg = config.YOLO_MODEL_CONFIGS.get(model_name, {})
        client = YOLOTritonFast(
            url=config.TRITON_YOLO_URL,
            model_name=model_name,
            input_size=model_cfg.get("input_size", 640),
            conf_thresh=model_cfg.get("conf_thresh", 0.5),
            iou_thresh=model_cfg.get("iou_thresh", 0.45),
            label_map=model_cfg.get("label_map"),
            input_name=model_cfg.get("input_name", "images"),
            output_name=model_cfg.get("output_name", "output0"),
            output_format=model_cfg.get("output_format", "yolo_v8_v11"),
            use_shared_memory=_use_shm,
            warmup=True
        )
        _yolo_client_cache[model_name] = client
        logger.info(f"YOLO 客户端初始化: {model_name} @ {config.TRITON_YOLO_URL} "
                    f"(classes={client.num_classes}, input_size={client.input_size}, "
                    f"conf={client.conf_thresh}, iou={client.iou_thresh}, shm={_use_shm})")
    return _yolo_client_cache[model_name]


def yolo_pre_detect(frame: np.ndarray, algo_code: str) -> Optional[List[Box]]:
    """
    YOLO 预检测，模拟生产环境的 stream_worker._pre_detect_with_yolo 逻辑。

    :return:
        None  -> 未配置预检测或客户端异常，应放行到 SAM3
        []    -> 预检测未通过（无目标），应跳过 SAM3
        [Box] -> 预检测通过，同时返回检测框供日志参考
    """
    pre_config = config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES.get(algo_code, {})
    if not pre_config:
        return None  # 未配置预检测，直接放行

    for model_name, required_classes in pre_config.items():
        try:
            client = _get_yolo_client(model_name)
            # 尝试将类别名映射为类别 ID，类似生产环境逻辑
            class_ids = []
            unknown_classes = []
            for name in required_classes:
                found_ids = [cid for cid, cname in client.label_map.items() if cname == name]
                if found_ids:
                    class_ids.extend(found_ids)
                else:
                    unknown_classes.append(name)
            if unknown_classes:
                logger.warning(f"YOLO 类别映射未找到: {unknown_classes}，回退到全类别检测")
                class_ids = None

            boxes = client.predict(frame, classes=class_ids)
        except Exception as e:
            logger.warning(f"YOLO 预检测异常 {model_name}: {e}，放行到 SAM3")
            return None

        detected_labels = {b.label for b in boxes}
        has_target = any(cls in detected_labels for cls in required_classes)

        if has_target:
            matched = [cls for cls in required_classes if cls in detected_labels]
            logger.info(f"YOLO 预检测通过 | 模型={model_name}, 匹配={matched}, 框数={len(boxes)}")
            return boxes
        else:
            logger.info(f"YOLO 预检测未通过 | 模型={model_name}, 需{required_classes}, 实{detected_labels}")
            return []

    return None


def detect_yolo_only(frame: np.ndarray, algo_code: str) -> List[Box]:
    """仅用 YOLO 检测（取该算法配置的第一个模型，无配置则用默认 yolo11_plan）"""
    pre_config = config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES.get(algo_code, {})
    if pre_config:
        model_name = list(pre_config.keys())[0]
    else:
        model_name = "yolo11_plan"

    try:
        client = _get_yolo_client(model_name)
        return client.predict(frame)
    except Exception as e:
        logger.error(f"YOLO 检测失败: {e}")
        return []


def detect_sam3_only(frame: np.ndarray, algo_code: str) -> List[Box]:
    """直接调用 SAM3 推理"""
    from detect.sam3 import call_sam3
    prompts = config.ALGORITHM_SAM3_PROMPT.get(str(algo_code), ["person"])
    print(prompts)
    return_mask = config.ALGORITHM_SAM3_RETURN_MASK.get(str(algo_code), False)
    sam3_url = config.ALGORITHM_SAM3_URL.get(str(algo_code), "http://192.168.100.75:18002/predict")
    return call_sam3(frame, prompts, return_mask=return_mask, url=sam3_url)


def detect_yolo_then_sam3(frame: np.ndarray, algo_code: str) -> List[Box]:
    """YOLO 预检 -> SAM3 精检（模拟生产流程，支持 YOLO 框合并）"""
    pre_result = yolo_pre_detect(frame, algo_code)

    if pre_result == []:
        # 预检测明确未通过，返回空列表（不调用 SAM3）
        return []

    # 预检测通过或未配置，调用 SAM3
    all_boxes = detect_sam3_only(frame, algo_code)

    # 模拟生产环境的 use_yolo_boxes 合并逻辑
    if str(algo_code) in config.USE_YOLO_BOXES and pre_result:
        sam_labels = {b.label for b in all_boxes}
        merged = [yb for yb in pre_result if yb.label not in sam_labels]
        if merged:
            all_boxes.extend(merged)
            logger.info(f"YOLO 框合并 | 补充类别: {[b.label for b in merged]}, 合并后总框数={len(all_boxes)}")

    return all_boxes


def process_image(image_path: str, task, output_dir: str, mode: str):
    """处理单张图片：推理 -> 分析 -> 绘制 OSD -> 保存结果"""
    # 使用 np.fromfile + cv2.imdecode 以支持中文路径
    img_bytes = np.fromfile(image_path, dtype=np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error(f"无法读取图片: {image_path}")
        return False

    logger.info(f"处理图片: {image_path} ({frame.shape[1]}x{frame.shape[0]}) | 模式={mode}")

    # 1. 推理
    t0 = time.time()
    if mode == "yolo-only":
        all_boxes = detect_yolo_only(frame, str(task.algorithmCode))
    elif mode == "sam3-only":
        all_boxes = detect_sam3_only(frame, str(task.algorithmCode))
    else:  # yolo-sam3（默认）
        all_boxes = detect_yolo_then_sam3(frame, str(task.algorithmCode))
    t1 = time.time()

    # 2. 统计类别
    label_counts = {}
    for b in all_boxes:
        label_counts[b.label] = label_counts.get(b.label, 0) + 1
    logger.info(f"推理完成 | 耗时={t1-t0:.3f}s | 总框数={len(all_boxes)} | 类别={label_counts}")

    # 3. 分析
    t2_start = time.time()
    fences = task.electricFence if task.electricFence else None
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    violations = analyze_for_task(
        all_boxes, task, fences=fences,
        image_width=image_width, image_height=image_height, frame=frame
    )
    t2_end = time.time()
    logger.info(f"分析完成 | 耗时={t2_end-t2_start:.3f}s | 违规数={len(violations)}")

    # 4. 绘制 OSD
    if violations:
        logger.warning(f"⚠️ 检测到违规! 算法={task.algorithmCode}, 违规目标={len(violations)}")

    osd_frame = render_alert_frame(
        frame=frame,
        all_boxes=all_boxes,
        violations=violations,
        task=task,
        fences=fences,
        draw_all_boxes=True,
        draw_violation_boxes=True
    )

    # 5. 保存结果
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, f"{name}_result{ext}")
    # 使用 cv2.imencode + tofile 以支持中文路径
    success, encoded = cv2.imencode(os.path.splitext(output_path)[1], osd_frame)
    if success:
        encoded.tofile(output_path)
        logger.info(f"结果已保存: {output_path}")
    else:
        logger.error(f"保存结果失败: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="本地图片违章检测测试脚本")
    parser.add_argument("-i", "--image", required=True,
                        help="输入图片路径或图片目录")
    parser.add_argument("-o", "--output", default="./test_results",
                        help="输出结果目录 (默认: ./test_results)")
    parser.add_argument("-a", "--algo", default="200",
                        help="算法代码: 200=手套/面罩, 201=安全帽/皮肤, 204=电子围栏 (默认: 200)")
    parser.add_argument("-m", "--mode", default="yolo-sam3",
                        choices=["yolo-sam3", "yolo-only", "sam3-only"],
                        help="推理模式: yolo-sam3(预检+精检), yolo-only(仅YOLO), sam3-only(仅SAM3) (默认: yolo-sam3)")
    parser.add_argument("--fence", default="",
                        help="电子围栏坐标，格式: 'x1#y1,x2#y2,x3#y3'，多区域用 || 分隔")
    parser.add_argument("--no-shm", action="store_true",
                        help="关闭 Triton 共享内存，使用 HTTP numpy 传输")
    args = parser.parse_args()

    global _use_shm
    _use_shm = not args.no_shm

    os.makedirs(args.output, exist_ok=True)

    algo_names = {"200": "手套/面罩检测", "201": "安全帽/皮肤检测", "204": "电子围栏"}
    algo_name = algo_names.get(args.algo, f"算法{args.algo}")
    fence_data = parse_fence_string(args.fence) if args.fence else None

    task = MockTask(
        algorithm_code=args.algo,
        algorithm_name=algo_name,
        electric_fence=fence_data
    )

    logger.info(f"算法: {algo_name} ({args.algo}) | 模式: {args.mode} | shm={_use_shm} | 输入: {args.image}")

    # 收集图片列表
    image_paths = []
    if os.path.isdir(args.image):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for f in sorted(os.listdir(args.image)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(args.image, f))
        logger.info(f"目录下找到 {len(image_paths)} 张图片")
    else:
        image_paths = [args.image]

    if not image_paths:
        logger.error("没有找到可处理的图片")
        sys.exit(1)

    success = 0
    start = time.time()
    for path in image_paths:
        if process_image(path, task, args.output, args.mode):
            success += 1

    elapsed = time.time() - start
    logger.info(f"完成 | 成功 {success}/{len(image_paths)} | 总耗时 {elapsed:.2f}s")

    # 主动关闭 YOLO 客户端，释放 Triton 共享内存，避免解释器关闭阶段 segfault
    global _yolo_client_cache
    for client in list(_yolo_client_cache.values()):
        try:
            client.close()
        except Exception as e:
            logger.debug(f"关闭 YOLO 客户端异常: {e}")
    _yolo_client_cache.clear()


if __name__ == "__main__":
    main()
