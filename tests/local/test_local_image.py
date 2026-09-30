"""
本地图片违章检测测试脚本（配置驱动）

测试算法、图片、围栏、推理模式、是否启用分类器/VL 二次确认，均可通过
config/algorithms.yaml 的 test_local 段配置：

    test_local:
      default_image_dir: ./asserts/images
      enable_classifier: true   # 启用分类器二次确认
      enable_vl: true           # 启用 VL 大模型二次确认
      algorithms:
        - {code: '8', fence: '100#100,700#100,700#500,100#500'}
        - {code: '58'}

处理流水线（与生产一致，均由配置驱动）：
    推理(YOLO/SAM3) -> 规则分析(analyze_for_task)
    -> 分类器二次确认(classify_for_task, 按 algorithm_classifiers)
    -> VL 大模型二次确认(vl_analyze_for_task, 按 algorithm_vl_config)

支持三种推理模式：
  - yolo-sam3（默认）: YOLO 预检 + SAM3 精检，模拟生产流程
  - yolo-only        : 仅用 YOLO 快速检测
  - sam3-only        : 直接调用 SAM3，不走 YOLO 预检

用法示例：
    # 按配置文件测试全部算法（不指定 -a 时默认全部）
    python tests/local/test_local_image.py

    # 指定测试图片目录（未在配置中单独指定图片的算法都使用该目录）
    python tests/local/test_local_image.py -i ./asserts/images

    # 只测试单个算法（危险区域闯入，算法8），图片/围栏从配置读取
    python tests/local/test_local_image.py -a 8

    # 单个算法 + 手动指定图片
    python tests/local/test_local_image.py -i asserts/images/test.jpg -a 8

    # 只用 YOLO 快速检测（适合验证 YOLO 本身）
    python tests/local/test_local_image.py -i asserts/images/test.jpg -a 8 --mode yolo-only

    # 只用 SAM3
    python tests/local/test_local_image.py -i asserts/images/test.jpg -a 8 --mode sam3-only

    # 命令行临时指定围栏（优先级高于配置）
    python tests/local/test_local_image.py -i asserts/images/test.jpg -a 8 --fence "100#100,200#100,200#200,100#200"

    # YOLO + SAM3 + 大模型：算法58（安全带）默认即走 yolo->sam3->规则->VL 全流程
    python tests/local/test_local_image.py -a 58

    # 强制 VL 复核：规则引擎未命中违规时，也把全部检测框提交大模型复核
    python tests/local/test_local_image.py -a 58 --force-vl

    # 关闭 VL 大模型二次确认（只跑规则引擎）
    python tests/local/test_local_image.py -a 8 --no-vl

    # 关闭分类器二次确认
    python tests/local/test_local_image.py -a 14 --no-classifier

    # 关闭 Triton 共享内存（回退 HTTP）
    python tests/local/test_local_image.py -i asserts/images/test.jpg -a 8 --no-shm
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

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
from core.classifier import classify_for_task
from llm.vl_analyzer import vl_analyze_for_task
from utils.osd import render_alert_frame

logger = setup_logger("test_local")

# YOLO 客户端缓存（测试脚本内单例）
_yolo_client_cache = {}
_use_shm = True  # 是否对 Triton 使用共享内存
_use_classifier = True  # 是否启用分类器二次确认
_use_vl = True          # 是否启用 VL 大模型二次确认
_force_vl = False       # 规则引擎未命中违规时，也强制把检测框提交 VL 复核


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


def get_algo_name(algo_code: str) -> str:
    """从配置 code_descriptions 获取算法名称，未配置时回退到 '算法{code}'"""
    return config.CODE_DESCRIPTIONS.get(str(algo_code), f"算法{algo_code}")


def _resolve_project_path(path: str) -> str:
    """将相对路径解析为相对项目根目录的绝对路径"""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _resolve_image_for_algo(algo_code: str, cli_image: str, default_image_dir: str) -> str:
    """
    解析某算法码的测试图片（返回绝对路径）：
    test_local 配置 image > 命令行 -i > 默认图片目录
    """
    cfg_image = (config.TEST_LOCAL_ALGORITHMS.get(str(algo_code), {}) or {}).get("image") or ""
    if cfg_image:
        # 配置的 image 优先按项目根目录解析，其次按默认图片目录解析
        resolved = _resolve_project_path(cfg_image)
        if os.path.exists(resolved):
            return resolved
        return os.path.join(default_image_dir, cfg_image)
    if cli_image:
        return _resolve_project_path(cli_image)
    return default_image_dir


def _resolve_fence_for_algo(algo_code: str, cli_fence: str) -> Optional[List]:
    """解析围栏：命令行 --fence 优先，其次 test_local 配置，最后为空"""
    fence_str = (cli_fence
                 or (config.TEST_LOCAL_ALGORITHMS.get(str(algo_code), {}) or {}).get("fence")
                 or "")
    return parse_fence_string(fence_str) if fence_str else None


def _sort_codes_key(code: str):
    """算法码排序键：优先按数值排序，非纯数字时按字符串排序"""
    try:
        return (0, int(code))
    except ValueError:
        return (1, code)


def _sanitize_dir_name(name: str) -> str:
    """将算法名称清洗为安全的目录名（去掉 Windows 非法字符，避免目录创建失败）"""
    name = str(name or "unknown").strip()
    for ch in ('\\', '/', ':', '*', '?', '"', '<', '>', '|'):
        name = name.replace(ch, '_')
    return name or "unknown"


def _get_yolo_client(model_name: str):
    """获取或创建 YOLO 客户端（测试脚本内单例），自动从配置读取模型参数"""
    global _yolo_client_cache
    if model_name not in _yolo_client_cache:
        from detect.triton_client_fast import YOLOTritonFast
        model_cfg = config.YOLO_MODEL_CONFIGS.get(model_name, {})
        protocol = model_cfg.get("protocol", config.TRITON_PROTOCOL)
        if not _use_shm and protocol == "shm":
            protocol = "http"
        endpoint = config.triton_endpoint(protocol)
        client = YOLOTritonFast(
            url=endpoint,
            model_name=model_name,
            input_size=model_cfg.get("input_size", 640),
            conf_thresh=model_cfg.get("conf_thresh", 0.5),
            iou_thresh=model_cfg.get("iou_thresh", 0.45),
            label_map=model_cfg.get("label_map"),
            input_name=model_cfg.get("input_name", "IMAGE"),
            output_name=model_cfg.get("output_name", "output0"),
            output_format=model_cfg.get("output_format", "yolo_v8_v11"),
            use_shared_memory=_use_shm,
            protocol=protocol,
            backend=model_cfg.get("backend", "ascend"),
            max_detections=model_cfg.get("max_detections", 300),
            warmup=True
        )
        _yolo_client_cache[model_name] = client
        logger.info(f"YOLO 客户端初始化: {model_name} @ {endpoint} "
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
    """仅用 YOLO 检测（取该算法配置的第一个模型，无配置则用默认 NV12 ensemble）。"""
    pre_config = config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES.get(algo_code, {})
    if pre_config:
        model_name = list(pre_config.keys())[0]
    else:
        model_name = "YOLO26_DET_PRE_YUV_ENSEMBLE"

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
    return_mask = config.ALGORITHM_SAM3_RETURN_MASK.get(str(algo_code), False)
    # 未在 sam3_url_groups 分组的算法码回退到 config.SAM3_URL_OBJ
    sam3_url = config.ALGORITHM_SAM3_URL.get(str(algo_code)) or config.SAM3_URL_OBJ
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

    # 3. 分析（规则引擎）
    t2_start = time.time()
    fences = task.electricFence if task.electricFence else None
    image_width = frame.shape[1]
    image_height = frame.shape[0]
    violations = analyze_for_task(
        all_boxes, task, fences=fences,
        image_width=image_width, image_height=image_height, frame=frame
    )
    t2_end = time.time()
    logger.info(f"规则分析完成 | 耗时={t2_end-t2_start:.3f}s | 违规数={len(violations)}")

    # 3.1 分类器二次确认（如吸烟分类，按 algorithm_classifiers 配置）
    classifier_names = config.ALGORITHM_CLASSIFIERS.get(str(task.algorithmCode), [])
    if _use_classifier and classifier_names and violations:
        violations = classify_for_task(
            frame, task, violations,
            image_width=image_width, image_height=image_height,
        )
        logger.info(f"分类器二次确认 | classifiers={classifier_names} | 剩余违规数={len(violations)}")

    # 3.2 VL 大模型二次确认（按 algorithm_vl_config 配置，如 32 睡岗 / 58 安全带 / 205 安全）
    # 仅对启用了 VL 的算法生效；--force-vl 时即使规则引擎未命中违规，也会把全部检测框提交大模型复核
    vl_cfg = config.ALGORITHM_VL_CONFIG.get(str(task.algorithmCode), {})
    vl_configured = bool(vl_cfg.get("enabled", False))
    if _use_vl and vl_configured:
        if violations:
            vl_candidates = violations
        elif _force_vl and all_boxes:
            # 规则引擎未命中违规，但用户显式要求跑大模型：把所有检测框当作候选
            vl_candidates = list(all_boxes)
            logger.info(f"--force-vl | 规则引擎未命中违规，将全部检测框作为候选提交 VL 复核（{len(vl_candidates)} 个）")
        else:
            vl_candidates = []

        if vl_candidates:
            logger.info(f"VL 大模型二次确认 | module={vl_cfg.get('module')} | 输入候选数={len(vl_candidates)}")
            violations = vl_analyze_for_task(
                frame, task, vl_candidates,
                image_width=image_width, image_height=image_height,
            )
            logger.info(f"VL 二次确认后违规数={len(violations)}")

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

    # 5. 保存结果：有违章/无违章分目录保存
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    # 有违章 -> violation/，无违章 -> no_violation/
    sub_dir = "violation" if violations else "no_violation"
    save_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{name}_result{ext}")
    # 使用 cv2.imencode + tofile 以支持中文路径
    success, encoded = cv2.imencode(os.path.splitext(output_path)[1], osd_frame)
    if success:
        encoded.tofile(output_path)
        logger.info(f"结果已保存: {output_path} (违规数={len(violations)})")
    else:
        logger.error(f"保存结果失败: {output_path}")

    return True


def collect_images(image_source: str) -> List[str]:
    """收集要处理的图片列表（支持单个文件或目录）"""
    image_paths = []
    if os.path.isdir(image_source):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for f in sorted(os.listdir(image_source)):
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(image_source, f))
        logger.info(f"目录下找到 {len(image_paths)} 张图片: {image_source}")
    else:
        image_paths = [image_source]
    return image_paths


def run_algorithm(task, image_source: str, output_dir: str, mode: str) -> Tuple[int, int]:
    """对单个算法批量处理图片，返回 (成功数, 图片总数)"""
    image_paths = collect_images(image_source)
    if not image_paths:
        logger.warning(f"算法 {task.algorithmCode} 没有可处理的图片: {image_source}")
        return 0, 0
    success = 0
    for path in image_paths:
        if process_image(path, task, output_dir, mode):
            success += 1
    return success, len(image_paths)


def main():
    parser = argparse.ArgumentParser(description="本地图片违章检测测试脚本（配置驱动）")
    parser.add_argument("-i", "--image", default="",
                        help="输入图片路径或图片目录；未指定时按算法从 test_local 配置读取，否则用默认图片目录")
    parser.add_argument("-o", "--output", default="./test_results",
                        help="输出根目录；结果按算法名称分目录，再按是否有违章分 violation/no_violation 子目录 (默认: ./test_results)")
    parser.add_argument("-a", "--algo", default="",
                        help="算法代码；不指定时测试 config/algorithms.yaml 中配置支持的全部算法")
    parser.add_argument("-m", "--mode", default="yolo-sam3",
                        choices=["yolo-sam3", "yolo-only", "sam3-only"],
                        help="推理模式: yolo-sam3(预检+精检), yolo-only(仅YOLO), sam3-only(仅SAM3) (默认: yolo-sam3)")
    parser.add_argument("--fence", default="",
                        help="电子围栏坐标，格式: 'x1#y1,x2#y2,x3#y3'，多区域用 || 分隔；优先级高于配置文件")
    parser.add_argument("--no-classifier", action="store_true",
                        help="关闭分类器二次确认（默认按 test_local.enable_classifier）")
    parser.add_argument("--no-vl", action="store_true",
                        help="关闭 VL 大模型二次确认（默认按 test_local.enable_vl）")
    parser.add_argument("--force-vl", action="store_true",
                        help="强制 VL 复核：规则引擎未命中违规时，也把全部检测框提交大模型复核")
    parser.add_argument("--no-shm", action="store_true",
                        help="关闭 Triton 共享内存，使用 HTTP numpy 传输")
    args = parser.parse_args()

    global _use_shm, _use_classifier, _use_vl, _force_vl
    _use_shm = not args.no_shm
    _use_classifier = config.TEST_LOCAL_ENABLE_CLASSIFIER and not args.no_classifier
    _use_vl = config.TEST_LOCAL_ENABLE_VL and not args.no_vl
    _force_vl = args.force_vl

    os.makedirs(args.output, exist_ok=True)

    # 确定要测试的算法码列表：指定 -a 只测单个，否则按配置测全部
    if args.algo:
        algo_codes = [str(args.algo)]
    else:
        # supported_codes 与 test_local 中补充的算法码取并集
        algo_codes = sorted(
            set(config.ALGORITHM_CODES) | set(config.TEST_LOCAL_ALGORITHMS.keys()),
            key=_sort_codes_key,
        )

    default_image_dir = _resolve_project_path(config.TEST_LOCAL_DEFAULT_IMAGE_DIR)
    if not os.path.isdir(default_image_dir) and not args.image:
        logger.warning(f"默认图片目录不存在: {default_image_dir}，请用 -i 指定图片或修改 test_local.default_image_dir")

    # 汇总启用 VL 的算法码，便于确认大模型测试范围
    vl_enabled = {c: cfg for c, cfg in config.ALGORITHM_VL_CONFIG.items() if cfg.get("enabled")}
    logger.info(f"待测试算法: {algo_codes} | 默认模式: {args.mode} | shm={_use_shm} | 默认图片目录: {default_image_dir}")
    logger.info(f"分类器二次确认: {'开启' if _use_classifier else '关闭'} | "
                f"VL 大模型: {'开启' if _use_vl else '关闭'}"
                f"{'（--force-vl）' if _force_vl else ''}"
                f"{f' | 启用 VL 的算法: {vl_enabled}' if _use_vl and vl_enabled else ''}")

    total_success = 0
    total_images = 0
    start = time.time()

    for algo_code in algo_codes:
        algo_name = get_algo_name(algo_code)
        algo_test_cfg = config.TEST_LOCAL_ALGORITHMS.get(str(algo_code), {}) or {}
        image_source = _resolve_image_for_algo(algo_code, args.image, default_image_dir)
        fences = _resolve_fence_for_algo(algo_code, args.fence)
        algo_mode = algo_test_cfg.get("mode") or args.mode

        if str(algo_code) in config.FENCE_ALGORITHMS and not fences:
            logger.warning(f"算法 {algo_code}({algo_name}) 属于围栏类算法，但未配置围栏，区域判定将基于空围栏")

        task = MockTask(
            algorithm_code=algo_code,
            algorithm_name=algo_name,
            electric_fence=fences,
        )

        # 结果统一按算法名称分目录存储（单算法/多算法一致）
        output_dir = os.path.join(args.output, _sanitize_dir_name(algo_name))
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"==== 开始算法 {algo_code}({algo_name}) | 模式={algo_mode} | "
                    f"围栏={'有' if fences else '无'} | 图片={image_source} ====")
        success, total = run_algorithm(task, image_source, output_dir, algo_mode)
        total_success += success
        total_images += total

    elapsed = time.time() - start
    logger.info(f"完成 | 成功 {total_success}/{total_images} | 总耗时 {elapsed:.2f}s")

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
