import cv2
import numpy as np
from typing import Dict

def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    使用 Numpy 向量化加速 RLE 编码。
    """
    if mask is None: return None
    
    # 扁平化
    flat = mask.ravel(order='F')
    flat = (flat > 0).astype(np.int8)
        
    if len(flat) == 0:
        return {'size': list(mask.shape), 'counts': []}
        
    diffs = np.where(flat[1:] != flat[:-1])[0] + 1
    bounds = np.concatenate(([0], diffs, [len(flat)]))
    counts = np.diff(bounds)
    counts_list = counts.tolist()
            
    if flat[0] == 1:
        counts_list = [0] + counts_list

    return {'size': list(mask.shape), 'counts': counts_list}


def rle_to_binary_mask(rle_mask : Dict) -> np.ndarray:
    """
    解码COCO RLE格式的mask,不依赖pycocotools。

    Args:
        rle_mask (dict):
            一个包含 'size' 和 'counts' 的字典。
            例如: {'size': [height, width], 'counts': RLE_array}

    Returns:
        np.ndarray:
            一个二维的二值NumPy数组 (height, width)，其中1表示掩码区域，0表示背景。
    """
    height, width = rle_mask['size']
    counts = rle_mask['counts']

    # 创建一个一维数组，总长度为图像像素数
    mask_1d = np.zeros(height * width, dtype=np.uint8)

    # `counts` 数组交替表示背景(0)和掩码(1)的行程长度。
    # 值为1的行程段总是在 `counts` 数组的奇数位索引上。
    current_pos = 0
    val = 0  # 初始值为0 (背景)
    for count in counts:
        # 当 val=0 时，这是背景像素段；当 val=1 时，这是掩码像素段。
        if val == 1:
            # 将一维数组的相应区域设置为1
            end_pos = current_pos + count
            mask_1d[current_pos:end_pos] = 1

        current_pos += count
        val = 1 - val  # 交替值 (0 -> 1, 1 -> 0)

    # 将一维数组变形为二维。
    # COCO RLE是列优先 (Fortran-style) 编码，所以必须设置 order='F'。
    mask_2d = mask_1d.reshape((height, width), order='F')

    return mask_2d
