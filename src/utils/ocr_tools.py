from typing import List, Dict, Tuple
import numpy as np

def postprocess_ocr_results(ocr_result: List) -> Dict:
    """后处理OCR结果"""
    if not ocr_result or not ocr_result[0]:
        return {"text": "", "boxes": [], "confidences": []}
    
    texts = []
    boxes = []
    confidences = []
    
    for res in ocr_result[0]:
        text = res[1][0]
        confidence = res[1][1]
        box = np.array(res[0]).tolist()
        
        texts.append(text)
        boxes.append(box)
        confidences.append(confidence)
    
    return {
        "text": " ".join(texts),
        "text_blocks": texts,
        "boxes": boxes,
        "confidences": confidences
    }

def calculate_iou(box1, box2):
    """计算两个框的IOU"""
    # 实现IOU计算逻辑
    pass
