import re
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer

def structure_text(ocr_text: Dict) -> Dict:
    """结构化OCR文本"""
    combined_text = ocr_text.get("text", "")
    text_blocks = ocr_text.get("text_blocks", [])
    
    # 简单段落分组逻辑
    paragraphs = []
    current_para = []
    
    for text in text_blocks:
        if text.endswith(('。', '！', '？')):
            current_para.append(text)
            paragraphs.append(" ".join(current_para))
            current_para = []
        else:
            current_para.append(text)
    
    if current_para:
        paragraphs.append(" ".join(current_para))
    
    # 提取键值对 (适用于药品说明书)
    key_values = {}
    for para in paragraphs:
        if ":" in para or "：" in para:
            parts = re.split(r":|：", para, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                key_values[key] = value
    
    return {
        "combined_text": combined_text,
        "paragraphs": paragraphs,
        "key_values": key_values
    }

def normalize_embeddings(embeddings) -> np.ndarray:
    """归一化嵌入向量"""
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)