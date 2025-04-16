import os
import yaml
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from transformers import CLIPProcessor, CLIPModel
from ..utils.ocr_tools import postprocess_ocr_results
from ..utils.text_processing import structure_text

class DocumentProcessor:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 初始化OCR引擎
        self.ocr_engine = PaddleOCR(
            lang=self.config['ocr']['lang'],
            use_angle_cls=self.config['ocr']['use_angle_cls'],
            det_model_dir=self.config['ocr'].get('det_model_dir'),
            rec_model_dir=self.config['ocr'].get('rec_model_dir')
        )
        
        # 初始化CLIP模型
        self.clip_model = CLIPModel.from_pretrained(self.config['clip']['model_name'])
        self.clip_processor = CLIPProcessor.from_pretrained(self.config['clip']['model_name'])
        self.clip_device = self.config['clip']['device']
        self.clip_model.to(self.clip_device)
    
    def process_document(self, doc_path: str) -> Dict:
        """处理单个文档，返回结构化数据"""
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        if doc_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return self._process_image(doc_path)
        else:
            raise ValueError(f"Unsupported document format: {doc_path}")
    
    def _process_image(self, image_path: str) -> Dict:
        """处理图片文档"""
        # 1. 使用OCR提取文本
        ocr_result = self.ocr_engine.ocr(image_path, cls=True)
        ocr_text = postprocess_ocr_results(ocr_result)
        
        # 2. 提取视觉特征
        image = Image.open(image_path)
        visual_embedding = self._get_image_embedding(image)
        
        # 3. 结构化文本
        structured_text = structure_text(ocr_text)
        
        return {
            "original_path": image_path,
            "ocr_raw": ocr_result,
            "ocr_text": ocr_text,
            "visual_embedding": visual_embedding,
            "structured_text": structured_text
        }
    
    def _get_image_embedding(self, image: Image) -> np.ndarray:
        """获取图片的CLIP嵌入"""
        inputs = self.clip_processor(
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.clip_device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().squeeze()
