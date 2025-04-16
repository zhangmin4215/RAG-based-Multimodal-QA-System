import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Optional
import numpy as np

class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode_image(self, image) -> np.ndarray:
        """编码图片为嵌入向量"""
        inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy().squeeze()
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本为嵌入向量"""
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features.cpu().numpy().squeeze()
