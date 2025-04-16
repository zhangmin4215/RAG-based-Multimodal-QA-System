from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Optional
import torch

class LlavaWrapper:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device_map: str = "auto", torch_dtype: str = "auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=self._get_torch_dtype(torch_dtype)
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def _get_torch_dtype(self, dtype_str: str):
        """将字符串转换为torch.dtype"""
        if dtype_str == "auto":
            return None
        return getattr(torch, dtype_str)
    
    def generate(self, images, text: str, **generation_kwargs) -> str:
        """生成回答"""
        inputs = self.processor(text, images, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            **generation_kwargs
        )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)