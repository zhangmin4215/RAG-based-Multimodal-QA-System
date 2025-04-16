from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel
from typing import Dict, Optional
import torch

def load_lora_weights(model: PreTrainedModel, lora_weights_path: str) -> PreTrainedModel:
    """加载LoRA权重到模型"""
    model = PeftModel.from_pretrained(model, lora_weights_path)
    return model.merge_and_unload()

def prepare_lora_model(model: PreTrainedModel, config: Dict) -> PreTrainedModel:
    """准备LoRA模型"""
    lora_config = LoraConfig(
        r=config['r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['target_modules'],
        lora_dropout=config['lora_dropout'],
        bias=config['bias'],
        task_type=config['task_type']
    )
    
    return get_peft_model(model, lora_config)