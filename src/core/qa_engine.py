import torch
from typing import Dict, List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
from ..models.lora_adapter import load_lora_weights

class RAGQAEngine:
    def __init__(self, retriever: MultimodalRetriever, generator: LlavaForConditionalGeneration):
        self.retriever = retriever
        self.generator = generator
    
    def answer(self, question: str, context_images: List[Image]=None) -> str:
        # 1. 检索阶段
        retrieved = self.retriever.retrieve(
            question, 
            query_img=context_images[0] if context_images else None
        )
        
        # 2. 构建增强上下文
        context = "\n\n".join([
            f"参考文档 {i+1} (相关性: {doc['score']:.2f}):\n{doc['text']}"
            for i, doc in enumerate(retrieved)
        ])
        
        # 3. 动态Prompt构建
        prompt = f"""基于以下药品说明书片段：{context}
                    请专业地回答：
                    问：{question}
                    答：
                """
        
        # 4. 生成阶段
        inputs = self.generator.processor(
            text=prompt,
            images=context_images,
            return_tensors="pt"
        ).to(self.generator.device)
        
        outputs = self.generator.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)