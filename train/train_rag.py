import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    get_linear_schedule_with_warmup,
    AdamW
)
from src.core.retriever import MultimodalRetriever
from src.data.dataset import MedicalQADataset
import faiss
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

class RAGTrainer:
    def __init__(self, config_path: str = "configs/rag_config.yaml"):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # 初始化组件
        self.retriever = self._init_retriever()
        self.generator, self.processor = self._init_generator()
        self.optimizer = self._init_optimizer()
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)

    def _init_retriever(self):
        """初始化多模态检索器"""
        retriever = MultimodalRetriever(self.config['retriever'])
        
        # 构建索引
        if not Path(self.config['data']['index_path']).exists():
            documents = self._load_documents()
            retriever.build_index(documents)
            retriever.save_index(self.config['data']['index_path'])
        else:
            retriever.load_index(self.config['data']['index_path'])
            
        return retriever

    def _init_generator(self):
        """初始化生成模型"""
        processor = LlavaProcessor.from_pretrained(self.config['generator']['model_name'])
        model = LlavaForConditionalGeneration.from_pretrained(
            self.config['generator']['model_name'],
            torch_dtype=torch.float16 if self.config['training']['fp16'] else torch.float32,
            device_map="auto"
        )
        
        # 应用LoRA
        if self.config['generator'].get('lora'):
            from peft import get_peft_model
            peft_config = LoraConfig(
                r=self.config['generator']['lora']['r'],
                lora_alpha=self.config['generator']['lora']['alpha'],
                target_modules=self.config['generator']['lora']['target_modules'],
                modules_to_save=["embed_tokens", "lm_head"],
                lora_dropout=self.config['generator']['lora']['dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        return model, processor

    def _init_optimizer(self):
        """初始化优化器"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.generator.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config['training']['weight_decay'],
            },
            {
                "params": [p for n, p in self.generator.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            eps=self.config['training']['adam_epsilon']
        )

    def _load_documents(self):
        """加载文档数据用于构建索引"""
        docs = []
        for img_path in Path(self.config['data']['documents_dir']).glob("*.jpg"):
            with open(f"{self.config['data']['ocr_gt_dir']}/{img_path.stem}.json") as f:
                ocr_data = json.load(f)
            docs.append({
                "image_path": str(img_path),
                "structured_text": ocr_data,
                "visual_embedding": None  # 将在build_index时生成
            })
        return docs

    def _make_rag_prompt(self, question: str, contexts: list) -> str:
        """构建RAG提示模板"""
        context_str = "\n".join([
            f"[参考 {i+1}] {ctx['text']}\n(相关性: {ctx['score']:.2f})"
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""基于以下药品说明书片段：
{context_str}

请以专业药师的身份回答：
问题：{question}
答案："""

    def train(self):
        # 准备数据集
        train_dataset = MedicalQADataset(
            image_dir=self.config['data']['images_dir'],
            ocr_dir=self.config['data']['ocr_gt_dir'],
            qa_path=self.config['data']['train_qa_path'],
            processor=self.processor
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        # 训练循环
        global_step = 0
        for epoch in range(self.config['training']['epochs']):
            self.generator.train()
            epoch_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                # 1. 检索增强
                contexts = []
                for q, img in zip(batch["questions"], batch["images"]):
                    img = Image.open(img) if isinstance(img, str) else img
                    contexts.append(self.retriever.retrieve(q, img))
                
                # 2. 构建输入
                inputs = self.processor(
                    text=[self._make_rag_prompt(q, ctx) for q, ctx in zip(batch["questions"], contexts)],
                    images=[Image.open(img) if isinstance(img, str) else img for img in batch["images"]],
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.config['generator']['max_input_length'],
                    truncation=True
                ).to(self.device)
                
                # 3. 准备标签
                labels = self.processor(
                    text=batch["answers"],
                    return_tensors="pt",
                    padding="longest",
                    max_length=self.config['generator']['max_output_length'],
                    truncation=True
                ).input_ids.to(self.device)
                
                # 4. 前向传播
                outputs = self.generator(
                    **inputs,
                    labels=labels,
                    output_retrieved=True
                )
                
                # 5. 反向传播
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 记录指标
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                global_step += 1
            
            # 保存检查点
            if (epoch + 1) % self.config['training']['save_epochs'] == 0:
                self._save_checkpoint(epoch)
    
    def _collate_fn(self, batch):
        """自定义批处理函数"""
        return {
            "images": [item["image"] for item in batch],
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch]
        }
    
    def _save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint_dir = Path(self.config['training']['output_dir']) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存生成器
        self.generator.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_dir / "training_state.pt")

if __name__ == "__main__":
    trainer = RAGTrainer()
    trainer.train()