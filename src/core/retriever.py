import os
import faiss
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from ..utils.text_processing import normalize_embeddings

class MultimodalRetriever:
    def __init__(self, config):
        # 初始化双编码器
        self.text_encoder = SentenceTransformer(config['text_encoder'])
        self.visual_encoder = CLIPModel.from_pretrained(config['clip'])
        
        # 多模态索引
        self.index = faiss.IndexFlatIP(768)  # 文本+视觉联合维度
        self.doc_store = {}

    def add_to_index(self, documents: List[Dict]):
        """构建多模态索引"""
        for doc in documents:
            # 文本嵌入
            text_emb = self.text_encoder.encode(doc["structured_text"])
            
            # 视觉嵌入
            img = Image.open(doc["image_path"])
            visual_emb = self.visual_encoder.get_image_features(img)
            
            # 联合嵌入 (加权拼接)
            joint_emb = np.concatenate([
                config['text_weight'] * text_emb,
                config['visual_weight'] * visual_emb
            ])
            
            self.index.add(joint_emb)
            self.doc_store[len(self.doc_store)] = doc

    def retrieve(self, query: str, query_img: Optional[Image]=None, top_k=3) -> List[Dict]:
        """检索最相关的文档片段"""
        # 文本查询嵌入
        query_text_emb = self.text_encoder.encode(query)
        
        # 视觉查询嵌入（如果提供图片）
        query_visual_emb = self.visual_encoder.get_image_features(query_img) if query_img else np.zeros_like(text_emb)
        
        # 联合查询向量
        query_joint_emb = np.concatenate([
            config['text_weight'] * query_text_emb,
            config['visual_weight'] * query_visual_emb
        ])
        
        # FAISS搜索
        distances, indices = self.index.search(query_joint_emb, top_k)
        
        return [{
            "score": float(distances[0][i]),
            "text": self.doc_store[idx]["structured_text"],
            "image_path": self.doc_store[idx]["image_path"]
        } for i, idx in enumerate(indices[0])]