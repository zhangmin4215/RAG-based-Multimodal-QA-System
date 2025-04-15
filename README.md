# 多模态RAG图文问答系统

基于RAG技术的图文内容问答系统，整合了CLIP视觉编码、OCR文本识别、LLaVA多模态大模型和FAISS向量检索。

## 功能特性

- CLIP+OCR混合检索方案
- LLaVA-1.5微调模型问答
- TensorRT加速OCR处理
- 结构化文本对齐
- 现代化Web界面

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/rag-multimodal-qa.git
cd rag-multimodal-qa

# 安装依赖
pip install -r requirements.txt

# 下载模型权重 (需要手动下载)
# 放入 models/ 目录

# 启动服务
uvicorn app:app --reload
