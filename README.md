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
git clone https://github.com/zhangmin4215/RAG-based-Multimodal-QA-System.git
cd RAG-based-Multimodal-QA-System

# 安装依赖
pip install -r requirements.txt

# 下载模型权重 (需要手动下载)
# 放入 models/ 目录

# 启动服务
uvicorn app:app --reload

RAG-based-Multimodal-QA-System/
├── README.md
├── requirements.txt
├── Dockerfile
├── app.py
├── config.py
├── models/
│   ├── clip_encoder.py
│   ├── llava_integration.py
│   ├── text_aligner.py
│   └── tensorrt_optimizer.py
├── services/
│   ├── ocr_service.py
│   ├── retrieval_service.py
│   └── qa_service.py
├── utils/
│   ├── faiss_manager.py
│   ├── prompt_templates.py
│   └── preprocessing.py
├── data/
│   ├── sample_docs/  # 示例文档库
│   └── test_images/   # 测试图片
├── static/  # 网页静态文件
└── tests/   # 单元测试
