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
├── configs/                  # 配置文件
│   ├── model_config.yaml
│   └── system_config.yaml
├── data/                     # 示例数据
│   ├── images/
│   └── documents/
├── docs/                     # 项目文档
├── scripts/                  # 实用脚本
├── src/
│   ├── core/                 # 核心功能
│   │   ├── __init__.py
│   │   ├── document_processor.py  # 文档处理
│   │   ├── retriever.py           # 检索系统
│   │   └── qa_engine.py           # 问答引擎
│   ├── models/               # 模型相关
│   │   ├── clip_encoder.py
│   │   ├── llava_wrapper.py
│   │   └── lora_adapter.py
│   ├── utils/                # 工具函数
│   │   ├── ocr_tools.py
│   │   └── text_processing.py
│   ├── api/                  # API服务
│   │   ├── app.py
│   │   └── schemas.py
│   └── web/                  # Web界面
│       └── app.py
├── tests/                    # 单元测试
├── train/                    # 训练脚本
│   └── finetune_llava.py
├── requirements.txt          # Python依赖
├── README.md                 # 项目说明
└── LICENSE
