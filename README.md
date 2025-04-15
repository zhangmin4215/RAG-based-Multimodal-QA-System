# 基于RAG的图文内容问答系统

本项目是一个基于检索增强生成(RAG)技术的图文内容问答系统，结合了OCR、CLIP视觉编码和大模型微调技术。

## 功能特性

- 多模态文档处理(图片+文本)
- 混合检索(文本+视觉)
- 基于LLaVA的问答引擎
- LoRA高效微调大模型
- 药品说明书结构化处理

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt```

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
