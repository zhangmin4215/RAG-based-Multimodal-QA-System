# 基于RAG的多模态图文问答系统

## 📌 项目概述

本系统是一个基于检索增强生成(RAG)技术的多模态问答系统，能够处理图文混合内容并回答用户问题。系统结合了OCR文本提取、CLIP视觉编码和大语言模型，特别针对药品说明书等结构化文档进行了优化。

**核心功能**：
- 多模态文档处理（图像+文本）
- 混合特征检索（文本语义+视觉特征）
- 动态Prompt优化的问答生成
- 支持LoRA高效微调大模型

## 🚀 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/rag-multimodal-qa.git
cd RAG-based-Multimodal-QA-System

# 创建conda环境（可选）
conda create -n rag_qa python=3.9
conda activate rag_qa

# 安装依赖
pip install -r requirements.txt
```

### 数据准备

```bash
将需要处理的文档图片放入data/documents/目录，支持JPG/PNG格式。
```

### 启动服务
```bash
# 启动后端API服务
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 启动前端Web界面（新终端）
streamlit run src/web/app.py

访问 http://localhost:8501 使用Web界面
```

## 🧩 扩展功能
### 模型微调
```bash
python train/finetune_llava.py \
  --train_data data/train.json \
  --eval_data data/eval.json \
  --output_dir models/finetuned
```

### API调用示例
```bash
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"text": "这个药的副作用有哪些？"}
)
print(response.json())


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

