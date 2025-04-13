# RAG-based-Multimodal-QA-System

RAG-based-Multimodal-QA-Syste/  
｜—— data/                    # 存放药品说明书图片和预处理数据  
｜—— models/                  # 存放训练好的模型  
｜—— src/  
｜   ｜—— config.py            # 配置文件  
｜   ｜—— data_processing.py   # 数据处理脚本  
｜   ｜—— ocr_processor.py     # OCR处理模块  
｜   ｜—— retrieval.py         # 检索系统  
｜   ｜—— multimodal_qa.py     # 多模态问答模型  
｜   ｜—— api.py               # FastAPI后端  
｜   ｜—— train.py             # 模型训练脚本  
｜—— frontend/                # Streamlit前端  
｜—— requirements.txt         # Python依赖  
｜—— README.md                # 项目说明  
｜—— .gitignore  

### 1.数据准备
* 将2000张药品说明书图片放入```data/images```目录
* 运行```python src/data_processing.py```预处理数据

### 2.训练模型
* 准备训练数据后运行```python src/train.py```微调LLaVA模型

### 3.启动后端
```uvicorn src.api:app --reload --host 0.0.0.0 --port 8000```

### 4.启动前端
```streamlit run frontend/app.py```
