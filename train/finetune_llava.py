import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import yaml

# 加载配置
with open("../configs/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 准备数据集
def prepare_dataset(dataset_path):
    """准备训练数据集"""
    dataset = load_dataset("json", data_files=dataset_path)
    
    def preprocess_function(examples):
        # 预处理逻辑
        return examples
    
    return dataset.map(preprocess_function, batched=True)

# 加载模型
model = LlavaForConditionalGeneration.from_pretrained(
    config['llava']['model_name'],
    device_map=config['llava']['device_map'],
    torch_dtype=getattr(torch, config['llava']['torch_dtype'])
)

tokenizer = AutoTokenizer.from_pretrained(config['llava']['model_name'])
tokenizer.pad_token = tokenizer.eos_token

# 应用LoRA
peft_config = LoraConfig(
    r=config['lora']['r'],
    lora_alpha=config['lora']['lora_alpha'],
    target_modules=config['lora']['target_modules'],
    lora_dropout=config['lora']['lora_dropout'],
    bias=config['lora']['bias'],
    task_type=config['lora']['task_type']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    fp16=True,
    remove_unused_columns=False
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepare_dataset("data/train.json"),
    eval_dataset=prepare_dataset("data/eval.json"),
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 训练
trainer.train()

# 保存模型
model.save_pretrained("./saved_model")
