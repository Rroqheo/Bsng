#!/usr/bin/env python3
"""
大模型训练脚本 (适配M2 Ultra 64GB内存)
使用LoRA微调技术，只训练少量参数
"""

import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import gc

class LargeModelTrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
    def load_quantized_model(self, model_path):
        """加载量化大模型"""
        print(f"加载量化模型: {model_path}")
        
        # 使用4-bit量化加载
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        return model
    
    def setup_lora_config(self):
        """LoRA配置 - 只训练少量参数"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # 降低rank以节省内存
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    
    def train_large_model(self, model_path, output_dir="./large_model_results"):
        """训练大模型"""
        print("开始训练大模型...")
        
        # 加载模型和tokenizer
        model = self.load_quantized_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        # 训练参数 (针对大模型优化)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=5,
            save_steps=100,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # 开始训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_list([{"input_ids": [1,2,3,4,5]}]),
            data_collator=lambda x: {"input_ids": torch.stack([torch.tensor(f) for f in x])}
        )
        
        trainer.train()
        trainer.save_model()
        
        return trainer

def main():
    print("=== 🚀 M2 Ultra 大模型训练工具 ===")
    print("使用LoRA微调技术，适配64GB内存")
    
    trainer = LargeModelTrainer()
    
    # 选择模型路径
    model_path = "./models/CodeLlama-34B-4bit"
    
    if os.path.exists(model_path):
        print(f"开始训练: {model_path}")
        trainer.train_large_model(model_path)
    else:
        print(f"❌ 模型不存在: {model_path}")
        print("请先运行量化脚本")

if __name__ == "__main__":
    main()
