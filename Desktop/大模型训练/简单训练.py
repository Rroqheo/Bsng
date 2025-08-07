#!/usr/bin/env python3
"""
简化的中文模型训练脚本
避免tokenizer并行化问题
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class SimpleChineseTrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"=== 🚀 简化中文模型训练 ===")
        print(f"设备: {self.device}")
        
    def load_model_and_tokenizer(self, model_name="microsoft/DialoGPT-medium"):
        """加载模型和tokenizer"""
        print(f"加载模型: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        return model, tokenizer
    
    def create_simple_dataset(self, tokenizer):
        """创建简单的训练数据集"""
        chinese_texts = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个子集。",
            "深度学习使用神经网络。",
            "自然语言处理专注于语言理解。",
            "计算机视觉处理图像和视频。",
            "强化学习通过交互学习。",
            "神经网络模拟人脑结构。",
            "卷积神经网络处理图像数据。",
            "循环神经网络处理序列数据。",
            "Transformer在NLP中表现出色。"
        ]
        
        # 简单的tokenization
        tokenized_data = []
        for text in chinese_texts:
            # 使用tokenizer进行编码
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            )
            
            tokenized_data.append({
                "input_ids": encoded["input_ids"][0].tolist(),
                "attention_mask": encoded["attention_mask"][0].tolist()
            })
        
        return Dataset.from_list(tokenized_data)
    
    def create_training_args(self, output_dir="./simple_chinese_results"):
        """创建训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            save_strategy="steps",
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,  # 禁用fp16避免MPS问题
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            dataloader_num_workers=0,  # 禁用多进程
            max_grad_norm=1.0,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            optim="adamw_torch",
        )
    
    def train(self, model_path="microsoft/DialoGPT-medium", output_dir="./simple_chinese_results"):
        """开始训练"""
        print("开始简化中文模型训练...")
        
        # 加载模型和tokenizer
        model, tokenizer = self.load_model_and_tokenizer(model_path)
        
        # 创建数据集
        train_dataset = self.create_simple_dataset(tokenizer)
        eval_dataset = self.create_simple_dataset(tokenizer)
        
        # 训练参数
        training_args = self.create_training_args(output_dir)
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        print("🚀 开始训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        print(f"✅ 训练完成！模型保存在: {output_dir}")
        
        return trainer

def main():
    """主函数"""
    trainer = SimpleChineseTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
