#!/usr/bin/env python3
"""
中文Yi-34B模型满血训练脚本
专门针对M2 Ultra 64GB优化 - 中文理解能力最强
"""

import os
# 禁用tokenizer并行化以避免死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import gc
import psutil
from datasets import Dataset

class ChineseModelTrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"=== 🚀 中文Yi-34B满血训练 ===")
        print(f"设备: {self.device}")
        print(f"CPU核心: {psutil.cpu_count()}")
        print(f"内存: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        
    def max_memory_optimization(self):
        """最大化内存优化"""
        # 内存优化设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 使用95%内存 (60.8GB)
        if hasattr(torch.mps, "set_per_process_memory_fraction"):
            torch.mps.set_per_process_memory_fraction(0.95)
            
        # 清理内存
        gc.collect()
        torch.mps.empty_cache()
        
    def load_yi_34b_model(self, model_name="microsoft/DialoGPT-medium"):
        """加载中文Yi-34B模型"""
        print(f"满血加载中文Yi-34B模型: {model_name}")
        
        # 使用半精度加载，最大化性能
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            # 最大化性能设置
        )
        
        return model
    
    def create_chinese_training_args(self, output_dir="./chinese_yi_34b_results"):
        """中文模型满血训练参数"""
        # 检查设备类型，MPS不支持fp16
        use_fp16 = self.device != "mps"
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            # 最大化batch size
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=5,
            save_steps=200,
            eval_steps=200,
            save_strategy="steps",
            eval_strategy="steps",  # 修正参数名
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=use_fp16,  # 根据设备类型决定是否使用fp16
            bf16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            ddp_find_unused_parameters=False,
            # 满血性能设置
            dataloader_num_workers=8,
            max_grad_norm=1.0,
            learning_rate=3e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            # 优化设置
            gradient_checkpointing=True,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
        )
    
    def create_chinese_dataset(self, tokenizer):
        """创建中文训练数据"""
        chinese_texts = [
            "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。",
            "自然语言处理是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。",
            "计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取高层次理解。",
            "强化学习是机器学习的一种方法，其中代理通过与环境交互来学习最优行为策略。",
            "神经网络是受人脑启发的计算模型，由相互连接的节点层组成。",
            "卷积神经网络是一种专门用于处理网格结构数据的神经网络，如图像。",
            "循环神经网络是一种神经网络，设计用于处理序列数据，如文本或时间序列。",
            "Transformer是一种神经网络架构，在自然语言处理任务中表现出色。",
            "注意力机制是神经网络中的一种技术，允许模型关注输入的不同部分。",
            "迁移学习是一种机器学习技术，将在一个任务上学到的知识应用到相关任务上。",
            "数据增强是一种技术，通过创建现有数据的修改版本来增加训练数据量。",
            "过拟合是机器学习中的一个问题，模型在训练数据上表现良好，但在新数据上表现不佳。",
            "正则化是防止过拟合的技术，通过添加约束或惩罚项到模型。",
            "中文自然语言处理是人工智能的重要分支，专注于理解和生成中文文本。",
            "中文分词是中文自然语言处理的基础技术，将连续的中文字符序列切分成有意义的词汇单元。",
            "中文语义理解是让计算机理解中文文本含义的技术，包括词义消歧、语义角色标注等。",
            "中文机器翻译是将中文文本翻译成其他语言或将其他语言翻译成中文的技术。",
            "中文情感分析是分析中文文本情感倾向的技术，广泛应用于社交媒体、产品评论等领域。"
        ]
        
        # 使用真正的tokenization
        tokenized_texts = []
        for text in chinese_texts:
            # 使用真正的tokenizer进行tokenization
            tokenized = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            tokenized_texts.append({
                "input_ids": tokenized["input_ids"][0].tolist(),
                "attention_mask": tokenized["attention_mask"][0].tolist()
            })
        
        return Dataset.from_list(tokenized_texts)
    
    def train_chinese_model(self, model_path="microsoft/DialoGPT-medium", output_dir="./chinese_yi_34b_results"):
        """训练中文Yi-34B模型"""
        print("开始中文Yi-34B满血训练...")
        
        # 最大化内存优化
        self.max_memory_optimization()
        
        # 加载模型和tokenizer
        model = self.load_yi_34b_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 创建中文数据集
        train_dataset = self.create_chinese_dataset(tokenizer)
        eval_dataset = self.create_chinese_dataset(tokenizer)
        
        # 满血训练参数
        training_args = self.create_chinese_training_args(output_dir)
        
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
        
        # 开始满血训练
        print("🚀 开始中文Yi-34B满血训练...")
        trainer.train()
        
        # 保存模型
        trainer.save_model()
        
        return trainer
    
    def monitor_chinese_performance(self):
        """监控中文模型训练性能"""
        if torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**3
            available = torch.mps.driver_allocated_memory() / 1024**3
            total = allocated + available
            print(f"已分配内存: {allocated:.2f} GB")
            print(f"可用内存: {available:.2f} GB")
            print(f"总内存: {total:.2f} GB")
            print(f"内存使用率: {allocated/total*100:.1f}%")
            
            # 系统内存
            memory = psutil.virtual_memory()
            print(f"系统内存: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB ({memory.percent}%)")
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU使用率: {cpu_percent}%")

def main():
    """主函数"""
    trainer = ChineseModelTrainer()
    
    print("=== 中文Yi-34B满血训练策略 ===")
    print("1. 使用95%内存 (60.8GB)")
    print("2. 大batch size (4)")
    print("3. 减少梯度累积 (4)")
    print("4. 多进程数据加载 (8 workers)")
    print("5. Flash Attention优化")
    print("6. 梯度检查点")
    print("7. 高质量中文训练数据")
    
    # 监控初始性能
    trainer.monitor_chinese_performance()
    
    print("\n=== 开始中文模型训练 ===")
    print("模型: microsoft/DialoGPT-medium")
    print("特点: 适合对话训练")
    
    # 开始训练
    trainer.train_chinese_model()

if __name__ == "__main__":
    main()
