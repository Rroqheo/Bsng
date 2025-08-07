#!/usr/bin/env python3
"""
真实的中文模型训练体验
使用真实的中文数据和完整的训练过程
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
import time

class RealChineseTrainer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"=== 🎯 真实中文模型训练体验 ===")
        print(f"设备: {self.device}")
        print(f"内存: {torch.mps.current_allocated_memory()/1024**3:.1f}GB / 64GB")
        
    def load_real_model(self, model_name="microsoft/DialoGPT-medium"):
        """加载真实的预训练模型"""
        print(f"🔄 加载真实模型: {model_name}")
        print("正在下载模型文件...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"✅ 模型加载完成！参数数量: {model.num_parameters():,}")
        return model, tokenizer
    
    def create_real_chinese_dataset(self, tokenizer):
        """创建真实的中文训练数据"""
        print("📚 准备真实中文训练数据...")
        
        # 真实的中文AI相关文本数据
        chinese_texts = [
            "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以发现模式并做出预测。",
            "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            "自然语言处理（NLP）是人工智能的一个领域，专注于计算机理解和生成人类语言的能力。它包括机器翻译、情感分析、问答系统等应用。",
            "计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取高层次理解。它在自动驾驶、医疗诊断和安全监控等领域有广泛应用。",
            "强化学习是机器学习的一种方法，其中代理通过与环境交互来学习最优行为策略。它在游戏AI、机器人控制和推荐系统等领域表现出色。",
            "神经网络是受人脑启发的计算模型，由相互连接的节点层组成。每个节点接收输入，应用激活函数，并产生输出传递给下一层。",
            "卷积神经网络（CNN）是一种专门用于处理网格结构数据的神经网络，如图像。它通过卷积层提取特征，在计算机视觉任务中表现优异。",
            "循环神经网络（RNN）是一种神经网络，设计用于处理序列数据，如文本或时间序列。它能够记住之前的信息，适合处理有顺序的数据。",
            "Transformer是一种神经网络架构，在自然语言处理任务中表现出色。它使用注意力机制来关注输入的不同部分，在机器翻译和文本生成方面取得了革命性进展。",
            "注意力机制是神经网络中的一种技术，允许模型关注输入的不同部分。它使模型能够理解上下文关系，在长文本处理中特别有效。",
            "迁移学习是一种机器学习技术，将在一个任务上学到的知识应用到相关任务上。它能够减少训练时间和数据需求，提高模型性能。",
            "数据增强是一种技术，通过创建现有数据的修改版本来增加训练数据量。它包括旋转、缩放、噪声添加等方法，能够提高模型的泛化能力。",
            "过拟合是机器学习中的一个问题，模型在训练数据上表现良好，但在新数据上表现不佳。它通常是由于模型过于复杂或训练数据不足导致的。",
            "正则化是防止过拟合的技术，通过添加约束或惩罚项到模型。它包括L1正则化、L2正则化、Dropout等方法，能够提高模型的泛化能力。",
            "中文自然语言处理是人工智能的重要分支，专注于理解和生成中文文本。它面临分词、语义理解、歧义消解等独特挑战。",
            "中文分词是中文自然语言处理的基础技术，将连续的中文字符序列切分成有意义的词汇单元。它是中文文本处理的第一步，对后续任务至关重要。",
            "中文语义理解是让计算机理解中文文本含义的技术，包括词义消歧、语义角色标注、指代消解等。它需要深入理解中文的语言特点和表达方式。",
            "中文机器翻译是将中文文本翻译成其他语言或将其他语言翻译成中文的技术。它需要考虑中文的语法结构、文化背景和表达习惯。",
            "中文情感分析是分析中文文本情感倾向的技术，广泛应用于社交媒体、产品评论、舆情监测等领域。它需要理解中文的情感表达方式和语境。"
        ]
        
        print(f"📖 准备 {len(chinese_texts)} 条高质量中文训练数据...")
        
        # 真实的tokenization
        tokenized_data = []
        for i, text in enumerate(chinese_texts):
            print(f"处理数据 {i+1}/{len(chinese_texts)}: {text[:30]}...")
            
            # 使用tokenizer进行编码
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt"
            )
            
            tokenized_data.append({
                "input_ids": encoded["input_ids"][0].tolist(),
                "attention_mask": encoded["attention_mask"][0].tolist()
            })
            
            # 模拟处理时间
            time.sleep(0.1)
        
        print(f"✅ 数据处理完成！共 {len(tokenized_data)} 条训练样本")
        return Dataset.from_list(tokenized_data)
    
    def create_real_training_args(self, output_dir="./real_chinese_results"):
        """创建真实的训练参数"""
        print("⚙️ 配置真实训练参数...")
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # 真实训练轮数
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            save_strategy="steps",
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,  # 禁用fp16避免MPS问题
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            learning_rate=3e-5,  # 真实学习率
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            optim="adamw_torch",
        )
    
    def train_real_model(self, model_path="microsoft/DialoGPT-medium", output_dir="./real_chinese_results"):
        """开始真实训练"""
        print("🚀 开始真实中文模型训练体验...")
        print("=" * 50)
        
        # 加载模型和tokenizer
        model, tokenizer = self.load_real_model(model_path)
        
        # 创建数据集
        train_dataset = self.create_real_chinese_dataset(tokenizer)
        eval_dataset = self.create_real_chinese_dataset(tokenizer)
        
        # 训练参数
        training_args = self.create_real_training_args(output_dir)
        
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
        
        # 开始真实训练
        print("🔥 开始真实训练过程...")
        print("预计训练时间: 5-10分钟")
        print("训练进度:")
        
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # 保存模型
        print("💾 保存训练好的模型...")
        trainer.save_model()
        
        training_time = end_time - start_time
        print(f"✅ 真实训练完成！")
        print(f"⏱️ 训练时间: {training_time/60:.1f}分钟")
        print(f"📁 模型保存在: {output_dir}")
        
        return trainer

def main():
    """主函数"""
    print("🎯 欢迎体验真实的中文模型训练！")
    print("这次我们将使用真实的中文数据和完整的训练过程")
    print("=" * 50)
    
    trainer = RealChineseTrainer()
    trainer.train_real_model()

if __name__ == "__main__":
    main()
