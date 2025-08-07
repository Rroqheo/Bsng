#!/usr/bin/env python3
"""
测试简化训练的中文模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleModelTester:
    def __init__(self, model_path="./simple_chinese_results"):
        self.model_path = model_path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"=== 🧪 测试简化训练的中文模型 ===")
        print(f"模型路径: {model_path}")
        print(f"设备: {self.device}")
        
    def load_model(self):
        """加载训练好的模型"""
        print("正在加载模型...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print("✅ 模型加载完成")
        return model, tokenizer
    
    def test_chinese_dialogue(self, model, tokenizer):
        """测试中文对话"""
        print("\n=== 🗣️ 中文对话测试 ===")
        
        test_questions = [
            "人工智能是什么？",
            "机器学习的基本原理是什么？",
            "深度学习与传统机器学习有什么区别？",
            "自然语言处理有哪些应用？",
            "中文自然语言处理有什么特点？"
        ]
        
        for question in test_questions:
            print(f"\n问题: {question}")
            
            # 编码输入
            inputs = tokenizer(question, return_tensors="pt", padding=True)
            
            # 将输入移到正确的设备
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=1.0,
                    do_sample=False,  # 使用贪婪解码
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"回答: {response}")
            print("-" * 50)
    
    def test_text_generation(self, model, tokenizer):
        """测试文本生成"""
        print("\n=== 📝 文本生成测试 ===")
        
        prompts = [
            "今天天气很好，",
            "人工智能技术正在快速发展，",
            "中文是世界上最古老的语言之一，"
        ]
        
        for prompt in prompts:
            print(f"\n提示: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            # 将输入移到正确的设备
            if self.device == "mps":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=80,
                    num_return_sequences=1,
                    temperature=1.0,
                    do_sample=False,  # 使用贪婪解码
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成: {generated_text}")
            print("-" * 50)
    
    def test_model_performance(self, model, tokenizer):
        """测试模型性能"""
        print("\n=== ⚡ 性能测试 ===")
        
        import time
        
        test_text = "人工智能"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        # 将输入移到正确的设备
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 测试推理速度
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=30,
                num_return_sequences=1,
                do_sample=False,  # 使用贪婪解码
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        
        inference_time = end_time - start_time
        print(f"推理时间: {inference_time:.2f}秒")
        
        # 测试内存使用
        if torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**3
            print(f"模型内存使用: {allocated:.2f} GB")

def main():
    """主函数"""
    tester = SimpleModelTester()
    
    try:
        # 加载模型
        model, tokenizer = tester.load_model()
        
        # 测试对话
        tester.test_chinese_dialogue(model, tokenizer)
        
        # 测试文本生成
        tester.test_text_generation(model, tokenizer)
        
        # 测试性能
        tester.test_model_performance(model, tokenizer)
        
        print("\n🎉 模型测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查模型文件是否存在且完整")

if __name__ == "__main__":
    main()
