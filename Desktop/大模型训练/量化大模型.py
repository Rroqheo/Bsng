#!/usr/bin/env python3
"""
量化大模型以适配M2 Ultra 64GB内存
支持：Llama-2-70B, CodeLlama-34B, Yi-34B, Qwen-72B
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import gc

class LargeModelQuantizer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
    def quantize_model(self, model_name, output_dir, bits=4):
        """量化大模型"""
        print(f"开始量化模型: {model_name}")
        print(f"量化位数: {bits}-bit")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 量化配置
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False
        )
        
        # 量化模型
        print("开始量化...")
        model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config=quantize_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 保存量化模型
        print("保存量化模型...")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 清理内存
        del model
        gc.collect()
        torch.mps.empty_cache()
        
        print(f"✅ 量化完成！保存到: {output_dir}")

def main():
    print("=== 🚀 M2 Ultra 大模型量化工具 ===")
    print("将大模型压缩到64GB内存内")
    
    quantizer = LargeModelQuantizer()
    
    # 示例：量化CodeLlama-34B
    quantizer.quantize_model(
        "codellama/CodeLlama-34b-hf",
        "./models/CodeLlama-34B-4bit",
        4
    )

if __name__ == "__main__":
    main()
