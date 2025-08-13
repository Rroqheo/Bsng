# Qwen Code - 智能代码生成工具

基于 Qwen 2.5 Coder 模型的智能代码生成、补全、解释、优化和调试工具。

## ✨ 功能特性

- 🎯 **代码生成**: 根据自然语言描述生成高质量代码
- 🔧 **代码补全**: 智能补全未完成的代码片段
- 📖 **代码解释**: 详细解释代码的功能和实现原理
- ⚡ **代码优化**: 提升代码性能和可读性
- 🐛 **代码调试**: 自动修复代码错误
- 🌍 **多语言支持**: 支持 20+ 种编程语言
- 💬 **交互模式**: 实时对话式代码生成
- 🚀 **流式输出**: 实时查看生成过程

## 🛠️ 支持的编程语言

Python, JavaScript, TypeScript, Java, C++, C, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, R, SQL, HTML, CSS, Bash, PowerShell 等

## 📦 快速开始

### 1. 自动安装（推荐）

```bash
chmod +x setup.sh
./setup.sh
```

### 2. 手动安装

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行演示

```bash
python demo.py
```

## 🎮 使用方法

### 交互模式

```bash
python qwen_code.py --interactive
```

在交互模式中，你可以使用以下命令：

- `generate <需求>` - 生成代码
- `complete <代码>` - 代码补全
- `explain <代码>` - 代码解释
- `optimize <代码>` - 代码优化
- `debug <代码> [错误信息]` - 代码调试
- `lang <语言>` - 设置编程语言
- `quit` - 退出

### 命令行模式

```bash
# 代码生成
python qwen_code.py --mode generate --prompt "写一个快速排序算法" --language python

# 代码补全
python qwen_code.py --mode complete --code "def fibonacci(n):" --language python

# 代码解释
python qwen_code.py --mode explain --code "你的代码" --language python

# 代码优化
python qwen_code.py --mode optimize --code "你的代码" --language python

# 代码调试
python qwen_code.py --mode debug --code "有问题的代码" --error "错误信息" --language python
```

### Python API

```python
from qwen_code import QwenCodeGenerator

# 初始化
generator = QwenCodeGenerator()
generator.load_model()

# 生成代码
code = generator.generate_code("写一个计算器类", "python")
print(code)

# 代码补全
completed = generator.complete_code("def hello_world():", "python")
print(completed)

# 代码解释
explanation = generator.explain_code("print('Hello, World!')", "python")
print(explanation)
```

## ⚙️ 配置选项

编辑 `config.json` 文件来自定义配置：

```json
{
    "model_config": {
        "default_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "device": "auto"
    },
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_new_tokens": 2048
    }
}
```

## 🎯 使用示例

### 示例 1: 生成 Python 函数

**输入**: "写一个计算两个数最大公约数的函数"

**输出**:
```python
def gcd(a, b):
    """
    计算两个数的最大公约数（欧几里得算法）
    
    Args:
        a (int): 第一个数
        b (int): 第二个数
    
    Returns:
        int: 最大公约数
    """
    while b:
        a, b = b, a % b
    return a

# 使用示例
print(gcd(48, 18))  # 输出: 6
```

### 示例 2: 代码优化

**输入代码**:
```python
def find_max(numbers):
    max_num = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > max_num:
            max_num = numbers[i]
    return max_num
```

**优化后**:
```python
def find_max(numbers):
    """
    找到列表中的最大值
    
    Args:
        numbers (list): 数字列表
    
    Returns:
        数字: 最大值
    """
    if not numbers:
        raise ValueError("列表不能为空")
    
    return max(numbers)  # 使用内置函数，更简洁高效
```

## 🔧 高级功能

### 流式输出

```bash
python qwen_code.py --mode generate --prompt "你的需求" --stream
```

### 自定义模型

```bash
python qwen_code.py --model "Qwen/Qwen2.5-Coder-14B-Instruct" --interactive
```

### 指定设备

```bash
python qwen_code.py --device cuda --interactive  # 使用 GPU
python qwen_code.py --device cpu --interactive   # 使用 CPU
```

## 📋 系统要求

- Python 3.8+
- 8GB+ RAM（推荐 16GB+）
- GPU（可选，但强烈推荐用于更快的推理）

### GPU 支持

- NVIDIA GPU with CUDA 11.8+
- Apple Silicon (M1/M2) with MPS
- AMD GPU with ROCm（实验性支持）

## 🚨 注意事项

1. **首次运行**: 首次运行时会自动下载模型文件（约 4-15GB），请确保网络连接稳定
2. **内存需求**: 7B 模型需要至少 8GB RAM，14B+ 模型需要 16GB+ RAM
3. **GPU 加速**: 使用 GPU 可以显著提升生成速度
4. **模型选择**: 可以根据硬件配置选择合适的模型大小

## 🔍 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 使用更小的模型
   python qwen_code.py --model "Qwen/Qwen2.5-Coder-1.5B-Instruct"
   ```

2. **CUDA 错误**
   ```bash
   # 强制使用 CPU
   python qwen_code.py --device cpu
   ```

3. **网络问题**
   ```bash
   # 设置代理（如果需要）
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如果遇到问题，请：

1. 查看本 README 的故障排除部分
2. 提交 Issue 描述问题
3. 加入社区讨论

---

**享受智能编程的乐趣！** 🎉

## Microsoft Edge 下载设置（避免保存到桌面）

- 更改默认下载目录
  - Edge → Settings → Downloads → Location → Change
  - 建议选择 `~/Downloads` 或 `~/Documents/Downloads`
- 每次下载前询问保存位置
  - Edge → Settings → Downloads → 打开 “Ask me where to save each file”
- 关闭自动打开下载
  - Edge → Settings → Downloads → 关闭 “Automatically open downloads”
- 设为默认浏览器（可选）
  - macOS 设置 → 桌面与程序坞 → 默认网页浏览器 → Microsoft Edge
- 命令行快速打开（macOS）
  - ```bash
    open -a "Microsoft Edge"
    ```