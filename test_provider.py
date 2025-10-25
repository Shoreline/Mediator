"""
Provider 测试脚本
用于快速测试不同 provider 对图片的推理能力

============================================================
📖 使用方法
============================================================

1. 快速测试（使用默认图片 kitten.jpeg）
   python test_provider.py --question "What do you see in this image?"

2. 指定图片测试
   python test_provider.py --image example/cars.jpg --question "How many cars are there?"

3. 测试所有示例图片
   python test_provider.py --all --question "Describe this image in detail"

4. 使用不同的 Provider
   # OpenAI (默认)
   python test_provider.py --provider openai --model gpt-4o --question "你看到了什么？"
   
   # Qwen (需要先实现 QwenProvider)
   python test_provider.py --provider qwen --model qwen2.5-vl-7b --question "描述这张图片"
   
   # VSP (需要先实现 VSPProvider)
   python test_provider.py --provider vsp --model vsp-model --question "Analyze this image"

5. 调整温度参数
   python test_provider.py --question "What's in the image?" --temp 0.7

6. 查看帮助
   python test_provider.py --help

============================================================
🎯 功能特点
============================================================
- 灵活的问题输入：支持命令行直接输入问题
- 多图片测试：可以测试单张或所有示例图片
- 预定义测试用例：内置了两个测试用例（小猫和汽车）
- 清晰的输出格式：带有表情符号的友好输出
- 错误处理：捕获并显示各种错误信息
- 自定义图片：可以测试任何路径的图片

============================================================
"""
import os
import asyncio
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from provider import get_provider, BaseProvider

# ============ 配置 ============

@dataclass
class TestConfig:
    provider: str                 # "openai" / "qwen" / "vsp"
    model_name: str               # e.g., "gpt-4o"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None
    proxy: Optional[str] = None

# ============ 测试用例 ============

@dataclass
class TestCase:
    name: str
    image_path: str
    question: str

# 预定义的测试用例
TEST_CASES: List[TestCase] = [
    TestCase(
        name="小猫图片",
        image_path="example/kitten.jpeg",
        question="What do you see in this image? Describe it in detail."
    ),
    TestCase(
        name="汽车图片",
        image_path="example/cars.jpg",
        question="What vehicles are in this image? What colors are they?"
    ),
]

# ============ 辅助函数 ============

def img_to_b64(path: str) -> str:
    """将图片转换为 base64 编码"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def create_test_prompt(test_case: TestCase, custom_question: str = None) -> Dict[str, Any]:
    """
    创建测试用的 prompt 结构
    """
    question = custom_question or test_case.question
    
    parts = [
        {"type": "text", "text": "You are a helpful multimodal assistant."},
        {"type": "text", "text": f"Question: {question}"},
        {"type": "image", "b64": img_to_b64(test_case.image_path)}
    ]
    
    return {
        "parts": parts,
        "meta": {"test_case": test_case.name}
    }

# ============ 测试函数 ============

async def test_provider(
    provider_name: str,
    model_name: str,
    test_case: TestCase = None,
    custom_question: str = None,
    temperature: float = 0.0
):
    """
    测试单个 provider
    
    Args:
        provider_name: provider 名称 ("openai" / "qwen" / "vsp")
        model_name: 模型名称
        test_case: 测试用例（如果为 None，使用第一个默认用例）
        custom_question: 自定义问题（如果提供，覆盖 test_case 的问题）
        temperature: 温度参数
    """
    # 使用默认测试用例
    if test_case is None:
        test_case = TEST_CASES[0]
    
    # 创建配置
    cfg = TestConfig(
        provider=provider_name,
        model_name=model_name,
        temperature=temperature
    )
    
    print(f"\n{'='*60}")
    print(f"🧪 测试 Provider: {provider_name}")
    print(f"📦 模型: {model_name}")
    print(f"🖼️  图片: {test_case.image_path}")
    print(f"❓ 问题: {custom_question or test_case.question}")
    print(f"{'='*60}\n")
    
    try:
        # 获取 provider
        provider = get_provider(cfg)
        
        # 创建 prompt
        prompt_struct = create_test_prompt(test_case, custom_question)
        
        # 发送请求
        print("⏳ 发送请求中...")
        response = await provider.send(prompt_struct, cfg)
        
        # 显示结果
        print(f"\n✅ 响应成功！\n")
        print(f"{'─'*60}")
        print(response)
        print(f"{'─'*60}\n")
        
        return response
        
    except NotImplementedError as e:
        print(f"⚠️  Provider 未实现: {e}")
        return None
    except Exception as e:
        print(f"❌ 错误: {type(e).__name__}: {e}")
        return None

async def test_all_images(provider_name: str, model_name: str, question: str):
    """
    用同一个问题测试所有示例图片
    """
    print(f"\n🚀 开始批量测试所有图片")
    print(f"Provider: {provider_name}, Model: {model_name}")
    print(f"问题: {question}\n")
    
    for test_case in TEST_CASES:
        await test_provider(provider_name, model_name, test_case, question)

async def test_custom_image(
    provider_name: str,
    model_name: str,
    image_path: str,
    question: str
):
    """
    测试自定义图片和问题
    """
    test_case = TestCase(
        name=f"自定义: {os.path.basename(image_path)}",
        image_path=image_path,
        question=question
    )
    await test_provider(provider_name, model_name, test_case, question)

# ============ 命令行接口 ============

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 Provider 推理能力")
    parser.add_argument("--provider", default="openai", 
                       help="Provider 名称: openai/qwen/vsp")
    parser.add_argument("--model", default="gpt-4o",
                       help="模型名称")
    parser.add_argument("--image", default=None,
                       help="图片路径（默认使用 example/kitten.jpeg）")
    parser.add_argument("--question", default=None,
                       help="问题文本（必需）")
    parser.add_argument("--all", action="store_true",
                       help="测试所有示例图片")
    parser.add_argument("--temp", type=float, default=0.0,
                       help="Temperature 参数")
    
    args = parser.parse_args()
    
    # 测试所有图片
    if args.all:
        if not args.question:
            print("❌ 错误: 使用 --all 时必须提供 --question 参数")
            return
        await test_all_images(args.provider, args.model, args.question)
    
    # 测试单张图片
    else:
        if not args.question:
            print("❌ 错误: 必须提供 --question 参数")
            print("\n示例用法:")
            print('  python test_provider.py --question "What do you see?"')
            print('  python test_provider.py --image example/cars.jpg --question "How many cars?"')
            print('  python test_provider.py --all --question "Describe this image"')
            return
        
        # 选择图片
        if args.image:
            await test_custom_image(args.provider, args.model, args.image, args.question)
        else:
            # 使用默认测试用例
            await test_provider(args.provider, args.model, 
                              custom_question=args.question,
                              temperature=args.temp)

if __name__ == "__main__":
    asyncio.run(main())

