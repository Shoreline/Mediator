#!/usr/bin/env python3
"""
检测 VSP 是否使用了工具（基于简单的模式匹配）
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_result_section(log_content: str) -> str:
    """
    提取 LLM 的实际回复部分（模型的实际输出）
    
    处理两种格式的 log：
    1. 新格式（有 'ATTENTION! YOUR ACTUAL TASK BEGINS HERE' 标记）：
       在该标记之后找 '# RESULT #:' 
    2. 旧格式（没有该标记）：
       使用最后一个 '# RESULT #:' 之后的内容
    """
    result_marker = "# RESULT #:"
    attention_marker = "ATTENTION! YOUR ACTUAL TASK BEGINS HERE"
    
    # 检查是否有新格式的标记
    attention_idx = log_content.rfind(attention_marker)
    
    if attention_idx != -1:
        # 新格式：在 ATTENTION 标记之后找 RESULT
        content_after_attention = log_content[attention_idx:]
        result_idx = content_after_attention.find(result_marker)
        
        if result_idx == -1:
            return ""
        
        # 返回 ATTENTION 之后的 RESULT 部分
        return content_after_attention[result_idx:]
    else:
        # 旧格式：使用最后一个 RESULT
        last_result_idx = log_content.rfind(result_marker)
        
        if last_result_idx == -1:
            return ""
        
        # 返回最后一个 RESULT 之后的内容
        return log_content[last_result_idx:]

def check_tool_usage(result_section: str) -> bool:
    """
    检查是否使用了 VSP 工具
    
    判断标准：在 # RESULT #: 之后是否包含 [VSP_TOOL_USED] 标记
    
    这个标记由 tools.py 中的工具函数在实际执行时输出，
    确保检测的是真正的 VSP 工具调用，而不是 LLM 自己编写的通用 Python 代码。
    """
    if not result_section:
        return False
    
    # 查找 [VSP_TOOL_USED] 标记
    # 这个标记只有在工具函数真正被调用时才会出现
    return '[VSP_TOOL_USED]' in result_section

def check_code_usage(result_section: str) -> bool:
    """
    检查是否使用了代码
    
    判断标准：在 # RESULT #: 之后是否包含 ```python 代码块
    
    这表示 LLM 生成了 Python 代码来解决问题。
    """
    if not result_section:
        return False
    
    # 查找 ```python 代码块
    return '```python' in result_section

def extract_user_interaction(log_content: str) -> str:
    """
    提取用户交互部分（去掉 VSP 的通用示例文本）
    
    返回最后一个 "# USER REQUEST #:" 之后的所有内容
    """
    user_request_marker = "# USER REQUEST #:"
    last_user_request_idx = log_content.rfind(user_request_marker)
    
    if last_user_request_idx == -1:
        return ""
    
    return log_content[last_user_request_idx:]

def save_examples_to_files(examples_with_content: dict, output_dir: str = "output"):
    """
    保存示例到文件
    
    Args:
        examples_with_content: {"used_tools": [(path, content), ...], "no_tools": [(path, content), ...]}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存使用了工具的示例
    used_tools_file = os.path.join(output_dir, "vsp_examples_used_tools.txt")
    with open(used_tools_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VSP 使用工具的示例（共 {} 个）\n".format(len(examples_with_content['used_tools'])))
        f.write("=" * 80 + "\n\n")
        
        for i, (path, content) in enumerate(examples_with_content['used_tools'], 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"示例 {i}/{len(examples_with_content['used_tools'])}\n")
            f.write(f"文件: {path}\n")
            f.write(f"{'='*80}\n\n")
            f.write(content)
            f.write("\n\n")
    
    # 保存未使用工具的示例
    no_tools_file = os.path.join(output_dir, "vsp_examples_no_tools.txt")
    with open(no_tools_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("VSP 未使用工具的示例（共 {} 个）\n".format(len(examples_with_content['no_tools'])))
        f.write("=" * 80 + "\n\n")
        
        for i, (path, content) in enumerate(examples_with_content['no_tools'], 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"示例 {i}/{len(examples_with_content['no_tools'])}\n")
            f.write(f"文件: {path}\n")
            f.write(f"{'='*80}\n\n")
            f.write(content)
            f.write("\n\n")
    
    print(f"\n✅ 示例已保存:")
    print(f"   - 使用工具: {used_tools_file}")
    print(f"   - 未使用工具: {no_tools_file}")

def analyze_vsp_logs(vsp_details_dir: str, summarize_examples: bool = False, max_examples: int = 100):
    """
    分析所有 VSP debug log
    
    Args:
        vsp_details_dir: VSP 详细输出目录
        summarize_examples: 是否保存示例到文件
        max_examples: 每种类型最多收集多少个示例
    """
    
    vsp_details_path = Path(vsp_details_dir)
    
    if not vsp_details_path.exists():
        print(f"❌ 目录不存在: {vsp_details_dir}")
        return
    
    # 查找所有 vsp_debug.log 文件
    log_files = list(vsp_details_path.rglob("vsp_debug.log"))
    
    print(f"📂 找到 {len(log_files)} 个 VSP debug log 文件\n")
    
    if len(log_files) == 0:
        print("⚠️  没有找到任何 vsp_debug.log 文件")
        return
    
    # 统计数据
    stats = {
        "total": 0,
        "used_vsp_tools": 0,
        "used_code": 0,
        "no_result_section": 0,
    }
    
    # 按 category 分组统计
    category_stats = defaultdict(lambda: {
        "used_vsp_tools": 0,
        "used_code": 0,
        "total": 0
    })
    
    # 示例文件（用于调试）
    examples = {
        "used_tools": [],
        "no_tools": []
    }
    
    # 示例内容（用于保存到文件）
    examples_with_content = {
        "used_tools": [],
        "no_tools": []
    }
    
    for log_file in log_files:
        stats["total"] += 1
        
        # 从路径中提取 category 和 index
        # 路径格式: .../vsp_TIMESTAMP/CATEGORY/INDEX/output/vsp_debug.log
        parts = log_file.parts
        try:
            category_idx = -4  # output 的上上上级是 category
            category = parts[category_idx]
            index = parts[category_idx + 1]
        except IndexError:
            category = "Unknown"
            index = "Unknown"
        
        # 读取文件内容
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
        except Exception as e:
            print(f"❌ 读取文件失败: {log_file} - {e}")
            continue
        
        # 提取 RESULT 部分
        result_section = extract_result_section(log_content)
        
        if not result_section:
            stats["no_result_section"] += 1
            continue
        
        # 检查是否使用了 VSP 工具
        used_vsp_tools = check_tool_usage(result_section)
        
        # 检查是否使用了代码
        used_code = check_code_usage(result_section)
        
        # VSP 工具和代码使用独立统计
        if used_vsp_tools:
            stats["used_vsp_tools"] += 1
            category_stats[category]["used_vsp_tools"] += 1
            if len(examples["used_tools"]) < 3:
                examples["used_tools"].append(str(log_file))
            
            # 如果需要保存示例，收集内容
            if summarize_examples and len(examples_with_content["used_tools"]) < max_examples:
                user_interaction = extract_user_interaction(log_content)
                if user_interaction:
                    examples_with_content["used_tools"].append((str(log_file), user_interaction))
        
        if used_code:
            stats["used_code"] += 1
            category_stats[category]["used_code"] += 1
            if len(examples["no_tools"]) < 3 and not used_vsp_tools:
                examples["no_tools"].append(str(log_file))
            
            # 如果需要保存示例，收集内容
            if summarize_examples and len(examples_with_content["no_tools"]) < max_examples and not used_vsp_tools:
                user_interaction = extract_user_interaction(log_content)
                if user_interaction:
                    examples_with_content["no_tools"].append((str(log_file), user_interaction))
        
        category_stats[category]["total"] += 1
    
    # 打印统计结果
    print(f"{'='*80}")
    print(f"📊 VSP 使用统计")
    print(f"{'='*80}\n")
    
    valid_total = stats['total'] - stats['no_result_section']
    print(f"总文件数: {stats['total']}")
    print(f"  - 使用了 VSP 工具: {stats['used_vsp_tools']} ({stats['used_vsp_tools']/valid_total*100:.1f}%)" if valid_total > 0 else "  - 使用了 VSP 工具: 0")
    print(f"  - 使用了代码: {stats['used_code']} ({stats['used_code']/valid_total*100:.1f}%)" if valid_total > 0 else "  - 使用了代码: 0")
    print(f"  - 无 RESULT 部分: {stats['no_result_section']}")
    
    # 按 category 统计
    print(f"\n{'='*80}")
    print(f"📋 按类别统计")
    print(f"{'='*80}\n")
    print(f"{'类别':<30} {'总数':<8} {'VSP工具':<10} {'代码':<10} {'VSP使用率':<12} {'代码使用率':<12}")
    print(f"{'-'*80}")
    
    for category in sorted(category_stats.keys()):
        cat_data = category_stats[category]
        total = cat_data["total"]
        used_vsp = cat_data["used_vsp_tools"]
        used_code = cat_data["used_code"]
        vsp_rate = used_vsp / total * 100 if total > 0 else 0
        code_rate = used_code / total * 100 if total > 0 else 0
        print(f"{category:<30} {total:<8} {used_vsp:<10} {used_code:<10} {vsp_rate:.1f}%         {code_rate:.1f}%")
    
    # 打印示例文件
    print(f"\n{'='*80}")
    print(f"📝 示例文件")
    print(f"{'='*80}\n")
    
    print("✅ 使用了工具的示例:")
    for example in examples["used_tools"]:
        print(f"   {example}")
    
    print("\n❌ 未使用工具的示例:")
    for example in examples["no_tools"]:
        print(f"   {example}")
    
    # 如果需要保存示例，保存到文件
    if summarize_examples:
        print(f"\n{'='*80}")
        print(f"💾 保存示例到文件")
        print(f"{'='*80}")
        save_examples_to_files(examples_with_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="分析 VSP debug log 文件，统计工具使用情况",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析默认目录
  python check_vsp_tool_usage.py
  
  # 分析指定目录
  python check_vsp_tool_usage.py --dir output/vsp_details/vsp_2025-11-12_20-18-34
  
  # 分析特定的批次
  python check_vsp_tool_usage.py --dir output/vsp_details/vsp_2025-11-12_20-18-34/08-Political_Lobbying
  
  # 保存示例到文件（去掉通用示例文本）
  python check_vsp_tool_usage.py --summarize_examples
  
  # 指定最多收集50个示例
  python check_vsp_tool_usage.py --summarize_examples --max_examples 50
        """
    )
    
    parser.add_argument(
        "--dir",
        default="output/vsp_details",
        help="VSP 详细输出目录路径（默认: output/vsp_details）"
    )
    
    parser.add_argument(
        "--summarize_examples",
        action="store_true",
        help="保存使用/未使用工具的示例到文件（默认: False）"
    )
    
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="每种类型最多收集多少个示例（默认: 100）"
    )
    
    args = parser.parse_args()
    
    # 展开用户路径（支持 ~ 符号）
    vsp_details_dir = os.path.expanduser(args.dir)
    
    analyze_vsp_logs(vsp_details_dir, summarize_examples=args.summarize_examples, max_examples=args.max_examples)

