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
    提取最后一个 # USER REQUEST # 之后的 # RESULT #: 内容
    
    VSP 的 prompt 包含很多 EXAMPLE，每个都有 # RESULT #:
    我们只关心最后一个真实的用户请求的结果
    """
    # 先找到最后一个 "# USER REQUEST #:"
    user_request_marker = "# USER REQUEST #:"
    last_user_request_idx = log_content.rfind(user_request_marker)
    
    if last_user_request_idx == -1:
        return ""
    
    # 在最后一个 USER REQUEST 之后找 RESULT
    content_after_user_request = log_content[last_user_request_idx:]
    
    result_marker = "# RESULT #:"
    result_idx = content_after_user_request.find(result_marker)
    
    if result_idx == -1:
        return ""
    
    return content_after_user_request[result_idx:]

def check_tool_usage(result_section: str) -> bool:
    """
    检查是否使用了工具
    
    判断标准：在 # RESULT #: 之后是否包含 ```python ... ``` 代码块
    """
    if not result_section:
        return False
    
    # 查找 ```python 代码块
    # 使用正则表达式匹配 ```python ... ``` 或 ```python ... ```的模式
    pattern = r'```python\s+.*?```'
    matches = re.search(pattern, result_section, re.DOTALL)
    
    return matches is not None

def analyze_vsp_logs(vsp_details_dir: str):
    """分析所有 VSP debug log"""
    
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
        "used_tools": 0,
        "no_tools": 0,
        "no_result_section": 0,
    }
    
    # 按 category 分组统计
    category_stats = defaultdict(lambda: {"used_tools": 0, "no_tools": 0, "total": 0})
    
    # 示例文件（用于调试）
    examples = {
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
        
        # 检查是否使用了工具
        used_tools = check_tool_usage(result_section)
        
        if used_tools:
            stats["used_tools"] += 1
            category_stats[category]["used_tools"] += 1
            if len(examples["used_tools"]) < 3:
                examples["used_tools"].append(str(log_file))
        else:
            stats["no_tools"] += 1
            category_stats[category]["no_tools"] += 1
            if len(examples["no_tools"]) < 3:
                examples["no_tools"].append(str(log_file))
        
        category_stats[category]["total"] += 1
    
    # 打印统计结果
    print(f"{'='*80}")
    print(f"📊 VSP 工具使用统计")
    print(f"{'='*80}\n")
    
    print(f"总文件数: {stats['total']}")
    print(f"  - 使用了工具: {stats['used_tools']} ({stats['used_tools']/stats['total']*100:.1f}%)")
    print(f"  - 未使用工具: {stats['no_tools']} ({stats['no_tools']/stats['total']*100:.1f}%)")
    print(f"  - 无 RESULT 部分: {stats['no_result_section']}")
    
    # 按 category 统计
    print(f"\n{'='*80}")
    print(f"📋 按类别统计")
    print(f"{'='*80}\n")
    print(f"{'类别':<30} {'总数':<8} {'使用工具':<10} {'未使用':<10} {'使用率':<10}")
    print(f"{'-'*80}")
    
    for category in sorted(category_stats.keys()):
        cat_data = category_stats[category]
        total = cat_data["total"]
        used = cat_data["used_tools"]
        not_used = cat_data["no_tools"]
        usage_rate = used / total * 100 if total > 0 else 0
        print(f"{category:<30} {total:<8} {used:<10} {not_used:<10} {usage_rate:.1f}%")
    
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
        """
    )
    
    parser.add_argument(
        "--dir",
        default="output/vsp_details",
        help="VSP 详细输出目录路径（默认: output/vsp_details）"
    )
    
    args = parser.parse_args()
    
    # 展开用户路径（支持 ~ 符号）
    vsp_details_dir = os.path.expanduser(args.dir)
    
    analyze_vsp_logs(vsp_details_dir)

