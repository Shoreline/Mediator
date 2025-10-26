#!/usr/bin/env python3
"""
JSONL 查看工具 - 让 JSONL 文件更易读

使用方法:
1. 查看所有记录（格式化）:
   python view_jsonl.py output/gpt-4o_2025-10-26_09-46-20.jsonl

2. 查看第 N 条记录:
   python view_jsonl.py output/gpt-4o_2025-10-26_09-46-20.jsonl --index 0

3. 只显示摘要:
   python view_jsonl.py output/gpt-4o_2025-10-26_09-46-20.jsonl --summary

4. 转换为 JSON 文件:
   python view_jsonl.py output/gpt-4o_2025-10-26_09-46-20.jsonl --to-json output.json
"""

import json
import sys
from pathlib import Path


def load_jsonl(filepath):
    """读取 JSONL 文件"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行解析失败: {e}", file=sys.stderr)
    return records


def print_record(record, index=None):
    """格式化打印单条记录"""
    if index is not None:
        print(f"\n{'='*80}")
        print(f"📄 记录 #{index}")
        print('='*80)
    
    print(json.dumps(record, indent=2, ensure_ascii=False))


def print_summary(records):
    """打印摘要信息"""
    print(f"\n📊 JSONL 文件摘要")
    print(f"{'='*80}")
    print(f"总记录数: {len(records)}")
    
    if records:
        print(f"\n字段结构:")
        first = records[0]
        for key in first.keys():
            print(f"  - {key}")
        
        print(f"\n前 3 条记录的 index:")
        for i, rec in enumerate(records[:3]):
            idx = rec.get('index', rec.get('origin', {}).get('index', 'N/A'))
            category = rec.get('origin', {}).get('category', 'N/A')
            question = rec.get('origin', {}).get('question', 'N/A')
            if len(question) > 60:
                question = question[:60] + "..."
            print(f"  [{i}] index={idx}, category={category}")
            print(f"      question: {question}")


def convert_to_json(records, output_path):
    """转换为标准 JSON 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"✅ 已转换为 JSON: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="JSONL 查看工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("file", help="JSONL 文件路径")
    parser.add_argument("--index", "-i", type=int, help="只显示第 N 条记录（从 0 开始）")
    parser.add_argument("--summary", "-s", action="store_true", help="只显示摘要")
    parser.add_argument("--to-json", "-j", help="转换为 JSON 文件并保存")
    parser.add_argument("--limit", "-l", type=int, help="限制显示前 N 条记录")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"❌ 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)
    
    records = load_jsonl(args.file)
    
    if not records:
        print("⚠️  文件为空或没有有效的 JSON 记录")
        sys.exit(1)
    
    if args.summary:
        print_summary(records)
    elif args.to_json:
        convert_to_json(records, args.to_json)
    elif args.index is not None:
        if 0 <= args.index < len(records):
            print_record(records[args.index], args.index)
        else:
            print(f"❌ 索引超出范围: {args.index} (共 {len(records)} 条记录)", file=sys.stderr)
            sys.exit(1)
    else:
        # 显示所有记录（或限制数量）
        limit = args.limit if args.limit else len(records)
        for i, record in enumerate(records[:limit]):
            print_record(record, i)
        
        if limit < len(records):
            print(f"\n... 还有 {len(records) - limit} 条记录（使用 --limit 增加显示数量）")


if __name__ == "__main__":
    main()

