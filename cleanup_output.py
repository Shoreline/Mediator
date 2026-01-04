#!/usr/bin/env python3
"""
清理 output/ 目录中任务数小于阈值的文件

使用方法：
    # 预览将要删除的文件（不实际删除）
    python cleanup_output.py --dry-run
    
    # 清理任务数 < 100 的文件（默认）
    python cleanup_output.py
    
    # 清理任务数 < 50 的文件
    python cleanup_output.py --threshold 50
    
    # 自动确认删除（不需要交互）
    python cleanup_output.py --yes
"""

import os
import re
import glob
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


def parse_jsonl_filename(filename: str) -> Tuple[int, int, str]:
    """
    从 JSONL 文件名中提取信息
    
    支持的格式：
    1. 无采样: {job_num}_tasks_{total}_{rest}.jsonl
    2. 有采样: {job_num}_sampled_{rate}_seed{seed}_tasks_{total}_{rest}.jsonl
    
    Returns:
        (job_num, task_count, timestamp_and_model) 或 (None, None, None) 如果无法解析
    """
    # 尝试匹配有采样的格式
    pattern1 = r'^(\d+)_sampled_[\d.]+_seed\d+_tasks_(\d+)_(.+)\.jsonl$'
    match = re.match(pattern1, filename)
    if match:
        job_num = int(match.group(1))
        task_count = int(match.group(2))
        rest = match.group(3)
        return job_num, task_count, rest
    
    # 尝试匹配无采样的格式
    pattern2 = r'^(\d+)_tasks_(\d+)_(.+)\.jsonl$'
    match = re.match(pattern2, filename)
    if match:
        job_num = int(match.group(1))
        task_count = int(match.group(2))
        rest = match.group(3)
        return job_num, task_count, rest
    
    return None, None, None


def find_related_files(jsonl_filename: str, output_dir: str = 'output') -> List[str]:
    """
    根据 JSONL 文件名查找所有相关文件
    
    相关文件包括：
    1. 原始 JSONL 文件
    2. CSV 评估文件
    3. eval_debug.jsonl 文件
    4. VSP/CoMT-VSP 详细输出目录
    5. 采样版本的 CSV 文件
    
    Returns:
        相关文件和目录的路径列表
    """
    related = []
    
    # 1. 原始 JSONL 文件
    jsonl_path = os.path.join(output_dir, jsonl_filename)
    if os.path.exists(jsonl_path):
        related.append(jsonl_path)
    
    # 提取基础信息
    basename = jsonl_filename.replace('.jsonl', '')
    
    # 2. eval_debug.jsonl 文件
    eval_debug_path = os.path.join(output_dir, f"{basename}_eval_debug.jsonl")
    if os.path.exists(eval_debug_path):
        related.append(eval_debug_path)
    
    # 3. CSV 文件（可能有多种格式）
    # 格式1: {job_num}_eval_tasks_{total}_{rest}.csv
    # 格式2: {job_num}_eval-sampled_{rate}_seed{seed}_tasks_{total}_{rest}.csv
    
    # 解析 JSONL 文件名以构建 CSV 文件名
    job_num, task_count, rest = parse_jsonl_filename(jsonl_filename)
    
    if job_num is not None:
        # 查找所有可能的 CSV 文件（包括采样和非采样版本）
        csv_patterns = [
            f"{job_num}_eval_tasks_{task_count}_*.csv",
            f"{job_num}_eval-sampled_*_tasks_{task_count}_*.csv",
            f"eval_{job_num}_*_tasks_{task_count}_*.csv",  # 新格式
        ]
        
        for pattern in csv_patterns:
            csv_files = glob.glob(os.path.join(output_dir, pattern))
            related.extend(csv_files)
    
    # 4. VSP/CoMT-VSP 详细输出目录
    # 从文件名中提取 timestamp
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', basename)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        
        # VSP 目录格式
        vsp_dirs = [
            f"{output_dir}/vsp_details/vsp_{timestamp}",
            f"{output_dir}/vsp_details/{job_num}_tasks_{task_count}_vsp_{timestamp}",
            f"{output_dir}/comt_vsp_details/vsp_{timestamp}",
            f"{output_dir}/comt_vsp_details/{job_num}_tasks_{task_count}_vsp_{timestamp}",
        ]
        
        for vsp_dir in vsp_dirs:
            if os.path.exists(vsp_dir):
                related.append(vsp_dir)
    
    return related


def find_files_to_cleanup(output_dir: str = 'output', threshold: int = 100) -> Dict[str, List[str]]:
    """
    查找需要清理的文件
    
    Args:
        output_dir: output 目录路径
        threshold: 任务数阈值，小于此值的文件将被清理
    
    Returns:
        {jsonl_filename: [related_files_list]}
    """
    cleanup_candidates = {}
    
    # 查找所有 JSONL 文件
    jsonl_files = glob.glob(os.path.join(output_dir, '*.jsonl'))
    
    for jsonl_path in jsonl_files:
        filename = os.path.basename(jsonl_path)
        
        # 跳过 eval_debug.jsonl 文件（这些会通过主 JSONL 文件一起处理）
        if filename.endswith('_eval_debug.jsonl'):
            continue
        
        # 解析文件名
        job_num, task_count, rest = parse_jsonl_filename(filename)
        
        if job_num is None or task_count is None:
            # 无法解析的文件名，跳过
            continue
        
        # 检查是否低于阈值
        if task_count < threshold:
            # 查找所有相关文件
            related_files = find_related_files(filename, output_dir)
            if related_files:
                cleanup_candidates[filename] = related_files
    
    return cleanup_candidates


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_or_dir_size(path: str) -> int:
    """获取文件或目录的总大小"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except OSError:
                    pass
        return total
    return 0


def print_cleanup_summary(cleanup_candidates: Dict[str, List[str]]):
    """打印清理摘要"""
    if not cleanup_candidates:
        print("\n✅ 没有找到需要清理的文件")
        return
    
    print(f"\n{'='*80}")
    print(f"🗑️  清理摘要")
    print(f"{'='*80}\n")
    
    total_files = 0
    total_dirs = 0
    total_size = 0
    
    for i, (jsonl_filename, related_files) in enumerate(sorted(cleanup_candidates.items()), 1):
        # 解析信息
        job_num, task_count, rest = parse_jsonl_filename(jsonl_filename)
        
        print(f"{i}. Job {job_num} (tasks={task_count})")
        print(f"   主文件: {jsonl_filename}")
        
        files_count = 0
        dirs_count = 0
        group_size = 0
        
        for related_path in related_files:
            size = get_file_or_dir_size(related_path)
            group_size += size
            
            if os.path.isdir(related_path):
                dirs_count += 1
                total_dirs += 1
                print(f"   └─ [DIR]  {os.path.basename(related_path)} ({format_file_size(size)})")
            else:
                files_count += 1
                total_files += 1
                print(f"   └─ [FILE] {os.path.basename(related_path)} ({format_file_size(size)})")
        
        total_size += group_size
        print(f"   小计: {files_count} 个文件, {dirs_count} 个目录, {format_file_size(group_size)}")
        print()
    
    print(f"{'='*80}")
    print(f"总计: {len(cleanup_candidates)} 个 job, {total_files} 个文件, {total_dirs} 个目录")
    print(f"将释放空间: {format_file_size(total_size)}")
    print(f"{'='*80}\n")


def delete_files_and_dirs(file_list: List[str]) -> Tuple[int, int]:
    """
    删除文件和目录
    
    Returns:
        (deleted_files, deleted_dirs)
    """
    import shutil
    
    deleted_files = 0
    deleted_dirs = 0
    
    for path in file_list:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
                deleted_dirs += 1
                print(f"  ✅ 已删除目录: {path}")
            elif os.path.isfile(path):
                os.remove(path)
                deleted_files += 1
                print(f"  ✅ 已删除文件: {path}")
        except Exception as e:
            print(f"  ❌ 删除失败 {path}: {e}")
    
    return deleted_files, deleted_dirs


def main():
    parser = argparse.ArgumentParser(
        description="清理 output/ 目录中任务数小于阈值的文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览将要删除的文件（不实际删除）
  python cleanup_output.py --dry-run
  
  # 清理任务数 < 100 的文件（默认）
  python cleanup_output.py
  
  # 清理任务数 < 50 的文件
  python cleanup_output.py --threshold 50
  
  # 自动确认删除（不需要交互）
  python cleanup_output.py --yes
        """
    )
    
    parser.add_argument('--threshold', type=int, default=100,
                       help='任务数阈值，小于此值的文件将被清理（默认: 100）')
    parser.add_argument('--output_dir', default='output',
                       help='output 目录路径（默认: output）')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式：只显示将要删除的文件，不实际删除')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='自动确认，不需要交互式询问')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"🧹 output/ 目录清理工具")
    print(f"{'='*80}")
    print(f"目录: {args.output_dir}")
    print(f"阈值: tasks < {args.threshold}")
    if args.dry_run:
        print(f"模式: 预览模式（不会实际删除）")
    print(f"{'='*80}\n")
    
    # 查找需要清理的文件
    print("🔍 扫描文件...")
    cleanup_candidates = find_files_to_cleanup(args.output_dir, args.threshold)
    
    if not cleanup_candidates:
        print("\n✅ 没有找到需要清理的文件")
        return
    
    # 打印摘要
    print_cleanup_summary(cleanup_candidates)
    
    # 如果是预览模式，直接退出
    if args.dry_run:
        print("💡 这是预览模式，没有删除任何文件")
        print("   要实际删除，请运行: python cleanup_output.py")
        return
    
    # 询问用户确认
    if not args.yes:
        response = input("❓ 确认删除以上文件？(yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n❌ 取消删除")
            return
    
    # 执行删除
    print(f"\n{'='*80}")
    print(f"🗑️  开始删除...")
    print(f"{'='*80}\n")
    
    total_deleted_files = 0
    total_deleted_dirs = 0
    
    for jsonl_filename, related_files in sorted(cleanup_candidates.items()):
        job_num, task_count, _ = parse_jsonl_filename(jsonl_filename)
        print(f"\n🗑️  删除 Job {job_num} (tasks={task_count}):")
        
        deleted_files, deleted_dirs = delete_files_and_dirs(related_files)
        total_deleted_files += deleted_files
        total_deleted_dirs += deleted_dirs
    
    # 打印完成摘要
    print(f"\n{'='*80}")
    print(f"✅ 清理完成！")
    print(f"{'='*80}")
    print(f"已删除: {len(cleanup_candidates)} 个 job")
    print(f"  - 文件: {total_deleted_files} 个")
    print(f"  - 目录: {total_deleted_dirs} 个")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


