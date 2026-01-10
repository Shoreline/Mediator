#!/usr/bin/env python3
"""
清理 output/ 目录中任务数小于阈值的 job 文件夹，或清理特定任务编号的 job

使用方法：
    # 预览将要删除的文件（不实际删除）
    python cleanup_output.py --dry-run
    
    # 清理任务数 < 100 的 job（默认）
    python cleanup_output.py
    
    # 清理任务数 < 50 的 job
    python cleanup_output.py --threshold 50
    
    # 清理特定任务编号的所有 job
    python cleanup_output.py --job-num 42
    
    # 清理多个任务编号的 job
    python cleanup_output.py --job-num 42 43 44
    
    # 自动确认删除（不需要交互）
    python cleanup_output.py --yes
"""

import os
import re
import glob
import argparse
from typing import List, Dict, Tuple


def parse_job_folder_name(folder_name: str) -> Tuple[int, int, str, str, str]:
    """
    从 job 文件夹名称中提取信息
    
    格式: job_{num}_tasks_{total}_{Provider}_{model}_{timestamp}
    例如: job_104_tasks_202_ComtVsp_qwen3-vl-8b_0104_193618
    
    Returns:
        (job_num, task_count, provider, model, timestamp) 或 (None, None, None, None, None) 如果无法解析
    """
    pattern = r'^job_(\d+)_tasks_(\d+)_([^_]+)_(.+)_(\d{4}_\d{6})$'
    match = re.match(pattern, folder_name)
    
    if match:
        job_num = int(match.group(1))
        task_count = int(match.group(2))
        provider = match.group(3)
        model = match.group(4)
        timestamp = match.group(5)
        return job_num, task_count, provider, model, timestamp
    
    return None, None, None, None, None


def find_job_folders_to_cleanup(output_dir: str = 'output', threshold: int = 100) -> Dict[str, Dict]:
    """
    查找需要清理的 job 文件夹
    
    Args:
        output_dir: output 目录路径
        threshold: 任务数阈值，小于此值的 job 将被清理
    
    Returns:
        {folder_name: {job_num, task_count, provider, model, timestamp, path, size}}
    """
    cleanup_candidates = {}
    
    # 查找所有 job_ 开头的目录
    job_folders = glob.glob(os.path.join(output_dir, 'job_*'))
    
    for folder_path in job_folders:
        if not os.path.isdir(folder_path):
            continue
        
        folder_name = os.path.basename(folder_path)
        
        # 解析文件夹名
        job_num, task_count, provider, model, timestamp = parse_job_folder_name(folder_name)
        
        if job_num is None or task_count is None:
            # 无法解析的文件夹名，跳过
            continue
        
        # 检查是否低于阈值
        if task_count < threshold:
            cleanup_candidates[folder_name] = {
                'job_num': job_num,
                'task_count': task_count,
                'provider': provider,
                'model': model,
                'timestamp': timestamp,
                'path': folder_path,
                'size': get_dir_size(folder_path)
            }
    
    return cleanup_candidates


def find_job_folders_by_job_num(output_dir: str = 'output', job_nums: List[int] = None) -> Dict[str, Dict]:
    """
    查找特定任务编号的所有 job 文件夹
    
    Args:
        output_dir: output 目录路径
        job_nums: 要查找的任务编号列表
    
    Returns:
        {folder_name: {job_num, task_count, provider, model, timestamp, path, size}}
    """
    if job_nums is None:
        job_nums = []
    
    cleanup_candidates = {}
    job_nums_set = set(job_nums)
    
    # 查找所有 job_ 开头的目录
    job_folders = glob.glob(os.path.join(output_dir, 'job_*'))
    
    for folder_path in job_folders:
        if not os.path.isdir(folder_path):
            continue
        
        folder_name = os.path.basename(folder_path)
        
        # 解析文件夹名
        job_num, task_count, provider, model, timestamp = parse_job_folder_name(folder_name)
        
        if job_num is None:
            # 无法解析的文件夹名，跳过
            continue
        
        # 检查是否是要删除的任务编号
        if job_num in job_nums_set:
            cleanup_candidates[folder_name] = {
                'job_num': job_num,
                'task_count': task_count,
                'provider': provider,
                'model': model,
                'timestamp': timestamp,
                'path': folder_path,
                'size': get_dir_size(folder_path)
            }
    
    return cleanup_candidates


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_dir_size(path: str) -> int:
    """获取目录的总大小"""
    total = 0
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except OSError:
                    pass
    return total


def print_cleanup_summary(cleanup_candidates: Dict[str, Dict]):
    """打印清理摘要"""
    if not cleanup_candidates:
        print("\n✅ 没有找到需要清理的 job 文件夹")
        return
    
    print(f"\n{'='*80}")
    print(f"🗑️  清理摘要")
    print(f"{'='*80}\n")
    
    total_size = 0
    
    for i, (folder_name, info) in enumerate(sorted(cleanup_candidates.items(), key=lambda x: x[1]['job_num']), 1):
        job_num = info['job_num']
        task_count = info['task_count']
        provider = info['provider']
        model = info['model']
        timestamp = info['timestamp']
        size = info['size']
        
        print(f"{i}. Job {job_num} (tasks={task_count})")
        print(f"   文件夹: {folder_name}")
        print(f"   Provider: {provider}")
        print(f"   Model: {model}")
        print(f"   Timestamp: {timestamp}")
        print(f"   大小: {format_file_size(size)}")
        
        # 列出文件夹内容
        folder_path = info['path']
        if os.path.exists(folder_path):
            contents = []
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    contents.append(f"[DIR]  {item}")
                else:
                    contents.append(f"[FILE] {item}")
            
            if contents:
                print(f"   内容:")
                for content in sorted(contents):
                    print(f"     └─ {content}")
        
        total_size += size
        print()
    
    print(f"{'='*80}")
    print(f"总计: {len(cleanup_candidates)} 个 job 文件夹")
    print(f"将释放空间: {format_file_size(total_size)}")
    print(f"{'='*80}\n")


def delete_job_folder(folder_path: str) -> bool:
    """
    删除 job 文件夹
    
    Returns:
        True if successful, False otherwise
    """
    import shutil
    
    try:
        shutil.rmtree(folder_path)
        print(f"  ✅ 已删除: {folder_path}")
        return True
    except Exception as e:
        print(f"  ❌ 删除失败 {folder_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="清理 output/ 目录中任务数小于阈值的 job 文件夹，或清理特定任务编号的 job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览将要删除的文件（不实际删除）
  python cleanup_output.py --dry-run
  
  # 清理任务数 < 100 的 job（默认）
  python cleanup_output.py
  
  # 清理任务数 < 50 的 job
  python cleanup_output.py --threshold 50
  
  # 清理特定任务编号的所有 job
  python cleanup_output.py --job-num 42
  
  # 清理多个任务编号的 job
  python cleanup_output.py --job-num 42 43 44
  
  # 自动确认删除（不需要交互）
  python cleanup_output.py --yes
        """
    )
    
    parser.add_argument('--threshold', type=int, default=100,
                       help='任务数阈值，小于此值的 job 将被清理（默认: 100）')
    parser.add_argument('--job-num', type=int, nargs='+', metavar='NUM',
                       help='指定要清理的任务编号（可以指定多个）')
    parser.add_argument('--output_dir', default='output',
                       help='output 目录路径（默认: output）')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式：只显示将要删除的文件，不实际删除')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='自动确认，不需要交互式询问')
    
    args = parser.parse_args()
    
    # 检查互斥参数
    if args.job_num and args.threshold != 100:
        print("❌ 错误: --job-num 和 --threshold 不能同时使用")
        return
    
    print(f"{'='*80}")
    print(f"🧹 output/ 目录清理工具（新版 - 基于 job 文件夹）")
    print(f"{'='*80}")
    print(f"目录: {args.output_dir}")
    
    if args.job_num:
        print(f"模式: 按任务编号清理")
        print(f"任务编号: {', '.join(map(str, sorted(args.job_num)))}")
    else:
        print(f"模式: 按任务数阈值清理")
        print(f"阈值: tasks < {args.threshold}")
    
    if args.dry_run:
        print(f"预览模式: 不会实际删除")
    print(f"{'='*80}\n")
    
    # 查找需要清理的 job 文件夹
    print("🔍 扫描 job 文件夹...")
    if args.job_num:
        cleanup_candidates = find_job_folders_by_job_num(args.output_dir, args.job_num)
    else:
        cleanup_candidates = find_job_folders_to_cleanup(args.output_dir, args.threshold)
    
    if not cleanup_candidates:
        print("\n✅ 没有找到需要清理的 job 文件夹")
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
        response = input("❓ 确认删除以上 job 文件夹？(yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\n❌ 取消删除")
            return
    
    # 执行删除
    print(f"\n{'='*80}")
    print(f"🗑️  开始删除...")
    print(f"{'='*80}\n")
    
    deleted_count = 0
    
    for folder_name, info in sorted(cleanup_candidates.items(), key=lambda x: x[1]['job_num']):
        job_num = info['job_num']
        task_count = info['task_count']
        folder_path = info['path']
        
        print(f"\n🗑️  删除 Job {job_num} (tasks={task_count}):")
        
        if delete_job_folder(folder_path):
            deleted_count += 1
    
    # 打印完成摘要
    print(f"\n{'='*80}")
    print(f"✅ 清理完成！")
    print(f"{'='*80}")
    print(f"已删除: {deleted_count} 个 job 文件夹")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
