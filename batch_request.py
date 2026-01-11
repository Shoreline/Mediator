#!/usr/bin/env python3
"""
批量运行 request.py 的脚本

通过配置 args_combo 列表，可以组合不同的参数运行多次 request.py

使用方式：
    python batch_request.py

配置说明：
    args_combo 是一个列表，每个元素代表一组参数变体
    - 如果元素是字符串，表示固定参数（所有组合都会使用）
    - 如果元素是列表，表示需要遍历的参数变体
    
    最终会生成所有变体的笛卡尔积组合

示例：
    args_combo = [
        "--categories 12-Health_Consultation",  # 固定参数
        [  # 需要遍历的参数变体
            '--provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct"',
            '--provider openrouter --model "qwen/qwen3-vl-235b-a22b-instruct"',
        ],
    ]
    
    这会运行 2 次 request.py：
    1. --categories 12-Health_Consultation --provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct"
    2. --categories 12-Health_Consultation --provider openrouter --model "qwen/qwen3-vl-235b-a22b-instruct"
"""

import subprocess
import sys
import os
import itertools
import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, TextIO


# ============ 日志管理 ============

class TeeWriter:
    """同时写入多个输出流的类"""
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, text):
        for w in self.writers:
            w.write(text)
            w.flush()
    
    def flush(self):
        for w in self.writers:
            w.flush()


# 全局日志文件句柄
_log_file: Optional[TextIO] = None
_original_stdout = None


def setup_logging(log_path: str):
    """设置日志输出到文件和控制台"""
    global _log_file, _original_stdout
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_file = open(log_path, 'w', encoding='utf-8')
    _original_stdout = sys.stdout
    sys.stdout = TeeWriter(_original_stdout, _log_file)
    
    return _log_file


def close_logging():
    """关闭日志文件"""
    global _log_file, _original_stdout
    
    if _original_stdout:
        sys.stdout = _original_stdout
    
    if _log_file:
        _log_file.close()
        _log_file = None


# ============ 配置区域 ============

# 参数组合配置
# - 字符串：固定参数（所有组合都会使用）
# - 列表：需要遍历的参数变体
args_combo = [
    # 固定参数：类别和任务数
    "--sampling_rate 0.12",
    
    # 需要遍历的参数变体：不同的 provider 和 model 组合
    [

        # '--provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-instruct" --comt_sample_id deletion-2371',
        # '--provider comt_vsp --model "google/gemini-2.5-flash" --comt_sample_id deletion-2371',
        '--provider comt_vsp --model "qwen/qwen3-vl-30b-a3b-instruct" --comt_sample_id deletion-2371',
        '--provider comt_vsp --model "qwen/qwen3-vl-8b-instruct" --comt_sample_id deletion-2371',
        '--provider comt_vsp --model "qwen/qwen3-vl-235b-a22b-thinking" --comt_sample_id deletion-2371',
        '--provider comt_vsp --model "qwen/qwen3-vl-30b-a3b-thinking" --comt_sample_id deletion-2371',
        '--provider comt_vsp --model "qwen/qwen3-vl-8b-thinking" --comt_sample_id deletion-2371',
        # '--provider comt_vsp --model "mistralai/ministral-14b-2512" --comt_sample_id deletion-2371',
        # '--provider comt_vsp --model "mistralai/ministral-8b-2512" --comt_sample_id deletion-2371',
        # '--provider comt_vsp --model "mistralai/ministral-3b-2512" --comt_sample_id deletion-2371',
        # '--provider comt_vsp --model "openai/gpt-5" --comt_sample_id deletion-2371',

    ],
]

# 是否显示详细输出
VERBOSE = True

# 是否在完成后生成报告
GENERATE_REPORT = True


# ============ 运行结果数据结构 ============

@dataclass
class RunResult:
    """单次运行的结果信息"""
    run_index: int                          # 运行序号
    args_str: str                           # 命令行参数
    success: bool                           # 是否成功
    start_time: datetime                    # 开始时间
    end_time: datetime                      # 结束时间
    duration: timedelta                     # 耗时
    task_num: Optional[int] = None          # 任务编号
    total_tasks: Optional[int] = None       # 总任务数
    output_file: Optional[str] = None       # 输出文件路径
    eval_file: Optional[str] = None         # 评估结果文件路径
    vsp_dir: Optional[str] = None           # VSP 详细输出目录
    error_message: Optional[str] = None     # 错误信息
    error_key: Optional[str] = None         # 错误标识（如 "stop_reason" 表示 request.py 内部自动停止）
    
    # 从参数中提取的信息
    provider: Optional[str] = None
    model: Optional[str] = None
    categories: Optional[str] = None
    max_tasks_arg: Optional[int] = None


def parse_args_str(args_str: str) -> dict:
    """从参数字符串中提取关键信息"""
    info = {}
    
    # 提取 provider
    provider_match = re.search(r'--provider\s+(\S+)', args_str)
    if provider_match:
        info['provider'] = provider_match.group(1)
    
    # 提取 model（可能带引号）
    model_match = re.search(r'--model\s+["\']?([^"\']+)["\']?', args_str)
    if model_match:
        info['model'] = model_match.group(1).strip()
    
    # 提取 categories
    categories_match = re.search(r'--categories\s+(\S+)', args_str)
    if categories_match:
        info['categories'] = categories_match.group(1)
    
    # 提取 max_tasks
    max_tasks_match = re.search(r'--max_tasks\s+(\d+)', args_str)
    if max_tasks_match:
        info['max_tasks_arg'] = int(max_tasks_match.group(1))
    
    return info


def parse_output(output: str) -> dict:
    """从 request.py 的输出中提取关键信息"""
    info = {}
    
    # 提取任务编号
    task_num_match = re.search(r'🔢 任务编号:\s*(\d+)', output)
    if task_num_match:
        info['task_num'] = int(task_num_match.group(1))
    
    # 提取输出文件路径（重命名后的）
    output_file_match = re.search(r'✅ 文件已重命名:\s*(\S+\.jsonl)', output)
    if output_file_match:
        info['output_file'] = output_file_match.group(1)
    else:
        # 尝试从"输出文件:"行提取
        output_file_match2 = re.search(r'输出文件:\s*(\S+\.jsonl)', output)
        if output_file_match2:
            info['output_file'] = output_file_match2.group(1)
    
    # 提取 VSP 详细输出目录
    vsp_dir_match = re.search(r'✅ VSP 详细输出目录已重命名:\s*(\S+)', output)
    if vsp_dir_match:
        info['vsp_dir'] = vsp_dir_match.group(1)
    
    # 提取评估结果文件
    eval_file_match = re.search(r'✅ 评估指标已保存:\s*(\S+\.csv)', output)
    if eval_file_match:
        info['eval_file'] = eval_file_match.group(1)
    
    # 提取总任务数
    total_tasks_match = re.search(r'总任务数:\s*(\d+)', output)
    if total_tasks_match:
        info['total_tasks'] = int(total_tasks_match.group(1))
    
    return info


def format_duration(td: timedelta) -> str:
    """格式化时间间隔"""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"


# ============ 主逻辑 ============

def generate_combinations(args_combo):
    """
    生成所有参数组合
    
    Args:
        args_combo: 参数组合配置列表
        
    Returns:
        参数组合列表，每个元素是完整的命令行参数字符串
    """
    # 将所有元素转换为列表格式
    normalized = []
    for item in args_combo:
        if isinstance(item, str):
            normalized.append([item])
        elif isinstance(item, list):
            normalized.append(item)
        else:
            raise ValueError(f"不支持的参数类型: {type(item)}")
    
    # 生成笛卡尔积
    combinations = list(itertools.product(*normalized))
    
    # 合并每个组合的参数
    result = []
    for combo in combinations:
        args = " ".join(combo)
        result.append(args)
    
    return result


def run_request(args_str: str, run_index: int, total_runs: int) -> RunResult:
    """
    运行一次 request.py
    
    Args:
        args_str: 命令行参数字符串
        run_index: 当前运行序号（从1开始）
        total_runs: 总运行次数
        
    Returns:
        RunResult 对象，包含运行结果的详细信息
    """
    print(f"\n{'='*80}")
    print(f"🚀 运行 [{run_index}/{total_runs}]")
    print(f"{'='*80}")
    print(f"📋 参数: {args_str}")
    print(f"{'='*80}\n")
    
    # 构建完整命令
    cmd = f"python request.py {args_str}"
    
    if VERBOSE:
        print(f"💻 执行命令: {cmd}\n")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 解析参数
    args_info = parse_args_str(args_str)
    
    try:
        # 设置环境变量，禁用 Python 的输出缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 运行命令，捕获输出同时显示在终端
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # 实时输出并收集
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # 实时显示
            sys.stdout.flush()   # 确保立即刷新到屏幕和日志
            output_lines.append(line)
        
        process.wait()
        output = ''.join(output_lines)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # 检测 STOP_REASON（由 request.py 内部自动停止时打印）
        stop_match = re.search(r"自动停止原因:\s*(.+)", output)
        error_msg = None
        error_key = None
        if stop_match:
            error_msg = stop_match.group(1).strip()
            error_key = "stop_reason"
        
        output_info = parse_output(output)
        
        success_flag = (process.returncode == 0) and (error_key is None)
        
        if success_flag:
            print(f"\n✅ 运行 [{run_index}/{total_runs}] 完成 (耗时: {format_duration(duration)})")
        else:
            print(f"\n❌ 运行 [{run_index}/{total_runs}] 失败")
            if process.returncode != 0:
                print(f"   退出码: {process.returncode}")
            if error_msg:
                print(f"   检测到错误: {error_msg}")
        
        return RunResult(
            run_index=run_index,
            args_str=args_str,
            success=success_flag,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            task_num=output_info.get('task_num'),
            total_tasks=output_info.get('total_tasks'),
            output_file=output_info.get('output_file'),
            eval_file=output_info.get('eval_file'),
            vsp_dir=output_info.get('vsp_dir'),
            error_message=error_msg or (f"退出码: {process.returncode}" if process.returncode != 0 else None),
            error_key=error_key,
            provider=args_info.get('provider'),
            model=args_info.get('model'),
            categories=args_info.get('categories'),
            max_tasks_arg=args_info.get('max_tasks_arg'),
        )
        
    except Exception as e:
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n❌ 运行 [{run_index}/{total_runs}] 异常")
        print(f"   错误: {e}")
        
        return RunResult(
            run_index=run_index,
            args_str=args_str,
            success=False,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            error_message=str(e),
            provider=args_info.get('provider'),
            model=args_info.get('model'),
            categories=args_info.get('categories'),
            max_tasks_arg=args_info.get('max_tasks_arg'),
        )


def print_results_summary(results: List[RunResult], batch_start: datetime, batch_end: datetime, stop_reason: Optional[str] = None):
    """打印所有运行结果的详细汇总"""
    batch_duration = batch_end - batch_start
    
    success_count = sum(1 for r in results if r.success)
    fail_count = sum(1 for r in results if not r.success)
    
    print(f"\n{'='*100}")
    print(f"{'='*100}")
    print(f"📊 批量运行结果汇总")
    print(f"{'='*100}")
    print(f"{'='*100}")
    
    # 总体统计
    print(f"\n📈 总体统计")
    print(f"{'─'*50}")
    print(f"  开始时间:     {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结束时间:     {batch_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时:       {format_duration(batch_duration)}")
    print(f"  总运行次数:   {len(results)}")
    print(f"  成功:         {success_count}")
    print(f"  失败:         {fail_count}")
    if stop_reason:
        print(f"  停止原因:     {stop_reason}")
    
    # 每次运行的详细信息
    print(f"\n{'='*100}")
    print(f"📋 各任务详细信息")
    print(f"{'='*100}")
    
    for r in results:
        status_icon = "✅" if r.success else "❌"
        print(f"\n{status_icon} 运行 #{r.run_index}")
        print(f"{'─'*80}")
        
        # 基本信息
        print(f"  状态:         {'成功' if r.success else '失败'}")
        print(f"  耗时:         {format_duration(r.duration)}")
        print(f"  开始时间:     {r.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  结束时间:     {r.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 参数信息
        print(f"\n  📌 请求参数:")
        if r.provider:
            print(f"     Provider:    {r.provider}")
        if r.model:
            print(f"     Model:       {r.model}")
        if r.categories:
            print(f"     Categories:  {r.categories}")
        if r.max_tasks_arg:
            print(f"     Max Tasks:   {r.max_tasks_arg}")
        print(f"     完整参数:   {r.args_str}")
        
        # 输出信息
        if r.success:
            print(f"\n  📁 输出文件:")
            if r.task_num:
                print(f"     任务编号:   {r.task_num}")
            if r.total_tasks:
                print(f"     实际任务数: {r.total_tasks}")
            if r.output_file:
                print(f"     JSONL 文件: {r.output_file}")
            if r.eval_file:
                print(f"     评估结果:   {r.eval_file}")
            if r.vsp_dir:
                print(f"     VSP 目录:   {r.vsp_dir}")
        else:
            print(f"\n  ⚠️ 错误信息:")
            print(f"     {r.error_message or '未知错误'}")
    
    # 输出文件汇总表
    successful_results = [r for r in results if r.success]
    if successful_results:
        print(f"\n{'='*100}")
        print(f"📁 输出文件汇总")
        print(f"{'='*100}")
        
        # 表头
        print(f"\n  {'#':<4} {'任务编号':<8} {'Provider':<12} {'Model':<35} {'耗时':<12} {'输出文件'}")
        print(f"  {'─'*4} {'─'*8} {'─'*12} {'─'*35} {'─'*12} {'─'*50}")
        
        for r in successful_results:
            task_num_str = str(r.task_num) if r.task_num else "N/A"
            provider_str = r.provider or "N/A"
            model_str = (r.model[:32] + "...") if r.model and len(r.model) > 35 else (r.model or "N/A")
            duration_str = format_duration(r.duration) if r.duration else "N/A"
            output_str = r.output_file or "N/A"
            
            print(f"  {r.run_index:<4} {task_num_str:<8} {provider_str:<12} {model_str:<35} {duration_str:<12} {output_str}")
    
    # 失败任务汇总
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n{'='*100}")
        print(f"❌ 失败任务汇总")
        print(f"{'='*100}")
        
        for r in failed_results:
            print(f"\n  运行 #{r.run_index}:")
            print(f"    参数: {r.args_str}")
            print(f"    错误: {r.error_message or '未知错误'}")
    
    print(f"\n{'='*100}")
    print(f"🏁 批量运行完成")
    print(f"{'='*100}\n")


def generate_batch_report(results: List[RunResult], first_task_num: Optional[int] = None):
    """
    调用 generate_report_with_charts.py 生成批量结果报告
    
    Args:
        results: 所有运行结果列表
        first_task_num: 批量运行的第一个任务编号（用于报告文件命名）
    """
    # 收集所有成功的 eval 文件
    eval_files = [r.eval_file for r in results if r.success and r.eval_file]
    
    if not eval_files:
        print("⚠️  没有找到评估结果文件，跳过报告生成")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 生成批量结果报告")
    print(f"{'='*80}")
    print(f"找到 {len(eval_files)} 个评估结果文件:")
    for f in eval_files:
        print(f"  - {f}")
    print()
    
    try:
        # 设置环境变量，禁用输出缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # 确定输出报告文件名
        if first_task_num is not None:
            report_output = f"output/batch_{first_task_num}_evaluation_report.html"
        else:
            report_output = "output/evaluation_report.html"
        
        # 构建命令，传递指定的评估文件
        files_arg = ' '.join(f'"{f}"' for f in eval_files)
        cmd = f'python generate_report_with_charts.py --files {files_arg} --output "{report_output}"'
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # 实时输出
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ 报告生成完成")
            print(f"📄 HTML 报告: {report_output}")
        else:
            print(f"\n⚠️  报告生成失败，退出码: {process.returncode}")
            
    except Exception as e:
        print(f"\n❌ 生成报告时发生错误: {e}")


def main():
    """主函数"""
    batch_start = datetime.now()
    timestamp_str = batch_start.strftime('%Y-%m-%d_%H-%M-%S')
    
    # 生成所有参数组合
    combinations = generate_combinations(args_combo)
    total_runs = len(combinations)
    
    # 创建临时日志文件（稍后会重命名）
    temp_log_path = f"output/batch_temp_{timestamp_str}.log"
    log_file = setup_logging(temp_log_path)
    
    try:
        print(f"\n{'='*80}")
        print(f"🔧 批量运行 request.py")
        print(f"{'='*80}")
        print(f"开始时间: {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行次数: {total_runs}")
        print(f"{'='*80}\n")
        
        # 显示所有组合
        print("📋 将运行以下组合:")
        for i, combo in enumerate(combinations, 1):
            print(f"   [{i}] {combo}")
        print()
        
        # 运行每个组合
        results: List[RunResult] = []
        stop_reason: Optional[str] = None
        for i, args_str in enumerate(combinations, 1):
            result = run_request(args_str, i, total_runs)
            results.append(result)
        
        # 记录结束时间
        batch_end = datetime.now()
        
        # 打印详细汇总
        print_results_summary(results, batch_start, batch_end, stop_reason)
        
        # 获取第一个成功任务的 task_num（用于日志文件和报告命名）
        first_task_num = None
        for r in results:
            if r.task_num is not None:
                first_task_num = r.task_num
                break
        
        # 生成批量结果报告
        if GENERATE_REPORT:
            generate_batch_report(results, first_task_num)
        
        # 关闭日志文件
        close_logging()
        
        # 重命名日志文件为最终名称
        if first_task_num is not None:
            final_log_name = f"batch-{first_task_num}_{total_runs}_{timestamp_str}.log"
        else:
            final_log_name = f"batch-0_{total_runs}_{timestamp_str}.log"
        
        final_log_path = f"output/{final_log_name}"
        
        if os.path.exists(temp_log_path):
            os.rename(temp_log_path, final_log_path)
            print(f"\n📝 日志已保存: {final_log_path}")
        
        # 返回退出码
        fail_count = sum(1 for r in results if not r.success)
        # 若内部自动停止则使用特殊退出码 2，便于上层监控
        if stop_reason:
            sys.exit(2)
        sys.exit(0 if fail_count == 0 else 1)
        
    except Exception as e:
        # 确保异常时也关闭日志
        close_logging()
        print(f"\n❌ 批量运行发生异常: {e}")
        raise


if __name__ == "__main__":
    main()

