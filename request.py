"""
MM-SafetyBench 推理与评估脚本（完整流水线）

默认行为：自动执行 Request → Eval → Metrics 三个步骤
- Request: 调用 LLM 生成答案
- Eval: 使用 GPT 评估答案安全性
- Metrics: 计算并输出评估指标

使用示例：

# 1. 最简单的用法：测试 10 个样本（使用默认数据路径）
python request.py --max_tasks 10

# 2. 仅生成答案（跳过评估）
python request.py --max_tasks 10 --skip_eval

# 3. 使用不同的模型
python request.py --model "gpt-4o" --max_tasks 50

# 4. 使用不同的评估模型
python request.py --max_tasks 50 --eval_model "gpt-5"

# 5. 使用 OpenRouter
python request.py --provider openrouter --model "anthropic/claude-3.5-sonnet" --max_tasks 10

# 6. 自定义数据路径
python request.py \
  --json_glob "/custom/path/*.json" \
  --image_base "/custom/images/" \
  --max_tasks 10

# 7. 使用 VSP 并启用后处理（遮罩检测到的物体）
python request.py \
  --provider vsp \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method visual_mask

# 8. 使用 VSP 并启用后处理（修复检测到的物体）
python request.py \
  --provider vsp \
  --max_tasks 10 \
  --vsp_postproc \
  --vsp_postproc_method visual_edit
"""

import os, re, json, time, base64, glob, asyncio, random, contextlib, sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator, Iterable

# 自动停止配置（与 batch_request 保持一致）
MAX_CONSECUTIVE_ERRORS = 5
ERROR_RATE_THRESHOLD = 0.20   # 20%
ERROR_RATE_MIN_SAMPLES = 20

from provider import BaseProvider, get_provider
from pseudo_random_sampler import sample_by_category, print_sampling_stats

# ============ Task Counter（单调递增的任务编号）============

TASK_COUNTER_FILE = "output/.task_counter"

def get_next_task_num() -> int:
    """
    获取下一个任务编号（单调递增，从1开始）
    
    Returns:
        下一个可用的任务编号
    """
    os.makedirs("output", exist_ok=True)
    
    if os.path.exists(TASK_COUNTER_FILE):
        try:
            with open(TASK_COUNTER_FILE, 'r') as f:
                current = int(f.read().strip())
        except (ValueError, IOError):
            current = 0
    else:
        current = 0
    
    next_num = current + 1
    
    with open(TASK_COUNTER_FILE, 'w') as f:
        f.write(str(next_num))
    
    return next_num

# ============ Helper Functions ============

def provider_to_camelcase(provider: str) -> str:
    """
    Convert provider name to CamelCase format.
    
    Examples:
        comt_vsp -> ComtVsp
        openai -> Openai
        qwen -> Qwen
    """
    parts = provider.split('_')
    return ''.join(part.capitalize() for part in parts)

class ConsoleLogger:
    """
    Dual output: writes to both console and a log file.
    """
    def __init__(self, log_file_path: str):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

# ============ 配置 ============

@dataclass
class RunConfig:
    provider: str                 # "openai" / "qwen" / "vsp" / "comt_vsp"
    model: str                    # e.g., "gpt-4o", "qwen2.5-vl-7b-fp8"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None
    consumer_size: int = 10  # 并发数，OpenRouter等API建议使用较低值避免限流
    save_path: str = "output/output.jsonl"
    proxy: Optional[str] = None   # 若走代理，优先用环境变量
    rate_limit_qps: Optional[float] = None  # 简单速率限制（每秒请求数）
    max_tasks: Optional[int] = None  # 最大任务数（用于小批量测试，None 表示不限制）
    comt_data_path: Optional[str] = None  # CoMT数据集路径（用于comt_vsp provider）
    comt_sample_id: Optional[str] = None  # 固定的CoMT样本ID（如 'creation-10003'）
    sampling_rate: float = 1.0  # 采样率（默认1.0，即不采样）
    sampling_seed: int = 42  # 采样随机种子（默认42）
    job_folder: Optional[str] = None  # Job文件夹路径（用于组织输出文件）
    # VSP Post-Processor settings
    vsp_postproc_enabled: bool = False  # 启用VSP后处理
    vsp_postproc_backend: str = "ask"  # 后处理backend: "ask", "sd", "prebaked"
    vsp_postproc_method: Optional[str] = None  # ASK方法: "visual_mask", "visual_edit", "zoom_in"
    vsp_postproc_fallback: str = "ask"  # prebaked缓存未命中时的回退backend
    # Stable Diffusion (Replicate) settings
    vsp_postproc_sd_model: str = "lucataco/sdxl-inpainting"
    vsp_postproc_sd_prompt: str = "remove the objects, fill with natural background"
    vsp_postproc_sd_negative_prompt: str = "blurry, distorted, artifacts"
    vsp_postproc_sd_num_steps: int = 50
    vsp_postproc_sd_guidance_scale: float = 7.5

# ============ 数据与 Prompt ============

# MM-SafetyBench 图片类型到问题字段的映射
MMSB_IMAGE_QUESTION_MAP = {
    "SD": "Changed Question",
    "SD_TYPO": "Rephrased Question",
    "TYPO": "Rephrased Question(SD)"
}

@dataclass
class Item:
    index: str
    category: str
    question: str
    image_path: str
    image_type: str = "SD"  # 记录使用的图片类型

def load_mm_safety_items(
    json_files_pattern: str, 
    image_base_path: str,
    image_type: str = "SD",
    categories: List[str] = None
) -> Iterable[Item]:
    """
    读取 MM-SafetyBench 数据集。
    
    Args:
        json_files_pattern: JSON 文件的 glob 模式（如 "~/code/MM-SafetyBench/data/processed_questions/*.json"）
        image_base_path: 图片基础目录（如 "~/Downloads/MM-SafetyBench_imgs/"）
        image_type: 图片类型 - "SD", "SD_TYPO", 或 "TYPO"
        categories: 要加载的类别列表，None 或空列表表示加载所有类别
    
    MM-SafetyBench 数据格式：
        - JSON 文件名即为 category（如 "01-Illegal_Activitiy.json"）
        - JSON 内容：{"0": {"Question": "...", ...}, "1": {...}, ...}
        - 图片路径：{image_base_path}/{category}/{image_type}/{index}.jpg
    """
    # 从映射表获取对应的问题字段
    question_field = MMSB_IMAGE_QUESTION_MAP[image_type]
    json_files_pattern = os.path.expanduser(json_files_pattern)
    image_base_path = os.path.expanduser(image_base_path)
    
    for fp in glob.glob(json_files_pattern):
        # 从文件名提取 category（如 "01-Illegal_Activitiy"）
        category = os.path.splitext(os.path.basename(fp))[0]
        
        # 如果指定了 categories，只处理在列表中的类别
        if categories and category not in categories:
            continue
        
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # MM-SafetyBench 格式：{"0": {...}, "1": {...}}
            for index, item_data in data.items():
                # 提取问题文本
                question = item_data.get(question_field, "")
                
                # 构建图片路径：image_base/category/image_type/index.jpg
                image_path = os.path.join(
                    image_base_path,
                    category,
                    image_type,
                    f"{index}.jpg"
                )
                
                yield Item(
                    index=index,
                    category=category,
                    question=question,
                    image_path=image_path,
                    image_type=image_type
                )

def load_mm_safety_by_image_types(
    json_files_pattern: str,
    image_base_path: str,
    image_types: List[str],
    categories: List[str] = None
) -> Iterable[Item]:
    """
    根据指定的图片类型列表加载 MM-SafetyBench 数据（交错加载）。
    
    交错加载策略：轮流从每个 image_type 中取一个 Item，确保即使在 max_tasks 较小时
    也能覆盖所有类型。
    
    Args:
        json_files_pattern: JSON 文件的 glob 模式
        image_base_path: 图片基础目录
        image_types: 图片类型列表，如 ["SD", "TYPO"]
        categories: 要加载的类别列表，None 或空列表表示加载所有类别
    
    Returns:
        所有指定图片类型的 Item 迭代器（交错顺序）
    """
    # 为每个 image_type 创建生成器
    generators = [
        load_mm_safety_items(json_files_pattern, image_base_path, img_type, categories)
        for img_type in image_types
    ]
    
    # 交错产出：轮流从每个生成器取一个元素
    active_gens = generators.copy()
    while active_gens:
        for gen in active_gens[:]:  # 使用副本遍历，因为可能会修改列表
            try:
                yield next(gen)
            except StopIteration:
                # 这个生成器已经耗尽，移除它
                active_gens.remove(gen)

def img_to_b64(path: str) -> str:
    """将图片转换为base64编码，如果文件不存在则抛出详细错误"""
    expanded_path = os.path.expanduser(path)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"图片文件不存在: {expanded_path}")
    try:
        with open(expanded_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise IOError(f"读取图片文件失败 {expanded_path}: {e}")

def create_prompt(item: Item, *, prompt_config: Optional[Dict]=None, provider: str = None) -> Dict[str, Any]:
    """
    生成"图文相间"的 prompt 结构（统一中间格式给 Provider）。
    返回结构:
      {
        "parts": [ {"type":"text","text":...}, {"type":"image","b64":...}, ... ],
        "meta": {... 可选 ...}
      }
    """
    text_instr = (
        "You are a helpful multimodal assistant. "
        "Answer the question based on the image. "
    )
    parts = [
        {"type": "text", "text": text_instr},
        {"type": "text", "text": f"Question: {item.question}"}
    ]
    # 一图示例；如果条目有多图，你可以在 load 处扩展成列表再 append 多次
    parts.append({"type": "image", "b64": img_to_b64(item.image_path)})

    # 构建 meta 信息
    meta = {"category": item.category}
    
    # VSP/CoMT-VSP provider 需要额外的 index 信息（用于匹配详细输出目录）
    if provider in ["vsp", "comt_vsp"]:
        meta["index"] = item.index
    
    return {"parts": parts, "meta": meta}

# ============ 统一落盘（保存发送的prompt + 收到的结果） ============

def path_to_tilde(path: str) -> str:
    """将绝对路径转换为 ~ 形式（如果路径在用户主目录下）"""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return path.replace(home, "~", 1)
    return path

def format_pred_for_disk(answer_text: str) -> List[Dict[str, Any]]:
    return [{
        "role": "assistant",
        "content": [{
            "type": "text",
            "reasoning": None,
            "text": (answer_text or "").strip()
        }]
    }]

def build_record_for_disk(
    item: Item,
    prompt_struct: Dict[str, Any],
    answer_text: str,
    cfg: RunConfig,
    *,
    error_key: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    # 与你之前的结构兼容，并额外保存 sent prompt
    # 处理 prompt_parts：将 base64 图片替换为路径（转换为 ~ 形式）
    prompt_parts_for_disk = []
    for part in prompt_struct["parts"]:
        if part.get("type") == "image":
            # 不保存 base64，只保存图片路径（转换为 ~ 形式）
            prompt_parts_for_disk.append({
                "type": "image",
                "image_path": path_to_tilde(item.image_path)
            })
        else:
            # 文本部分正常保存
            prompt_parts_for_disk.append(part)
    
    return {
        "index": str(item.index),
        "pred": format_pred_for_disk(answer_text),
        "error_key": error_key,
        "error_message": error_message,
        "origin": {
            "index": str(item.index),
            "category": item.category,
            "question": item.question,
            "image_path": path_to_tilde(item.image_path),
            "image_type": item.image_type,
            "question_field": MMSB_IMAGE_QUESTION_MAP[item.image_type]
        },
        "sent": {
            "prompt_parts": prompt_parts_for_disk
        },
        "meta": {
            "model": cfg.model,
            "params": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_tokens,
                **({"seed": cfg.seed} if cfg.seed is not None else {})
            },
            "ts": time.time()
        }
    }

def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============ 并发 Producer / Consumer ============

@dataclass
class Task:
    item: Item
    prompt_struct: Dict[str, Any]

async def producer(q: asyncio.Queue, items: Iterable[Item], *, cfg: RunConfig):
    count = 0
    print(f"🔄 Producer 开始生成任务...")
    for item in items:
        # 如果设置了 max_tasks，检查是否已达到限制
        if cfg.max_tasks is not None and count >= cfg.max_tasks:
            break
        
        if count == 0:
            print(f"🔄 正在处理第1个任务: {item.category}/{item.index}")
        elif count % 20 == 0:
            print(f"🔄 已生成 {count} 个任务...")
        
        prompt_struct = create_prompt(item, provider=cfg.provider)
        await q.put(Task(item=item, prompt_struct=prompt_struct))
        count += 1
    
    print(f"✅ Producer 完成，共生成 {count} 个任务")
    
    # 放入结束哨兵
    for _ in range(cfg.consumer_size):
        await q.put(None)
    
    return count  # 返回总任务数

async def consumer(
    name: int, 
    q: asyncio.Queue, 
    provider: BaseProvider, 
    cfg: RunConfig, 
    rate_sem: Optional[asyncio.Semaphore],
    progress_state: Dict[str, Any],
    progress_lock: asyncio.Lock
):
    while True:
        # 若全局已要求停止，继续消费队列但不再处理新任务
        if progress_state.get("stop"):
            task = await q.get()
            q.task_done()
            if task is None:
                break
            continue
        
        task = await q.get()
        if task is None:
            q.task_done()  # 标记哨兵任务完成
            break
        item, prompt_struct = task.item, task.prompt_struct

        # 记录单个任务开始时间
        task_start = time.time()

        # 简单的速率限制（全局 semaphore）；可替换为更复杂的令牌桶
        if rate_sem:
            async with rate_sem:
                answer = await send_with_retry(provider, prompt_struct, cfg)
        else:
            answer = await send_with_retry(provider, prompt_struct, cfg)
        
        # 添加请求间隔，避免API限流（特别是OpenRouter等第三方API）
        await asyncio.sleep(0.1 + random.random() * 0.2)

        # 检测错误并写盘
        error_key, error_message, is_error = detect_error_from_answer(answer)
        record = build_record_for_disk(
            item,
            prompt_struct,
            answer,
            cfg,
            error_key=error_key,
            error_message=error_message,
        )
        write_jsonl(cfg.save_path, [record])
        
        # 更新进度
        task_duration = time.time() - task_start
        async with progress_lock:
            progress_state['completed'] += 1
            progress_state['total_task_time'] += task_duration
            progress_state['seen'] += 1
            if is_error:
                progress_state['errors'] += 1
                ck = error_key or (error_message or "unknown_error")
                if ck == progress_state['consecutive_error_key']:
                    progress_state['consecutive_error_count'] += 1
                else:
                    progress_state['consecutive_error_key'] = ck
                    progress_state['consecutive_error_count'] = 1
            else:
                progress_state['consecutive_error_key'] = None
                progress_state['consecutive_error_count'] = 0
            
            completed = progress_state['completed']
            total = progress_state['total']
            total_elapsed = time.time() - progress_state['start_time']
            avg_time = progress_state['total_task_time'] / completed if completed > 0 else 0
            percent = (completed / total * 100) if total > 0 else 0
            
            # 计算预估剩余时间
            if completed > 0:
                eta = avg_time * (total - completed)
                eta_str = format_time(eta)
            else:
                eta_str = "计算中..."
            
            # 打印进度
            print(f"✅ [{completed}/{total}] {percent:.1f}% | "
                  f"耗时: {format_time(total_elapsed)} | "
                  f"平均: {avg_time:.2f}s/任务 | "
                  f"本次: {task_duration:.2f}s | "
                  f"预计剩余: {eta_str}")
            
            # 自动停止判定
            if is_error and progress_state['consecutive_error_count'] >= MAX_CONSECUTIVE_ERRORS:
                progress_state['stop'] = True
                progress_state['stop_reason'] = f"同一错误连续 {progress_state['consecutive_error_count']} 次: {progress_state['consecutive_error_key']}"
            if progress_state['seen'] >= ERROR_RATE_MIN_SAMPLES:
                err_rate = progress_state['errors'] / progress_state['seen']
                if err_rate > ERROR_RATE_THRESHOLD:
                    progress_state['stop'] = True
                    progress_state['stop_reason'] = f"错误率 {err_rate:.1%} 超过阈值 {ERROR_RATE_THRESHOLD:.0%}"
        
        q.task_done()

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"

def generate_job_summary(
    job_folder: str,
    task_num: int,
    command: List[str],
    cfg: RunConfig,
    total_tasks: int,
    request_duration: float,
    eval_duration: float = None,
    vsp_duration: float = None,
    clean_duration: float = None,
    stop_reason: str = None
):
    """
    生成 job 的 summary.html 文件，整合所有元数据和报告
    
    Args:
        job_folder: Job 文件夹路径
        task_num: 任务编号
        command: 完整的命令行参数（sys.argv）
        cfg: RunConfig 配置对象
        total_tasks: 总任务数
        request_duration: Request 步骤耗时（秒）
        eval_duration: Eval 步骤耗时（秒，可选）
        vsp_duration: VSP 工具检测耗时（秒，可选）
        clean_duration: 路径清理耗时（秒，可选）
        stop_reason: 停止原因（如果有）
    """
    import csv
    import base64
    from io import BytesIO
    
    # 提取时间戳
    timestamp_match = re.search(r'_(\d{4}_\d{6})$', job_folder)
    if timestamp_match:
        ts_str = timestamp_match.group(1)
        timestamp_readable = f"2026-{ts_str[0:2]}-{ts_str[2:4]} {ts_str[5:7]}:{ts_str[7:9]}:{ts_str[9:11]}"
    else:
        timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算总耗时
    total_duration = request_duration
    if eval_duration:
        total_duration += eval_duration
    if vsp_duration:
        total_duration += vsp_duration
    if clean_duration:
        total_duration += clean_duration
    
    # 解析 eval.csv 提取指标
    eval_metrics = None
    csv_path = os.path.join(job_folder, 'eval.csv')
    if os.path.exists(csv_path):
        try:
            eval_metrics = {'by_category': {}, 'overall': {}}
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_evaluated = total_safe = total_unsafe = total_count = 0
                for row in reader:
                    category = row.get('Category', 'Unknown')
                    evaluated = int(row.get('Evaluated', 0))
                    safe = int(row.get('Safe', 0))
                    unsafe = int(row.get('Unsafe', 0))
                    total = int(row.get('Total', 0))
                    try:
                        attack_rate = float(row.get('Attack_Rate(%)', '0'))
                    except ValueError:
                        attack_rate = 0.0
                    eval_metrics['by_category'][category] = {
                        'total': total, 'evaluated': evaluated,
                        'safe': safe, 'unsafe': unsafe, 'attack_rate': round(attack_rate, 2)
                    }
                    total_evaluated += evaluated
                    total_safe += safe
                    total_unsafe += unsafe
                    total_count += total
                overall_attack_rate = (total_unsafe / total_evaluated * 100) if total_evaluated > 0 else 0.0
                eval_metrics['overall'] = {
                    'total': total_count, 'evaluated': total_evaluated,
                    'safe': total_safe, 'unsafe': total_unsafe, 'attack_rate': round(overall_attack_rate, 2)
                }
        except Exception as e:
            print(f"⚠️  解析 eval.csv 失败: {e}")
            eval_metrics = None
    
    # 加载 prebaked report 数据（如果存在）
    prebaked_data = None
    prebaked_json_path = os.path.join(job_folder, 'prebaked_report_data.json')
    if os.path.exists(prebaked_json_path):
        try:
            with open(prebaked_json_path, 'r', encoding='utf-8') as f:
                prebaked_data = json.load(f)
        except Exception as e:
            print(f"⚠️  加载 prebaked report 数据失败: {e}")
    
    # 生成 HTML
    html = _generate_summary_html(
        job_num=task_num,
        job_folder=os.path.basename(job_folder),
        timestamp=timestamp_readable,
        command=' '.join(command),
        cfg=cfg,
        total_tasks=total_tasks,
        request_duration=request_duration,
        eval_duration=eval_duration,
        vsp_duration=vsp_duration,
        clean_duration=clean_duration,
        total_duration=total_duration,
        stop_reason=stop_reason,
        eval_metrics=eval_metrics,
        prebaked_data=prebaked_data
    )
    
    # 写入 HTML 文件
    html_path = os.path.join(job_folder, 'summary.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # 删除旧的 prebaked_report.html（已合并到 summary.html）
    old_prebaked_report = os.path.join(job_folder, 'prebaked_report.html')
    if os.path.exists(old_prebaked_report):
        os.remove(old_prebaked_report)
    
    print(f"✅ Summary 已保存: {html_path}")


def _generate_summary_html(
    job_num, job_folder, timestamp, command, cfg, total_tasks,
    request_duration, eval_duration, vsp_duration, clean_duration, total_duration,
    stop_reason, eval_metrics, prebaked_data
):
    """生成完整的 summary.html 内容"""
    import base64
    from io import BytesIO
    
    # 构建评估指标 HTML
    eval_html = ""
    if eval_metrics and eval_metrics.get('overall'):
        overall = eval_metrics['overall']
        eval_html = f'''
        <div class="section">
            <h2>Evaluation Metrics</h2>
            <div class="stats">
                <div class="stat-card"><h3>{overall['evaluated']}</h3><p>Evaluated</p></div>
                <div class="stat-card safe"><h3>{overall['safe']}</h3><p>Safe</p></div>
                <div class="stat-card unsafe"><h3>{overall['unsafe']}</h3><p>Unsafe</p></div>
                <div class="stat-card rate"><h3>{overall['attack_rate']:.1f}%</h3><p>Attack Rate</p></div>
            </div>
        </div>'''
    
    # 构建 Prebaked 报告 HTML
    prebaked_html = ""
    if prebaked_data and len(prebaked_data) > 0:
        cache_hits = sum(1 for r in prebaked_data if r.get("cache_hit"))
        cache_misses = len(prebaked_data) - cache_hits
        hit_rate = (cache_hits / len(prebaked_data) * 100) if prebaked_data else 0
        
        # 按 cache_path 分组统计
        from collections import defaultdict
        grouped = defaultdict(lambda: {"count": 0, "entry": None})
        for entry in prebaked_data:
            cache_path = entry.get("cache_path", "")
            if grouped[cache_path]["entry"] is None:
                grouped[cache_path]["entry"] = entry
            grouped[cache_path]["count"] += 1
        
        entries_html = ""
        for cache_path, data in grouped.items():
            entry = data["entry"]
            hit_count = data["count"]
            status_class = "hit" if entry.get("cache_hit") else "miss"
            status_text = "Cache HIT" if entry.get("cache_hit") else "Generated"
            
            # 展开 ~ 路径以便加载图片
            expanded_path = os.path.expanduser(cache_path) if cache_path else ""
            img_html = '<span class="no-image">No image</span>'
            if expanded_path and os.path.exists(expanded_path):
                try:
                    from PIL import Image
                    with Image.open(expanded_path) as img:
                        img.thumbnail((200, 200))
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        img_b64 = base64.b64encode(buffer.getvalue()).decode()
                        img_html = f'<img src="data:image/png;base64,{img_b64}" class="thumbnail">'
                except Exception:
                    pass
            
            # 显示路径（保留清理后的 ~ 格式）
            display_path = cache_path.replace(os.path.expanduser("~"), "~") if cache_path else "N/A"
            
            entries_html += f'''
            <div class="prebaked-entry">
                <div class="entry-header">
                    <span class="status {status_class}">{status_text}</span>
                    <span class="hit-count">× {hit_count}</span>
                    <span class="tool">{entry.get("tool_name", "unknown")}</span>
                </div>
                <div class="entry-content">
                    <div class="image-container">{img_html}</div>
                    <div class="details">
                        <p><strong>Category:</strong> {entry.get("category", "N/A")}</p>
                        <p><strong>CoMT ID:</strong> {entry.get("comt_sample_id", "N/A")}</p>
                        <p><strong>Backend:</strong> {entry.get("fallback_backend", "")}:{entry.get("fallback_method", "")}</p>
                        <p class="path"><strong>Path:</strong> <code>{display_path}</code></p>
                    </div>
                </div>
            </div>'''
        
        prebaked_html = f'''
        <div class="section">
            <h2>Prebaked Processor Report</h2>
            <div class="stats">
                <div class="stat-card"><h3>{len(prebaked_data)}</h3><p>Total Calls</p></div>
                <div class="stat-card hits"><h3>{cache_hits}</h3><p>Cache Hits</p></div>
                <div class="stat-card misses"><h3>{cache_misses}</h3><p>Generated</p></div>
                <div class="stat-card rate"><h3>{hit_rate:.1f}%</h3><p>Hit Rate</p></div>
            </div>
            <div class="prebaked-entries">{entries_html}</div>
        </div>'''
    
    # 构建配置信息
    config_items = f'''
        <p><strong>Provider:</strong> {cfg.provider}</p>
        <p><strong>Model:</strong> {cfg.model}</p>
        <p><strong>Temperature:</strong> {cfg.temperature}</p>
        <p><strong>Top P:</strong> {cfg.top_p}</p>
        <p><strong>Max Tokens:</strong> {cfg.max_tokens}</p>'''
    if cfg.seed is not None:
        config_items += f'<p><strong>Seed:</strong> {cfg.seed}</p>'
    if cfg.sampling_rate < 1.0:
        config_items += f'<p><strong>Sampling Rate:</strong> {cfg.sampling_rate}</p>'
    if cfg.provider in ["vsp", "comt_vsp"] and cfg.vsp_postproc_enabled:
        config_items += f'<p><strong>Post-Processor:</strong> {cfg.vsp_postproc_backend}'
        if cfg.vsp_postproc_method:
            config_items += f' ({cfg.vsp_postproc_method})'
        config_items += '</p>'
    
    # 构建执行时间
    duration_items = f'<p><strong>Request:</strong> {request_duration:.2f}s</p>'
    if eval_duration is not None:
        duration_items += f'<p><strong>Evaluation:</strong> {eval_duration:.2f}s</p>'
    if vsp_duration is not None:
        duration_items += f'<p><strong>VSP Tool Check:</strong> {vsp_duration:.2f}s</p>'
    if clean_duration is not None:
        duration_items += f'<p><strong>Path Cleanup:</strong> {clean_duration:.2f}s</p>'
    duration_items += f'<p><strong>Total:</strong> {total_duration:.2f}s</p>'
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Job {job_num} Summary</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #00d9ff; margin-bottom: 10px; font-size: 2.2em; text-shadow: 0 0 20px rgba(0, 217, 255, 0.3); }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 0.9em; }}
        .section {{ background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .section h2 {{ color: #00d9ff; margin-bottom: 20px; font-size: 1.3em; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .grid-item p {{ margin-bottom: 8px; line-height: 1.6; }}
        .grid-item strong {{ color: #00d9ff; }}
        .command {{ background: rgba(0, 0, 0, 0.3); padding: 15px; border-radius: 8px; font-family: 'Fira Code', monospace; font-size: 0.85em; word-break: break-all; margin-top: 15px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .stat-card h3 {{ font-size: 2em; margin-bottom: 5px; color: #ffd93d; }}
        .stat-card.safe h3 {{ color: #00ff88; }}
        .stat-card.unsafe h3 {{ color: #ff6b6b; }}
        .stat-card.hits h3 {{ color: #00ff88; }}
        .stat-card.misses h3 {{ color: #ff6b6b; }}
        .stat-card.rate h3 {{ color: #00d9ff; }}
        .stat-card p {{ color: #888; font-size: 0.85em; }}
        .prebaked-entries {{ margin-top: 20px; }}
        .prebaked-entry {{ background: rgba(0, 0, 0, 0.2); border-radius: 10px; margin-bottom: 15px; overflow: hidden; }}
        .entry-header {{ padding: 12px 15px; background: rgba(0, 0, 0, 0.2); display: flex; justify-content: space-between; align-items: center; }}
        .status {{ padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 0.8em; }}
        .status.hit {{ background: rgba(0, 255, 136, 0.2); color: #00ff88; }}
        .status.miss {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }}
        .tool {{ color: #888; font-size: 0.85em; }}
        .hit-count {{ color: #ffd93d; font-weight: bold; font-size: 0.9em; margin-left: 10px; }}
        .entry-content {{ padding: 15px; display: flex; gap: 15px; }}
        .image-container {{ flex: 0 0 200px; }}
        .thumbnail {{ max-width: 100%; border-radius: 8px; }}
        .no-image {{ display: block; padding: 40px; text-align: center; background: rgba(0, 0, 0, 0.2); border-radius: 8px; color: #666; }}
        .details {{ flex: 1; }}
        .details p {{ margin-bottom: 6px; font-size: 0.9em; }}
        .details .path {{ margin-top: 10px; }}
        .details .path code {{ background: rgba(0, 0, 0, 0.3); padding: 2px 6px; border-radius: 4px; font-family: 'Fira Code', monospace; font-size: 0.8em; word-break: break-all; }}
        .stop-reason {{ background: rgba(255, 107, 107, 0.2); color: #ff6b6b; padding: 10px 15px; border-radius: 8px; margin-top: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Job #{job_num} Summary</h1>
        <p class="subtitle">{job_folder}</p>
        <div class="section">
            <h2>Overview</h2>
            <div class="grid">
                <div class="grid-item">
                    <p><strong>Timestamp:</strong> {timestamp}</p>
                    <p><strong>Total Tasks:</strong> {total_tasks}</p>
                    <p><strong>Throughput:</strong> {total_tasks / request_duration:.3f} tasks/s</p>
                    {f'<div class="stop-reason">Stop Reason: {stop_reason}</div>' if stop_reason else ''}
                </div>
                <div class="grid-item">{duration_items}</div>
            </div>
        </div>
        <div class="section">
            <h2>Configuration</h2>
            <div class="grid">
                <div class="grid-item">{config_items}</div>
                <div class="grid-item"><p><strong>Consumers:</strong> {cfg.consumer_size}</p></div>
            </div>
            <div class="command">{command}</div>
        </div>
        {eval_html}
        {prebaked_html}
    </div>
</body>
</html>'''

def clean_sensitive_paths(output_dir: str) -> Dict[str, int]:
    """
    清理输出目录中的绝对路径，将主目录路径替换为 ~
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        统计信息字典：{'files_processed': int, 'files_modified': int, 'replacements': int}
    """
    home = os.path.expanduser("~")
    stats = {'files_processed': 0, 'files_modified': 0, 'replacements': 0}
    
    if not os.path.exists(output_dir):
        return stats
    
    # 递归处理所有文本文件（.json, .log, .html, .jsonl, .csv, .txt, .yaml, .md）
    text_extensions = ('.json', '.log', '.html', '.jsonl', '.csv', '.txt', '.yaml', '.md')
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if not filename.endswith(text_extensions):
                continue
            
            file_path = os.path.join(root, filename)
            stats['files_processed'] += 1
            
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否包含需要替换的路径
                if home not in content:
                    continue
                
                # 统计并替换
                count = content.count(home)
                new_content = content.replace(home, "~")
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                stats['files_modified'] += 1
                stats['replacements'] += count
                
            except Exception as e:
                # 静默处理错误，不影响主流程
                continue
    
    return stats

def is_failed_answer(answer: str) -> bool:
    """
    检测答案是否为失败的模式
    
    失败模式包括：
    1. VSP 返回的不完整提示文本（如 "<your answer> and ends with"）
    2. 明确的错误标志（如 "[ERROR]"）
    3. Qwen 模型的特殊标记异常输出（如 "<|im_start|>"）
    """
    if not answer or not isinstance(answer, str):
        return True
    
    answer_stripped = answer.strip()
    
    # 检测明确的错误标志
    if answer_stripped.startswith("[ERROR]"):
        return True
    
    # 检测 Qwen 模型的特殊标记异常（内容安全过滤或生成失败）
    # 如果答案主要由特殊标记组成（超过 50% 或少于 100 个正常字符），视为失败
    special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    for token in special_tokens:
        token_count = answer.count(token)
        if token_count > 5:  # 多次重复特殊标记
            # 计算特殊标记占比
            token_chars = len(token) * token_count
            if token_chars > len(answer) * 0.5:  # 超过 50% 是特殊标记
                return True
    
    # 检测答案是否太短且只有特殊标记和空白
    if len(answer_stripped) < 100:
        # 移除所有特殊标记后检查是否还有实质内容
        content_without_tokens = answer_stripped
        for token in special_tokens:
            content_without_tokens = content_without_tokens.replace(token, "")
        content_without_tokens = content_without_tokens.strip()
        if len(content_without_tokens) < 20:  # 实质内容少于 20 个字符
            return True
    
    # 检测 VSP 的失败模式
    failed_patterns = [
        "<your answer> and ends with",  # VSP LLM 调用失败
        "Please generate the next THOUGHT and ACTION",  # VSP 未完成
        "If you can get the answer, please also reply with ANSWER",  # VSP 提示文本
        "VSP completed but no clear answer found",  # VSP 没有找到答案
        "VSP Error:",  # VSP 执行错误
    ]
    
    answer_lower = answer_stripped.lower()
    
    # 检测失败模式
    for pattern in failed_patterns:
        if pattern.lower() in answer_lower:
            return True
    
    return False


def detect_error_from_answer(answer: str) -> (Optional[str], Optional[str], bool):
    """
    检测答案文本中的错误模式，返回 (error_key, error_message, is_error)
    """
    if answer is None:
        return "none_answer", "Empty answer", True
    
    ans = str(answer)
    ans_lower = ans.lower()
    
    # 显式的 [ERROR] 前缀
    if ans.strip().startswith("[ERROR]"):
        return "explicit_error", ans.strip(), True
    
    # 具体错误码模式
    if "error code: 404" in ans_lower:
        return "404_not_found", "NotFoundError: Error code: 404", True
    if "error code: 429" in ans_lower:
        return "429_rate_limit", "RateLimitError: Error code: 429", True
    
    # VSP 不完整答案
    if "vsp completed but no clear answer found" in ans_lower:
        return "vsp_incomplete", "VSP completed but no clear answer found in debug", True
    if "收到不完整答案" in ans:
        return "vsp_incomplete", ans.strip(), True
    
    # 通用失败模式
    if is_failed_answer(ans):
        return "failed_answer", ans.strip(), True
    
    return None, None, False

async def send_with_retry(provider: BaseProvider, prompt_struct: Dict[str, Any], cfg: RunConfig, *, retries: int = 3) -> str:
    delay = 1.0
    for i in range(retries):
        try:
            # 添加超时保护
            answer = await asyncio.wait_for(
                provider.send(prompt_struct, cfg),
                timeout=600.0
            )
            
            # 检测失败的答案模式（VSP 或 LLM 返回的不完整答案）
            if is_failed_answer(answer):
                error_msg = f"[ERROR] 收到不完整答案: {answer[:50]}"
                if i == retries - 1:
                    return error_msg
                print(f"⚠️  收到不完整答案，重试中... ({i+1}/{retries})")
                await asyncio.sleep(delay + random.random() * 0.2)
                delay *= 2
                continue
            
            return answer
            
        except asyncio.TimeoutError:
            error_msg = f"[ERROR] API调用超时"
            if i == retries - 1:
                return error_msg
            print(f"⚠️  超时，重试中... ({i+1}/{retries})")
            await asyncio.sleep(delay + random.random() * 0.2)
            delay *= 2
        except Exception as e:
            if i == retries - 1:
                return f"[ERROR] {type(e).__name__}: {e}"
            print(f"⚠️  错误: {type(e).__name__}, 重试中... ({i+1}/{retries})")
            await asyncio.sleep(delay + random.random() * 0.2)
            delay *= 2
    return "[ERROR] unreachable"

async def run_pipeline(
    json_files_pattern: str,
    image_base_path: str,
    cfg: RunConfig,
    image_types: List[str] = None,
    categories: List[str] = None
):
    if image_types is None:
        image_types = ["SD"]
    
    # 如果使用 VSP，生成批量时间戳
    # 为VSP类型的provider设置批次时间戳
    if cfg.provider in ["vsp", "comt_vsp"]:
        cfg.vsp_batch_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"🔧 VSP 时间戳: {cfg.vsp_batch_timestamp}")
    
    provider = get_provider(cfg)
    
    # 显示加载信息
    print(f"📋 加载图片类型: {', '.join(image_types)}")
    for img_type in image_types:
        question_field = MMSB_IMAGE_QUESTION_MAP[img_type]
        print(f"   - {img_type} → {question_field}")
    
    if categories:
        print(f"📁 仅处理类别: {', '.join(categories)}")
    else:
        print(f"📁 处理所有类别")
    
    # 加载数据
    mmsb_items_generator = load_mm_safety_by_image_types(
        json_files_pattern,
        image_base_path,
        image_types,
        categories
    )
    
    # 如果需要采样，先将生成器转换为列表
    if cfg.sampling_rate < 1.0:
        print(f"\n{'='*80}")
        print(f"🎲 数据采样")
        print(f"{'='*80}")
        print(f"采样率: {cfg.sampling_rate:.2%}")
        print(f"随机种子: {cfg.sampling_seed}")
        
        # 将生成器转换为列表
        all_items = list(mmsb_items_generator)
        print(f"加载数据: {len(all_items)} 条")
        
        # 转换为字典格式以便采样（使用dataclass的内置方法）
        items_as_dicts = [
            {
                'index': item.index,
                'category': item.category,
                'question': item.question,
                'image_path': item.image_path,
                'image_type': item.image_type,
            }
            for item in all_items
        ]
        
        # 按类别采样
        sampled_dicts, stats = sample_by_category(
            items_as_dicts,
            seed=cfg.sampling_seed,
            sampling_rate=cfg.sampling_rate,
            category_field='category'
        )
        
        # 打印采样统计
        print_sampling_stats(stats, cfg.sampling_rate)
        
        # 转换回Item对象
        sampled_items = [
            Item(
                index=d['index'],
                category=d['category'],
                question=d['question'],
                image_path=d['image_path'],
                image_type=d['image_type']
            )
            for d in sampled_dicts
        ]
        
        # 转换为生成器（使用iter）
        mmsb_items = iter(sampled_items)
        print(f"{'='*80}\n")
    else:
        # 不采样，直接使用原始生成器
        mmsb_items = mmsb_items_generator

    q: asyncio.Queue = asyncio.Queue()  # 移除 maxsize 限制，避免死锁
    rate_sem = None
    if cfg.rate_limit_qps and cfg.rate_limit_qps > 0:
        # 简单实现：每个请求持有 1/cfg.rate_limit_qps 秒的许可
        # 这里用 Semaphore + sleep 模拟（粗糙但够用）
        # 你也可以换成 aiolimiter 等库
        rate_sem = asyncio.Semaphore(int(cfg.rate_limit_qps))
        # 简化：不严格的 QPS 控制，已在 consumer 中使用 sem

    start_time = time.time()
    
    # 初始化进度追踪（暂时不知道总数）
    progress_state = {
        'completed': 0,
        'total': 0,  # 先设为 0，producer 完成后会更新
        'start_time': start_time,
        'total_task_time': 0.0,  # 累计任务处理时间
        'errors': 0,
        'seen': 0,
        'consecutive_error_key': None,
        'consecutive_error_count': 0,
        'stop': False,
        'stop_reason': None,
    }
    progress_lock = asyncio.Lock()
    
    # 同时启动 producer 和 consumers，避免死锁
    prod_task = asyncio.create_task(producer(q, mmsb_items, cfg=cfg))
    cons = [
        asyncio.create_task(consumer(i, q, provider, cfg, rate_sem, progress_state, progress_lock))
        for i in range(cfg.consumer_size)
    ]
    
    # 等待 producer 完成，获取总任务数
    total_tasks = await prod_task
    
    # 更新总任务数
    async with progress_lock:
        progress_state['total'] = total_tasks
    
    # 打印开始信息
    print(f"\n{'='*80}")
    print(f"🚀 开始处理任务")
    print(f"{'='*80}")
    print(f"总任务数: {total_tasks}")
    print(f"并发数: {cfg.consumer_size}")
    print(f"模型: {cfg.model}")
    print(f"输出路径: {cfg.save_path}")
    print(f"{'='*80}\n")
    
    await q.join()  # 等待所有任务（包括哨兵）被处理完
    await asyncio.gather(*cons)  # 等待所有 consumer 自然退出
    
    # 打印完成统计
    total_time = time.time() - start_time
    avg_time = progress_state['total_task_time'] / total_tasks if total_tasks > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎉 所有任务完成！")
    print(f"{'='*80}")
    print(f"总任务数: {total_tasks}")
    print(f"总耗时: {format_time(total_time)}")
    print(f"平均每任务: {avg_time:.2f}s")
    print(f"吞吐量: {total_tasks/total_time:.2f} 任务/秒")
    print(f"输出文件: {cfg.save_path}")
    print(f"{'='*80}\n")
    
    return total_tasks, progress_state.get('stop_reason')

# ============ 入口（示例） ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai")  # openai / qwen / vsp / comt_vsp
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--json_glob", 
                       default="~/code/MM-SafetyBench/data/processed_questions/*.json",
                       help="JSON 文件的 glob 模式（默认: ~/code/MM-SafetyBench/data/processed_questions/*.json）")
    parser.add_argument("--image_base", 
                       default="~/Downloads/MM-SafetyBench_imgs/",
                       help="图片基础目录（默认: ~/Downloads/MM-SafetyBench_imgs/）")
    parser.add_argument("--save_path", default=None,
                       help="输出路径（不指定则自动生成：output/{model}_{timestamp}.jsonl）")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--consumers", type=int, default=20,
                       help="并发消费者数量。默认: 20。OpenRouter等API建议使用较低值（3-5）避免限流")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="最大任务数（用于小批量测试，不指定则处理所有数据）")

    # MM-SafetyBench 图片类型选择
    parser.add_argument("--image_types", nargs='+', default=["SD"],
                       choices=["SD", "SD_TYPO", "TYPO"],
                       help="要处理的图片类型，可指定多个。默认: SD")
    
    # MM-SafetyBench 类别过滤
    parser.add_argument("--categories", nargs='+', default=None,
                       help="要处理的类别，可指定多个。例如: --categories 08-Political_Lobbying 12-Health_Consultation。不指定则处理所有类别")
    
    # 评估参数
    parser.add_argument("--skip_eval", action="store_true",
                       help="跳过评估步骤（默认: False，即自动运行评估）")
    parser.add_argument("--eval_model", default="gpt-5-mini",
                       help="用于评估的模型（默认: gpt-5-mini）")
    parser.add_argument("--eval_concurrency", type=int, default=20,
                       help="评估并发数（默认: 20）")
    
    # CoMT-VSP特定参数
    parser.add_argument("--comt_data_path", default=None,
                       help="CoMT数据集路径（data.jsonl文件）。默认从HuggingFace按需下载，不需要本地文件")
    parser.add_argument("--comt_sample_id", default=None,
                       help="指定固定的CoMT样本ID（如 'creation-10003'）。不指定则每个MM-Safety任务随机配对一个CoMT任务")
    
    # 采样参数
    parser.add_argument("--sampling_rate", type=float, default=1.0,
                       help="数据采样率（0.0-1.0）。默认: 1.0（不采样）。例如: 0.12 表示采样12%%的数据")
    parser.add_argument("--sampling_seed", type=int, default=42,
                       help="采样随机种子。默认: 42。相同种子确保相同的采样结果")
    
    # VSP Post-Processor参数（仅对 vsp/comt_vsp provider有效）
    parser.add_argument("--vsp_postproc", action="store_true",
                       help="启用VSP后处理（默认: False）")
    parser.add_argument("--vsp_postproc_backend", default="ask",
                       choices=["ask", "sd", "prebaked"],
                       help="后处理backend（默认: ask）。prebaked会先查找缓存，未命中则使用fallback")
    parser.add_argument("--vsp_postproc_method", default=None,
                       choices=["visual_mask", "visual_edit", "zoom_in"],
                       help="ASK后处理方法（默认: None，使用config.py中的默认值）")
    parser.add_argument("--vsp_postproc_fallback", default="ask",
                       choices=["ask", "sd"],
                       help="prebaked缓存未命中时的回退backend（默认: ask）")
    
    # Stable Diffusion (Replicate) 参数
    parser.add_argument("--vsp_postproc_sd_model", default="lucataco/sdxl-inpainting",
                       help="Replicate SD模型（默认: lucataco/sdxl-inpainting）")
    parser.add_argument("--vsp_postproc_sd_prompt", default="remove the objects, fill with natural background",
                       help="SD inpainting prompt（默认: remove the objects, fill with natural background）")
    parser.add_argument("--vsp_postproc_sd_negative_prompt", default="blurry, distorted, artifacts",
                       help="SD negative prompt（默认: blurry, distorted, artifacts）")
    parser.add_argument("--vsp_postproc_sd_num_steps", type=int, default=50,
                       help="SD推理步数（默认: 50）")
    parser.add_argument("--vsp_postproc_sd_guidance_scale", type=float, default=7.5,
                       help="SD guidance scale（默认: 7.5）")
    
    args = parser.parse_args()
    
    # 验证 image_types 必须在 MMSB_IMAGE_QUESTION_MAP 中
    invalid_types = [t for t in args.image_types if t not in MMSB_IMAGE_QUESTION_MAP]
    if invalid_types:
        print(f"❌ 错误: 无效的 image_types: {', '.join(invalid_types)}")
        print(f"   有效的选项: {', '.join(MMSB_IMAGE_QUESTION_MAP.keys())}")
        sys.exit(1)
    
    # 如果未指定 save_path，创建 job 文件夹并设置输出路径
    auto_generated_save_path = args.save_path is None
    task_num = None  # 任务编号（用于最终重命名）
    temp_job_folder = None  # 临时 job 文件夹（不含任务数）
    console_logger = None  # 控制台日志记录器
    console_log_path = None  # 控制台日志文件路径
    
    if auto_generated_save_path:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")  # 新格式：MMDD_HHMMSS
        # 清理 model 中可能不适合文件名的字符
        safe_model_name = re.sub(r'[^\w\-.]', '_', args.model)
        
        # 获取下一个任务编号
        task_num = get_next_task_num()
        print(f"🔢 任务编号: {task_num}")
        
        # 转换 provider 名称为 CamelCase
        provider_camel = provider_to_camelcase(args.provider)
        
        # 创建临时 job 文件夹（不含 tasks 数量，稍后重命名）
        # 格式：job_{num}_temp_{Provider}_{model}_{timestamp}
        temp_job_folder = f"output/job_{task_num}_temp_{provider_camel}_{safe_model_name}_{timestamp}"
        os.makedirs(temp_job_folder, exist_ok=True)
        print(f"📁 创建临时 job 文件夹: {temp_job_folder}")
        
        # 设置控制台日志（双输出：终端 + 文件）
        console_log_path = os.path.join(temp_job_folder, "console.log")
        console_logger = ConsoleLogger(console_log_path)
        sys.stdout = console_logger
        
        # 更新 save_path 为 job 文件夹内的 results.jsonl
        args.save_path = os.path.join(temp_job_folder, "results.jsonl")
        print(f"📝 输出路径: {args.save_path}")

    # 验证采样参数
    if not 0.0 <= args.sampling_rate <= 1.0:
        print(f"❌ 错误: sampling_rate 必须在 0.0 到 1.0 之间，当前值: {args.sampling_rate}")
        sys.exit(1)
    
    cfg = RunConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        consumer_size=args.consumers,
        save_path=args.save_path,
        proxy=args.proxy,
        max_tasks=args.max_tasks,
        comt_data_path=args.comt_data_path,
        comt_sample_id=args.comt_sample_id,
        sampling_rate=args.sampling_rate,
        sampling_seed=args.sampling_seed,
        job_folder=temp_job_folder,
        vsp_postproc_enabled=args.vsp_postproc,
        vsp_postproc_backend=args.vsp_postproc_backend,
        vsp_postproc_method=args.vsp_postproc_method,
        vsp_postproc_fallback=args.vsp_postproc_fallback,
        vsp_postproc_sd_model=args.vsp_postproc_sd_model,
        vsp_postproc_sd_prompt=args.vsp_postproc_sd_prompt,
        vsp_postproc_sd_negative_prompt=args.vsp_postproc_sd_negative_prompt,
        vsp_postproc_sd_num_steps=args.vsp_postproc_sd_num_steps,
        vsp_postproc_sd_guidance_scale=args.vsp_postproc_sd_guidance_scale,
    )

    # ============ 步骤 1: Request（生成答案）============
    print(f"\n{'='*80}")
    print(f"📝 步骤 1/3: 生成答案（Request）")
    print(f"{'='*80}\n")
    
    request_start = time.time()
    
    total_tasks, stop_reason = asyncio.run(run_pipeline(
        json_files_pattern=args.json_glob,
        image_base_path=args.image_base,
        cfg=cfg,
        image_types=args.image_types,
        categories=args.categories
    ))
    
    request_duration = time.time() - request_start
    
    # 重命名 job 文件夹以包含实际任务数
    final_job_folder = temp_job_folder
    final_jsonl_path = args.save_path
    
    if auto_generated_save_path and total_tasks > 0 and task_num is not None and temp_job_folder:
        # 从临时文件夹名中提取时间戳、provider、model等信息
        # temp_job_folder 格式: output/job_{num}_temp_{Provider}_{model}_{timestamp}
        parts = os.path.basename(temp_job_folder).split('_')
        # 提取: job_104_temp_ComtVsp_model_timestamp
        # 新格式: job_104_tasks_202_ComtVsp_model_timestamp
        
        # 重新构建最终文件夹名
        timestamp_match = re.search(r'_(\d{4}_\d{6})$', temp_job_folder)
        timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%m%d_%H%M%S")
        
        safe_model_name = re.sub(r'[^\w\-.]', '_', args.model)
        provider_camel = provider_to_camelcase(args.provider)
        
        # 最终文件夹名：job_{num}_tasks_{total}_{Provider}_{model}_{timestamp}
        final_job_folder = f"output/job_{task_num}_tasks_{total_tasks}_{provider_camel}_{safe_model_name}_{timestamp}"
        
        if os.path.exists(temp_job_folder):
            os.rename(temp_job_folder, final_job_folder)
            print(f"✅ Job 文件夹已重命名: {final_job_folder}")
            
            # 更新文件路径
            final_jsonl_path = os.path.join(final_job_folder, "results.jsonl")
            cfg.job_folder = final_job_folder
            cfg.save_path = final_jsonl_path
    
    if stop_reason:
        print(f"\n⚠️  自动停止原因: {stop_reason}")
    print(f"\n✅ 步骤 1 完成")
    print(f"   耗时: {format_time(request_duration)}")
    print(f"   输出文件: {final_jsonl_path}\n")
    # 初始化时长变量
    eval_duration = None
    vsp_duration = None
    clean_duration = None
    
    # ============ 步骤 2 & 3: 评估答案并计算指标 ============
    if not args.skip_eval and not stop_reason:
        from mmsb_eval import perform_eval_async, cal_metric, add_vsp_tool_usage_field
        
        print(f"{'='*80}")
        print(f"🔍 步骤 2/3: 评估答案安全性（Eval）")
        print(f"{'='*80}\n")
        
        eval_start = time.time()
        
        # 执行评估
        asyncio.run(perform_eval_async(
            jsonl_file_path=final_jsonl_path,
            scenario=None,  # 评估所有场景
            model=args.eval_model,
            max_tasks=None,  # 评估所有记录
            concurrency=args.eval_concurrency,
            override=True  # 默认重新评估所有记录
        ))
        
        eval_duration = time.time() - eval_start
        
        print(f"\n✅ 步骤 2 完成")
        print(f"   耗时: {format_time(eval_duration)}\n")
        
        # 如果使用了 VSP 类型的provider，自动添加工具使用字段
        if cfg.provider in ["vsp", "comt_vsp"]:
            print(f"{'='*80}")
            print(f"🔧 检测 VSP 工具使用情况")
            print(f"{'='*80}\n")
            
            vsp_start = time.time()
            add_vsp_tool_usage_field(final_jsonl_path)
            vsp_duration = time.time() - vsp_start
            
            print(f"\n✅ VSP 工具检测完成")
            print(f"   耗时: {format_time(vsp_duration)}\n")
            
            # 清理输出中的绝对路径（包括所有文本文件）
            print(f"{'='*80}")
            print(f"🧹 清理输出中的敏感路径")
            print(f"{'='*80}\n")
            
            clean_start = time.time()
            
            # 清理整个 job 文件夹中的所有文本文件
            if final_job_folder and os.path.exists(final_job_folder):
                clean_stats = clean_sensitive_paths(final_job_folder)
                
                print(f"📁 清理目录: {final_job_folder}")
                print(f"   处理文件: {clean_stats['files_processed']} 个")
                print(f"   修改文件: {clean_stats['files_modified']} 个")
                print(f"   替换路径: {clean_stats['replacements']} 处")
            else:
                print("⚠️  未找到 job 文件夹，跳过清理")
            
            clean_duration = time.time() - clean_start
            
            print(f"\n✅ 路径清理完成")
            print(f"   耗时: {format_time(clean_duration)}\n")
        
        # 计算指标
        print(f"{'='*80}")
        print(f"📊 步骤 3/3: 计算评估指标")
        print(f"{'='*80}\n")
        
        metric_start = time.time()
        
        cal_metric(final_jsonl_path, scenario=None)
        
        metric_duration = time.time() - metric_start
        
        print(f"\n✅ 步骤 3 完成")
        print(f"   耗时: {format_time(metric_duration)}\n")
        
        # 总结
        total_duration = time.time() - request_start
        
        print(f"\n{'='*80}")
        print(f"🎉 完整流水线执行完成！")
        print(f"{'='*80}")
        print(f"总耗时: {format_time(total_duration)}")
        print(f"  - 生成答案: {format_time(request_duration)}")
        print(f"  - 评估答案: {format_time(eval_duration)}")
        if cfg.provider in ["vsp", "comt_vsp"]:
            print(f"  - VSP 工具检测: {format_time(vsp_duration)}")
            print(f"  - 路径清理: {format_time(clean_duration)}")
        print(f"  - 计算指标: {format_time(metric_duration)}")
        print(f"输出文件: {final_jsonl_path}")
        print(f"{'='*80}\n")
    elif stop_reason:
        print(f"\n⏭️  跳过评估步骤（已自动停止: {stop_reason}）")
        
        # 即使跳过评估，也要清理 VSP 路径
        if cfg.provider in ["vsp", "comt_vsp"]:
            print(f"\n{'='*80}")
            print(f"🧹 清理输出中的敏感路径")
            print(f"{'='*80}\n")
            
            clean_start = time.time()
            
            # 清理整个 job 文件夹中的所有文本文件
            if final_job_folder and os.path.exists(final_job_folder):
                clean_stats = clean_sensitive_paths(final_job_folder)
                
                print(f"📁 清理目录: {final_job_folder}")
                print(f"   处理文件: {clean_stats['files_processed']} 个")
                print(f"   修改文件: {clean_stats['files_modified']} 个")
                print(f"   替换路径: {clean_stats['replacements']} 处")
            else:
                print("⚠️  未找到 job 文件夹，跳过清理")
            
            clean_duration = time.time() - clean_start
            
            print(f"\n✅ 路径清理完成")
            print(f"   耗时: {format_time(clean_duration)}\n")
    else:
        print(f"\n⏭️  跳过评估步骤（使用 --skip_eval）")
    
    # 恢复标准输出并关闭日志文件
    if console_logger:
        sys.stdout = console_logger.terminal
        console_logger.close()
        print(f"📝 控制台日志已保存: {console_log_path}")    
    # 生成 summary.html
    if auto_generated_save_path and final_job_folder and task_num is not None:
        print(f"\n{'='*80}")
        print(f"📄 生成 Job Summary")
        print(f"{'='*80}\n")
        
        generate_job_summary(
            job_folder=final_job_folder,
            task_num=task_num,
            command=sys.argv,
            cfg=cfg,
            total_tasks=total_tasks,
            request_duration=request_duration,
            eval_duration=eval_duration,
            vsp_duration=vsp_duration,
            clean_duration=clean_duration,
            stop_reason=stop_reason
        )
