"""
MM-SafetyBench 推理脚本

使用示例：

# 1. 测试 10 个样本（输出文件自动命名为：output/{model_name}_{timestamp}.jsonl）
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --max_tasks 10

# 2. 测试 50 个样本，使用较少的并发，指定输出路径
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --max_tasks 50 \
  --consumers 5 \
  --save_path "test_output.jsonl"

# 3. 处理全部数据（不指定 --max_tasks 和 --save_path）
python request.py \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/"

# 4. 使用不同的模型
python request.py \
  --provider openrouter \
  --model_name "anthropic/claude-3.5-sonnet" \
  --json_glob "~/code/MM-SafetyBench/data/processed_questions/*.json" \
  --image_base "~/Downloads/MM-SafetyBench_imgs/" \
  --max_tasks 10
"""

import os, re, json, time, base64, glob, asyncio, random, contextlib, sys
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator, Iterable

from provider import BaseProvider, get_provider

# ============ 配置 ============

@dataclass
class RunConfig:
    provider: str                 # "openai" / "qwen" / "vsp"
    model_name: str               # e.g., "gpt-4o", "qwen2.5-vl-7b-fp8"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: Optional[int] = None
    consumer_size: int = 10  # 并发数，OpenRouter等API建议使用较低值避免限流
    save_path: str = "output/output.jsonl"
    proxy: Optional[str] = None   # 若走代理，优先用环境变量
    rate_limit_qps: Optional[float] = None  # 简单速率限制（每秒请求数）
    max_tasks: Optional[int] = None  # 最大任务数（用于小批量测试，None 表示不限制）

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
    
    # VSP provider 需要额外的 index 信息
    if provider == "vsp":
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

def build_record_for_disk(item: Item, prompt_struct: Dict[str, Any], answer_text: str, cfg: RunConfig) -> Dict[str, Any]:
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
            "model": cfg.model_name,
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

        record = build_record_for_disk(item, prompt_struct, answer, cfg)
        write_jsonl(cfg.save_path, [record])
        
        # 更新进度
        task_duration = time.time() - task_start
        async with progress_lock:
            progress_state['completed'] += 1
            progress_state['total_task_time'] += task_duration
            completed = progress_state['completed']
            total = progress_state['total']
            total_elapsed = time.time() - progress_state['start_time']
            avg_time = progress_state['total_task_time'] / completed
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

async def send_with_retry(provider: BaseProvider, prompt_struct: Dict[str, Any], cfg: RunConfig, *, retries: int = 3) -> str:
    delay = 1.0
    for i in range(retries):
        try:
            # 添加超时保护（120秒）
            answer = await asyncio.wait_for(
                provider.send(prompt_struct, cfg),
                timeout=120.0
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
            error_msg = f"[ERROR] API调用超时（120秒）"
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
    if cfg.provider == "vsp":
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
    mmsb_items = load_mm_safety_by_image_types(
        json_files_pattern,
        image_base_path,
        image_types,
        categories
    )

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
        'total_task_time': 0.0  # 累计任务处理时间
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
    print(f"模型: {cfg.model_name}")
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

# ============ 入口（示例） ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai")  # openai / qwen / vsp
    parser.add_argument("--model_name", default="gpt-5")
    parser.add_argument("--json_glob", required=True)
    parser.add_argument("--image_base", required=True)
    parser.add_argument("--save_path", default=None,
                       help="输出路径（不指定则自动生成：output/{model_name}_{timestamp}.jsonl）")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--consumers", type=int, default=10,
                       help="并发消费者数量。默认: 10。OpenRouter等API建议使用较低值（3-5）避免限流")
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
    
    args = parser.parse_args()
    
    # 验证 image_types 必须在 MMSB_IMAGE_QUESTION_MAP 中
    invalid_types = [t for t in args.image_types if t not in MMSB_IMAGE_QUESTION_MAP]
    if invalid_types:
        print(f"❌ 错误: 无效的 image_types: {', '.join(invalid_types)}")
        print(f"   有效的选项: {', '.join(MMSB_IMAGE_QUESTION_MAP.keys())}")
        sys.exit(1)
    
    # 如果未指定 save_path，自动生成带时间戳的文件名
    if args.save_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 清理 model_name 中可能不适合文件名的字符
        safe_model_name = re.sub(r'[^\w\-.]', '_', args.model_name)
        
        if args.provider == "vsp":
            # VSP 使用 provider 名称作为前缀，并包含模型信息
            args.save_path = f"output/vsp_{safe_model_name}_{timestamp}.jsonl"
        else:
            args.save_path = f"output/{safe_model_name}_{timestamp}.jsonl"
        print(f"📝 自动生成输出路径: {args.save_path}")

    cfg = RunConfig(
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        consumer_size=args.consumers,
        save_path=args.save_path,
        proxy=args.proxy,
        max_tasks=args.max_tasks,
    )

    asyncio.run(run_pipeline(
        json_files_pattern=args.json_glob,
        image_base_path=args.image_base,
        cfg=cfg,
        image_types=args.image_types,
        categories=args.categories
    ))
