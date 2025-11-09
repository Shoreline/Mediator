import os
import json
import tempfile
import shutil
import subprocess
import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

# ============ Provider 接口与实现 ============

class BaseProvider:
    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        """输入 create_prompt 产物，返回 LLM/VSP 的**纯文本答案**。"""
        raise NotImplementedError

class OpenAIProvider(BaseProvider):
    def __init__(self):
        from openai import AsyncOpenAI # all methods in AsyncOpenAI are async
        self.client = AsyncOpenAI()

    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        parts = []
        for p in prompt_struct["parts"]:
            if p["type"] == "text":
                parts.append({"type":"input_text","text":p["text"]})
            elif p["type"] == "image":
                parts.append({"type":"input_image","image_url": f"data:image/jpeg;base64,{p['b64']}"})

        # 构建请求参数
        request_params = {
            "model": cfg.model_name,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_output_tokens": cfg.max_tokens,
            "input": [{"role":"user","content": parts}],
        }
        # 只在 seed 不为 None 时添加
        if cfg.seed is not None:
            request_params["seed"] = cfg.seed
        
        resp = await self.client.responses.create(**request_params)
        # 解析 Responses API
        txt = ""
        if getattr(resp, "output", None):
            for it in resp.output:
                if getattr(it, "type", "") == "message":
                    for c in getattr(it, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            txt += c.text or ""
        if not txt and getattr(resp, "output_text", None):
            txt = resp.output_text
        return (txt or "").strip()

class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url,
        )

    @staticmethod
    def _to_chat_blocks(prompt_struct: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = []
        for p in prompt_struct.get("parts", []):
            if p.get("type") == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image":
                mime = p.get("mime") or "image/jpeg"
                b64  = p.get("b64", "")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
        return blocks

    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        content_blocks = self._to_chat_blocks(prompt_struct)

        # 构建请求参数
        request_params = {
            "model": cfg.model_name,  # 如 "openai/gpt-4o" 或 "anthropic/claude-3.5-sonnet"
            "messages": [{"role": "user", "content": content_blocks}],
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
        }
        # 只在 seed 不为 None 时添加（并非所有模型都支持）
        if cfg.seed is not None:
            request_params["seed"] = cfg.seed
        
        resp = await self.client.chat.completions.create(**request_params)
        return (resp.choices[0].message.content or "").strip()


class QwenProvider(BaseProvider):
    """
    占位：按你现在的 Qwen 服务来改。目标是把 prompt_struct 转成你原有 HTTP payload，
    返回纯文本。若你已有异步版本，可直接移植到 send()。
    """
    def __init__(self, endpoint: str, api_key: Optional[str]=None):
        self.endpoint = endpoint
        self.api_key = api_key

    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        # TODO: 使用 aiohttp 调你自建的 Qwen 推理服务
        # 例：把 parts 转为你的接口格式（文本+base64图）
        # 返回最终文本
        raise NotImplementedError("Fill QwenProvider.send with your HTTP call")

class VSPProvider(BaseProvider):
    """
    VSP(VisualSketchpad) Provider: 通过子进程调用本地VSP工具
    
    目录结构：vsp_timestamp/category/index/
    - vsp_timestamp: 本次运行的时间戳（所有任务共享）
    - category: 任务类别（从 prompt_struct["meta"]["category"] 获取）
    - index: 任务编号（从 prompt_struct["meta"]["index"] 获取）
    """
    def __init__(self, vsp_path: str = "~/code/VisualSketchpad", 
                 output_dir: str = "output/vsp_details",
                 batch_timestamp: str = None):
        self.vsp_path = os.path.expanduser(vsp_path)
        self.agent_path = os.path.join(self.vsp_path, "agent")
        self.output_dir = output_dir  # VSP详细输出保存目录
        self.batch_timestamp = batch_timestamp  # 批量处理的时间戳
        os.makedirs(self.output_dir, exist_ok=True)
        
    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        """
        调用VSP工具处理多模态任务
        
        Args:
            prompt_struct: 包含文本和图片的结构化prompt
                - prompt_struct["meta"] 需包含 "category" 和 "index"
            cfg: 运行配置
            
        Returns:
            str: VSP的最终答案
        """
        import time
        
        # 统一使用批量模式：vsp_timestamp/category/index/
        if not self.batch_timestamp:
            raise ValueError("VSPProvider requires batch_timestamp")
        
        meta = prompt_struct.get("meta", {})
        category = meta.get("category", "unknown")
        index = meta.get("index", str(id(prompt_struct) % 10000))
        
        # 构建路径：output/vsp_details/vsp_2025-10-30_23-45-12/category/index/
        batch_root = os.path.join(self.output_dir, f"vsp_{self.batch_timestamp}")
        task_base_dir = os.path.abspath(os.path.join(batch_root, category, index))
        os.makedirs(task_base_dir, exist_ok=True)
        
        # 创建独立的input和output目录
        vsp_input_dir = os.path.join(task_base_dir, "input")  # VSP的输入
        vsp_output_dir = os.path.join(task_base_dir, "output")  # VSP的输出
        os.makedirs(vsp_input_dir, exist_ok=True)
        os.makedirs(vsp_output_dir, exist_ok=True)
        
        # 确定任务类型
        task_type = self._determine_task_type(prompt_struct)

        # 构建VSP任务输入（根据任务类型写入对应的文件格式）
        task_data = self._build_vsp_task(prompt_struct, vsp_input_dir, task_type)
        
        # 调用VSP（输出保存到vsp_output_dir）
        result = await self._call_vsp(vsp_input_dir, vsp_output_dir, task_type)
        
        # 从 debug log 中提取答案（VSP 专用方法）
        answer = self._extract_answer_vsp(vsp_output_dir)
        
        # 保存完整的VSP输出信息（供后续分析）
        self._save_vsp_metadata(task_base_dir, prompt_struct, task_data, result, answer)
        
        return answer
    
    def _build_vsp_task(self, prompt_struct: Dict[str, Any], task_dir: str, task_type: str) -> Dict[str, Any]:
        """构建VSP任务输入文件（vision任务的request.json格式）"""
        import base64
        
        # 提取文本内容和图片
        text_content = ""
        images = []
        
        for part in prompt_struct.get("parts", []):
            if part["type"] == "text":
                text_content += part["text"] + "\n"
            elif part["type"] == "image":
                images.append(part)
        
        text_content = text_content.strip()

        # 构建vision任务的request.json（使用绝对路径）
        task_data = {"query": text_content, "images": []}
        
        for i, img_part in enumerate(images):
            if img_part.get("type") == "image":
                b64_data = img_part.get("b64", "")
                if not b64_data:
                    continue
                # 直接解码base64并写入文件，不需要PIL
                image_data = base64.b64decode(b64_data)
                image_path = os.path.join(task_dir, f"image_{i}.jpg")
                with open(image_path, "wb") as f:
                    f.write(image_data)
                # 使用绝对路径（VSP支持绝对路径）
                task_data["images"].append(os.path.abspath(image_path))
        
        with open(os.path.join(task_dir, "request.json"), "w") as f:
            json.dump(task_data, f, indent=2)
            
        return task_data
    
    def _determine_task_type(self, prompt_struct: Dict[str, Any]) -> str:
        """确定任务类型，目前只支持vision"""
        return "vision"
    
    async def _call_vsp(self, task_dir: str, output_dir: str, task_type: str) -> Dict[str, Any]:
        """调用VSP工具（使用VSP自带python解释器 + run_agent 入口）"""

        # 使用相对路径的python（让shell找VSP venv的python）
        # 通过 -c 调用 run_agent，使用f-string直接嵌入参数
        py_cmd = f'from main import run_agent; run_agent("{task_dir}", "{output_dir}", task_type="{task_type}")'
        cmd = ["python", "-c", py_cmd]

        # 设置工作目录为 VSP 的 agent 目录，确保 imports 正确
        env = os.environ.copy()
        env["PYTHONPATH"] = self.agent_path
        # 激活VSP的venv
        vsp_python_bin = os.path.join(self.vsp_path, "sketchpad_env", "bin")
        env["PATH"] = f"{vsp_python_bin}:{env.get('PATH', '')}"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.agent_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            stdout_str = stdout.decode('utf-8', errors='ignore')
            stderr_str = stderr.decode('utf-8', errors='ignore')

            # 保存VSP的stdout和stderr用于调试
            debug_file = os.path.join(output_dir, "vsp_debug.log")
            with open(debug_file, "w") as f:
                f.write(f"=== VSP EXECUTION DEBUG ===\n")
                f.write(f"Return code: {process.returncode}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"\n=== STDOUT ===\n{stdout_str}\n")
                f.write(f"\n=== STDERR ===\n{stderr_str}\n")

            if process.returncode != 0:
                # 即使失败，也尝试读取部分输出
                output_file = os.path.join(output_dir, os.path.basename(task_dir), "output.json")
                if os.path.exists(output_file):
                    print(f"Warning: VSP failed but output.json exists, attempting to read...")
                    with open(output_file, "r") as f:
                        return json.load(f)
                
                raise RuntimeError(
                    f"VSP execution failed (code {process.returncode}). "
                    f"Check debug log: {debug_file}\n"
                    f"STDERR preview: {stderr_str[:500]}"
                )

            # 读取输出结果
            output_file = os.path.join(output_dir, os.path.basename(task_dir), "output.json")
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    return json.load(f)
            else:
                raise RuntimeError(f"VSP output file not found: {output_file}")

        except Exception as e:
            raise RuntimeError(f"Failed to call VSP: {str(e)}")
    
    def _save_vsp_metadata(self, output_dir: str, prompt_struct: Dict[str, Any], 
                           task_data: Dict[str, Any], vsp_result: Dict[str, Any], 
                           answer: str) -> None:
        """保存VSP执行的元数据，方便后续分析"""
        metadata = {
            "prompt_struct": prompt_struct,
            "task_data": task_data,
            "vsp_result": vsp_result,
            "extracted_answer": answer,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_file = os.path.join(output_dir, "mediator_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _extract_answer_vsp(self, vsp_output_dir: str) -> str:
        """
        从VSP的debug log中提取最终答案（VSP专用方法）
        
        VSP的答案格式：debug log 中最后一个 "ANSWER: ... TERMINATE" 块
        """
        debug_log_path = os.path.join(vsp_output_dir, "vsp_debug.log")
        
        if not os.path.exists(debug_log_path):
            return "VSP Error: debug log not found"
        
        try:
            with open(debug_log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            
            # 找到最后一个 ANSWER: 和 TERMINATE 之间的内容
            # 使用正则表达式匹配，支持多行
            import re
            
            # 首先找到最后一个 "# RESULT #:" 的位置
            result_marker = "# RESULT #:"
            last_result_idx = log_content.rfind(result_marker)
            
            # 查找所有 ANSWER: ... TERMINATE 模式
            pattern = r'ANSWER:\s*(.*?)\s*TERMINATE'
            matches = list(re.finditer(pattern, log_content, re.DOTALL))
            
            if matches:
                # 只考虑在最后一个 RESULT 之后的匹配
                valid_matches = []
                for match in matches:
                    # 如果找到了 RESULT 标记，只接受在其之后的匹配
                    if last_result_idx == -1 or match.start() > last_result_idx:
                        valid_matches.append(match)
                
                if valid_matches:
                    # 取最后一个有效匹配（最终答案）
                    last_match = valid_matches[-1]
                    answer = last_match.group(1).strip()
                    # 确保不是提示文本（如 "<your answer>"）
                    if answer and not answer.startswith('<your answer>'):
                        return answer
            
            # 如果没有找到标准格式，尝试查找最后的 ANSWER: 行
            # 但只在 "# RESULT #:" 之后查找（避免匹配提示文本中的 ANSWER）
            if last_result_idx != -1:
                # 只在最后一个 RESULT 部分中查找
                result_section = log_content[last_result_idx:]
                lines = result_section.split('\n')
                
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].startswith('ANSWER:'):
                        # 收集从 ANSWER: 开始到文件结束的所有内容
                        answer_lines = []
                        for j in range(i, len(lines)):
                            line = lines[j]
                            if j == i:
                                # 第一行，去掉 "ANSWER:" 前缀
                                answer_lines.append(line[7:].strip())
                            elif 'TERMINATE' in line:
                                # 遇到 TERMINATE，停止
                                break
                            else:
                                answer_lines.append(line)
                        answer = '\n'.join(answer_lines).strip()
                        # 确保不是提示文本（如 "<your answer> and ends with"）
                        if answer and not answer.startswith('<your answer>'):
                            return answer
            
            return "VSP completed but no clear answer found in debug log"
        
        except Exception as e:
            return f"VSP Error: Failed to read debug log: {str(e)}"

def get_provider(cfg: 'RunConfig') -> BaseProvider:
    if cfg.proxy:
        os.environ.setdefault("HTTPS_PROXY", cfg.proxy)
        os.environ.setdefault("HTTP_PROXY", cfg.proxy)

    if cfg.provider == "openai":
        return OpenAIProvider()
    elif cfg.provider == "openrouter":
        return OpenRouterProvider()
    elif cfg.provider == "qwen":
        return QwenProvider(endpoint=os.environ.get("QWEN_ENDPOINT","http://127.0.0.1:8000"),
                            api_key=os.environ.get("QWEN_API_KEY"))
    elif cfg.provider == "vsp":
        # 获取批量时间戳（必需）
        batch_timestamp = getattr(cfg, 'vsp_batch_timestamp', None)
        
        return VSPProvider(
            vsp_path=os.environ.get("VSP_PATH", "~/code/VisualSketchpad"),
            output_dir=os.environ.get("VSP_OUTPUT_DIR", "output/vsp_details"),
            batch_timestamp=batch_timestamp
        )
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")

