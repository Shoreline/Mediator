import os
from typing import Any, Dict, Optional
from dataclasses import dataclass

# ============ Provider 接口与实现 ============

class BaseProvider:
    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        """输入 create_prompt 产物，返回 LLM/VSP 的**纯文本答案**。"""
        raise NotImplementedError

class OpenAIProvider(BaseProvider):
    def __init__(self):
        from openai import AsyncOpenAI
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
    VSP(VisualSketchpad) 占位：通常走 HTTP API。
    """
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def send(self, prompt_struct: Dict[str, Any], cfg: 'RunConfig') -> str:
        # TODO: 使用 aiohttp.post(self.endpoint, json=...) 调 VSP
        # 根据 VSP 的返回格式，抽取纯文本答案（或你需要的字段）
        raise NotImplementedError("Fill VSPProvider.send with your VSP call")

def get_provider(cfg: 'RunConfig') -> BaseProvider:
    if cfg.proxy:
        os.environ.setdefault("HTTPS_PROXY", cfg.proxy)
        os.environ.setdefault("HTTP_PROXY", cfg.proxy)

    if cfg.provider == "openai":
        return OpenAIProvider()
    elif cfg.provider == "qwen":
        return QwenProvider(endpoint=os.environ.get("QWEN_ENDPOINT","http://127.0.0.1:8000"),
                            api_key=os.environ.get("QWEN_API_KEY"))
    elif cfg.provider == "vsp":
        return VSPProvider(endpoint=os.environ.get("VSP_ENDPOINT","http://127.0.0.1:9000"))
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")

