"""Task 5.4.5: OpenAI 兼容 HTTP server（可选）。

设计目标
--------
提供与 OpenAI Chat Completions / Completions API 兼容的 HTTP 接口，
便于现有客户端（如 OpenAI Python SDK、各种前端 UI）直接接入。

路由
----
- ``POST /v1/chat/completions``：聊天补全（接收 messages 数组，返回 OpenAI 格式响应）
- ``POST /v1/completions``：文本补全（接收 prompt 字符串，返回 OpenAI 格式响应）
- ``GET /v1/models``：列出可用模型（返回本地模型名）

实现策略
--------
- **优先 FastAPI**：如果安装了 ``fastapi`` 与 ``uvicorn``，使用它们
  （支持异步、流式 SSE、自动文档）。
- **降级 http.server**：若 FastAPI 不可用，用标准库 ``http.server`` 提供简化版本，
  仅支持 ``POST /v1/completions`` + ``GET /v1/models``，非流式。

调用方需要提供一个 ``StreamingGenerator`` 实例，server 内部用它生成 token。

非流式响应格式（OpenAI 兼容）
-----------------------------
.. code-block:: json

    {
      "id": "cmpl-xxx",
      "object": "text_completion",
      "created": 1700000000,
      "model": "verse-mamba2",
      "choices": [
        {"text": "...", "index": 0, "finish_reason": "stop"}
      ],
      "usage": {"prompt_tokens": 4, "completion_tokens": 32, "total_tokens": 36}
    }
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

# 兼容两种执行方式：
# - 作为模块运行 ``python -m verse_inference.server`` → 包内相对导入可用
# - 作为脚本运行 ``python server.py`` → 相对导入失败，回退到绝对导入
try:
    from .generator import StreamingGenerator
    from .sampler import Sampler
except ImportError:  # pragma: no cover - 仅在直接脚本执行时触发
    from verse_inference.generator import StreamingGenerator
    from verse_inference.sampler import Sampler


# ---------------------------------------------------------------------------
# 工具：构造 OpenAI 格式响应
# ---------------------------------------------------------------------------


def _make_completion_response(
    model: str,
    prompt_text: str,
    completion_text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """构造 OpenAI /v1/completions 兼容响应字典。"""
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "text": completion_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_chat_completion_response(
    model: str,
    prompt_messages: list,
    assistant_text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """构造 OpenAI /v1/chat/completions 兼容响应字典。"""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _messages_to_prompt(messages: list) -> str:
    """把 chat messages 数组拼接成单段 prompt 文本。

    简化版：按 role 顺序拼接，每条一行：
        user: <content>
        assistant: <content>
        ...
    最后追加一行 ``assistant:`` 作为生成提示。
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# FastAPI 版本
# ---------------------------------------------------------------------------


def create_app(generator: StreamingGenerator, model_name: str = "verse-model"):
    """创建 OpenAI 兼容的 FastAPI 应用。

    Args:
        generator: 已配置好的 ``StreamingGenerator`` 实例。
        model_name: 暴露给 API 的模型名（用于 /v1/models 与响应中）。

    Returns:
        FastAPI app（需要 ``pip install fastapi uvicorn``）。
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="VerseInference OpenAI-compatible API", version="0.1.0")

    # 请求 schema
    class CompletionRequest(BaseModel):
        model: Optional[str] = model_name
        prompt: str = ""
        max_tokens: int = 100
        temperature: float = 1.0
        top_p: float = 1.0
        stream: bool = False

    class ChatCompletionRequest(BaseModel):
        model: Optional[str] = model_name
        messages: list
        max_tokens: int = 100
        temperature: float = 1.0
        top_p: float = 1.0
        stream: bool = False

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "verse",
                }
            ],
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if generator.tokenizer is None:
            raise HTTPException(500, "tokenizer not configured")
        prompt_ids = generator.tokenizer.encode(req.prompt, add_special_tokens=False)
        prompt_tokens = len(prompt_ids)

        # 重建 sampler（按请求参数）
        sampler = Sampler(temperature=req.temperature, top_p=req.top_p)
        old_sampler = generator.sampler
        generator.sampler = sampler
        try:
            tokens = []
            for tok_id in generator.generate(
                prompt_ids,
                max_new_tokens=req.max_tokens,
            ):
                tokens.append(tok_id)
        finally:
            generator.sampler = old_sampler

        completion_text = generator.tokenizer.decode(tokens)
        resp = _make_completion_response(
            model_name, req.prompt, completion_text,
            prompt_tokens, len(tokens),
        )
        return JSONResponse(resp)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if generator.tokenizer is None:
            raise HTTPException(500, "tokenizer not configured")
        prompt = _messages_to_prompt(req.messages)
        prompt_ids = generator.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = len(prompt_ids)

        sampler = Sampler(temperature=req.temperature, top_p=req.top_p)
        old_sampler = generator.sampler
        generator.sampler = sampler
        try:
            tokens = []
            for tok_id in generator.generate(
                prompt_ids,
                max_new_tokens=req.max_tokens,
            ):
                tokens.append(tok_id)
        finally:
            generator.sampler = old_sampler

        assistant_text = generator.tokenizer.decode(tokens)
        resp = _make_chat_completion_response(
            model_name, req.messages, assistant_text,
            prompt_tokens, len(tokens),
        )
        return JSONResponse(resp)

    @app.get("/")
    def root():
        return {"message": "VerseInference OpenAI-compatible API. See /docs."}

    return app


# ---------------------------------------------------------------------------
# 纯 http.server 版本（fallback）
# ---------------------------------------------------------------------------


def create_http_server(
    generator: StreamingGenerator,
    model_name: str = "verse-model",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """用标准库 ``http.server`` 创建简化版 OpenAI 兼容 server。

    仅支持：
    - ``POST /v1/completions``（非流式）
    - ``GET /v1/models``

    若需要 chat 接口或流式响应，请安装 ``fastapi`` 与 ``uvicorn`` 后使用 ``create_app``。
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, obj: Any):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/v1/models":
                self._send_json(200, {
                    "object": "list",
                    "data": [
                        {
                            "id": model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "verse",
                        }
                    ],
                })
            elif self.path == "/" or self.path == "/health":
                self._send_json(200, {"status": "ok", "model": model_name})
            else:
                self._send_json(404, {"error": {"message": "Not Found"}})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                req = json.loads(body)
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Invalid JSON: {e}"}})
                return

            if self.path == "/v1/completions":
                self._handle_completion(req)
            elif self.path == "/v1/chat/completions":
                self._handle_chat_completion(req)
            else:
                self._send_json(404, {"error": {"message": "Not Found"}})

        def _handle_completion(self, req: dict):
            if generator.tokenizer is None:
                self._send_json(500, {"error": {"message": "tokenizer not configured"}})
                return
            prompt = req.get("prompt", "")
            max_tokens = int(req.get("max_tokens", 100))
            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            prompt_ids = generator.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens = len(prompt_ids)

            sampler = Sampler(temperature=temperature, top_p=top_p)
            old_sampler = generator.sampler
            generator.sampler = sampler
            try:
                tokens = []
                for tok_id in generator.generate(prompt_ids, max_new_tokens=max_tokens):
                    tokens.append(tok_id)
            finally:
                generator.sampler = old_sampler

            completion_text = generator.tokenizer.decode(tokens)
            resp = _make_completion_response(
                model_name, prompt, completion_text,
                prompt_tokens, len(tokens),
            )
            self._send_json(200, resp)

        def _handle_chat_completion(self, req: dict):
            if generator.tokenizer is None:
                self._send_json(500, {"error": {"message": "tokenizer not configured"}})
                return
            messages = req.get("messages", [])
            max_tokens = int(req.get("max_tokens", 100))
            temperature = float(req.get("temperature", 1.0))
            top_p = float(req.get("top_p", 1.0))
            prompt = _messages_to_prompt(messages)
            prompt_ids = generator.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens = len(prompt_ids)

            sampler = Sampler(temperature=temperature, top_p=top_p)
            old_sampler = generator.sampler
            generator.sampler = sampler
            try:
                tokens = []
                for tok_id in generator.generate(prompt_ids, max_new_tokens=max_tokens):
                    tokens.append(tok_id)
            finally:
                generator.sampler = old_sampler

            assistant_text = generator.tokenizer.decode(tokens)
            resp = _make_chat_completion_response(
                model_name, messages, assistant_text,
                prompt_tokens, len(tokens),
            )
            self._send_json(200, resp)

        def log_message(self, fmt, *args):  # noqa: N802
            # 简化日志：仅打印到 stderr
            import sys
            sys.stderr.write(f"[verse-server] {self.address_string()} - {fmt % args}\n")

    server = ThreadingHTTPServer((host, port), _Handler)
    return server


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def run_server(
    generator: StreamingGenerator,
    model_name: str = "verse-model",
    host: str = "0.0.0.0",
    port: int = 8000,
    prefer_fastapi: bool = True,
):
    """启动 HTTP server。

    若 ``prefer_fastapi=True`` 且已安装 ``fastapi`` / ``uvicorn``，用 FastAPI；
    否则用标准库 ``http.server``。
    """
    if prefer_fastapi:
        try:
            import uvicorn  # noqa: F401
            app = create_app(generator, model_name=model_name)
            print(f"[verse-server] FastAPI serving on http://{host}:{port}")
            uvicorn.run(app, host=host, port=port, log_level="info")
            return
        except ImportError:
            print("[verse-server] fastapi/uvicorn not installed, falling back to http.server")

    server = create_http_server(
        generator, model_name=model_name, host=host, port=port,
    )
    print(f"[verse-server] http.server serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    # 当作脚本运行：构建一个 demo generator 并启动 server
    # 需要 PYTHONPATH 包含 verse_torch / verse_nex / verse_compat / verse_tokenizer / verse_inference
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="VerseInference OpenAI-compatible server")
    parser.add_argument("--arch", default="mamba2", help="arch: mamba2 / rwkv7 / hybrid")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="verse-mamba2")
    args = parser.parse_args()

    # 延迟导入，避免在 __init__ 时强依赖；兼容脚本直接执行与 -m 模块执行
    try:
        from verse_infra.verse_inference.model_loader import ModelLoader
    except ImportError:  # pragma: no cover - 仅在脚本直接执行且 PYTHONPATH 含包父目录时触发
        from .model_loader import ModelLoader
    from verse_infra.verse_tokenizer import CharTokenizer

    loader = ModelLoader(
        arch=args.arch,
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
    )
    model = loader.load()
    tokenizer = CharTokenizer()
    gen = StreamingGenerator(model, tokenizer=tokenizer)

    run_server(gen, model_name=args.model_name, host=args.host, port=args.port)


__all__ = ["create_app", "create_http_server", "run_server"]
