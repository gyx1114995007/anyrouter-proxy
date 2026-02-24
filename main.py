import os
import json
import logging
import itertools
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

# ========== 加载配置 ===========
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()

DEBUG = config.get("debug", False)
ANYROUTER_URL = config.get("anyrouter_url", "")
FIXED_UA = config.get("user_agent", "")
API_KEYS = config.get("api_keys", [])

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("proxy")

# 轮询代代器
key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

PORT = int(os.getenv("PORT", "8080"))

app = FastAPI()

# AnyRouter 固定 system
FIXED_SYSTEM = [
    {
        "type": "text",
        "text": "You are Claude Code, Anthropic's official CLI for Claude.",
        "cache_control": {"type": "ephemeral"}
    },
    {
        "type": "text",
        "text": ".",
        "cache_control": {"type": "ephemeral"}
    }
]


def get_next_key():
    """轮询获取下一个 API key"""
    if key_cycle:
        return next(key_cycle)
    return ""


def get_user_system_text(original_system):
    """提取用户原始 system 提示词文本"""
    if isinstance(original_system, str) and original_system.strip():
        return original_system.strip()
    elif isinstance(original_system, list):
        parts = []
        for item in original_system:
            if isinstance(item, dict) and item.get("text", "").strip():
                parts.append(item["text"].strip())
        return "\n\n".join(parts) if parts else ""
    return ""


def transform_body(body: dict) -> dict:
    """转换请求体为 AnyRouter 兼容格式"""
    user_system = get_user_system_text(body.get("system"))
    body["system"] = FIXED_SYSTEM

    # 用户提示词注入第一条 user message
    if user_system:
        for msg in body.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                prefix = f"[System Instructions]\n{user_system}\n[/System Instructions]\n\n"
                if isinstance(content, str):
                    msg["content"] = prefix + content
                elif isinstance(content, list) and content:
                    content[0]["text"] = prefix + content[0].get("text", "")
                break

    # 清理空 assistant + 合并连续同角色
    cleaned = []
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant":
            c = msg.get("content", "")
            if isinstance(c, str) and not c.strip():
                continue
            if isinstance(c, list) and not any(
                isinstance(b, dict) and b.get("text", "").strip() for b in c
            ):
                continue
        if cleaned and cleaned[-1]["role"] == msg["role"]:
            prev = cleaned[-1]
            pt = prev["content"] if isinstance(prev["content"], str) else prev["content"][0].get("text", "")
            ct = msg["content"] if isinstance(msg["content"], str) else msg["content"][0].get("text", "")
            prev["content"] = [{"type": "text", "text": f"{pt}\n{ct}", "cache_control": {"type": "ephemeral"}}]
        else:
            cleaned.append(msg)
    body["messages"] = cleaned

    # 给所有 content 块加 cache_control
    for msg in body["messages"]:
        c = msg.get("content", "")
        if isinstance(c, str):
            msg["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
        elif isinstance(c, list):
            for block in c:
                if isinstance(block, dict) and "cache_control" not in block:
                    block["cache_control"] = {"type": "ephemeral"}

    body.setdefault("max_tokens", 32000)
    body.setdefault("temperature", 1)
    body.setdefault("stream", True)
    return body


# ========== 热重载配置接口 ==========
@app.post("/reload")
async def reload_config():
    """热重载 config.json，无需重启服务"""
    global config, DEBUG, ANYROUTER_URL, FIXED_UA, API_KEYS, key_cycle
    try:
        config = load_config()
        DEBUG = config.get("debug", False)
        ANYROUTER_URL = config.get("anyrouter_url", "")
        FIXED_UA = config.get("user_agent", "")
        API_KEYS = config.get("api_keys", [])
        key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        logger.info(f"配置已重载，{len(API_KEYS)} 个 key")
        return JSONResponse({"status": "ok", "keys_count": len(API_KEYS)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ========== 代理主路由 ==========
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    body_bytes = await request.body()

    # 客户端传了 key 就用客户积的，否则轮询
    auth = request.headers.get("Authorization", "").strip()
    if not auth or auth == "Bearer":
        key = get_next_key()
        auth = f"Bearer {key}" if key else ""

    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
        "User-Agent": FIXED_UA
    }

    target_url = f"{ANYROUTER_URL.rstrip('/')}/{path}"

    if request.method == "POST" and body_bytes:
        try:
            body = json.loads(body_bytes)
            if DEBUG:
                logger.debug(f"原始请求: {json.dumps(body, ensure_ascii=False)[:300]}")
            body = transform_body(body)
            body_bytes = json.dumps(body).encode()
            if DEBUG:
                logger.debug(f"转发到: {target_url}")
        except json.JSONDecodeError:
            logger.error("JSON 解析失败，原样透传")

    is_stream = b'"stream": true' in body_bytes or b'"stream":true' in body_bytes

    if is_stream:
        async def stream():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", target_url, content=body_bytes, headers=headers) as resp:
                    if resp.status_code != 200:
                        err = b""
                        async for chunk in resp.aiter_bytes():
                            err += chunk
                        logger.error(f"AnyRouter {resp.status_code}: {err.decode('utf-8', errors='replace')}")
                        yield err
                    else:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(target_url, content=body_bytes, headers=headers)
            return StreamingResponse(content=iter([resp.content]), status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务 | keys: {len(API_KEYS)} | target: {ANYROUTER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
