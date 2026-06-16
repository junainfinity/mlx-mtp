"""mlx-mtp serve — a pure-stdlib, OpenAI-compatible streaming inference server.

Runs a single pure-MLX Qwen3.5/3.6 VLM checkpoint behind an OpenAI Chat Completions
API so any OpenAI client (incl. agent frameworks) can drive it with native MTP
speculative decoding, vision, tools, and `<think>` reasoning — using ONLY Apple MLX
(mlx.core/mlx.nn) for compute and the Python stdlib for the HTTP/SSE layer. No
FastAPI/uvicorn, no third-party ML-inference framework.

Endpoints (http://host:port):
  GET  /v1/models            — readiness/health probe + model list
  POST /v1/chat/completions  — OpenAI chat completions, streaming (SSE) or blocking

Decoding:
  - "mtp"      native embedded-MTP D1 speculative decode (greedy, lossless vs vanilla)
  - "vanilla"  standard AR with temperature/top_p/top_k sampling
  Selected per-request via the `decoder` field, else the server default (from the
  MLX_MTP_DECODER / MLX_VLM_DRAFT_KIND env var, else "mtp").

CLI (drop-in for an `omlx serve`-style launcher):
  python -m mlx_mtp.serve serve --model-dir DIR --port 8400 --host 127.0.0.1 \
         --max-concurrent-requests 1 [--api-key KEY]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx

from mlx_mtp.loader import load as _load
from mlx_mtp.tokenizer import apply_chat_template_messages, eos_ids as _eos_set

THINK_OPEN = 248068
THINK_CLOSE = 248069
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


# ===========================================================================
# sampling
# ===========================================================================

def sample_token(logits, temperature: float, top_p: float, top_k: int) -> int:
    """Sample one token id from a 1-D logits vector. Greedy if temperature<=0."""
    if not temperature or temperature <= 0:
        return int(mx.argmax(logits, axis=-1).item())
    try:
        lg = logits.astype(mx.float32) * (1.0 / float(temperature))
        if top_k and top_k > 0:
            k = min(int(top_k), lg.shape[-1])
            kth = mx.sort(lg)[-k]
            lg = mx.where(lg < kth, mx.array(-1e9, mx.float32), lg)
        if top_p and 0.0 < top_p < 1.0:
            idx = mx.argsort(-lg)
            sl = lg[idx]
            csum = mx.cumsum(mx.softmax(sl, axis=-1), axis=-1)
            keep = csum < top_p
            keep = mx.concatenate([mx.array([True]), keep[:-1]])
            sl = mx.where(keep, sl, mx.array(-1e9, mx.float32))
            lg = sl[mx.argsort(idx)]
        return int(mx.random.categorical(lg).item())
    except Exception:
        return int(mx.argmax(logits, axis=-1).item())


# ===========================================================================
# message / vision / tool parsing
# ===========================================================================

def _data_uri_to_image(url: str):
    from PIL import Image

    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[7:]).convert("RGB")
    raise ValueError("only data: and file:// image_url are supported (no network fetch)")


def normalize_messages(messages):
    """Split OpenAI messages into (template_messages, images).

    image_url content blocks are pulled out as PIL images and replaced with an
    {"type": "image"} placeholder the model's chat template understands."""
    images = []
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif btype in ("image_url", "image"):
                    url = (block.get("image_url") or {}).get("url") if btype == "image_url" else block.get("image")
                    if url:
                        images.append(_data_uri_to_image(url))
                        parts.append({"type": "image"})
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out, images


def _coerce(v: str):
    """Best-effort type a string parameter value (5 -> int, true -> bool, else str)."""
    try:
        return json.loads(v)
    except Exception:
        return v


def parse_tool_call(inner: str):
    """Parse the inner text of a <tool_call>…</tool_call> block to an OpenAI tool_call.

    Handles both the JSON form {"name":..,"arguments":{..}} and the Qwen XML form
    <function=NAME><parameter=KEY>VAL</parameter>…</function>."""
    name, args = None, None
    s = inner.strip()
    # JSON form
    try:
        obj = json.loads(s)
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
    except Exception:
        pass
    # Qwen XML form
    if name is None:
        m = re.search(r"<function=([^>]+)>(.*)</function>", s, re.DOTALL)
        if m:
            name = m.group(1).strip()
            params = {pm.group(1).strip(): _coerce(pm.group(2).strip())
                      for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", m.group(2), re.DOTALL)}
            if params:
                args = params
            else:
                body = m.group(2).strip()
                try:
                    args = json.loads(body)
                except Exception:
                    args = {"_raw": body} if body else {}
    if name is None:
        return None
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": args}}


# ===========================================================================
# engine — load once, stream tokens (vanilla sampling | native MTP greedy)
# ===========================================================================

def resolve_model_dir(path: str) -> str:
    """Accept either a single model dir (has config.json) or a tree of models; for a
    tree, pick one model preferring mxfp8 > mxfp4 > any quantized > first."""
    p = Path(path).expanduser()
    if (p / "config.json").exists():
        return str(p)
    cands = [d for d in sorted(p.iterdir())
             if d.is_dir() and (d / "config.json").exists() and list(d.glob("*.safetensors"))]
    if not cands:
        raise FileNotFoundError(f"no model (config.json + safetensors) found in {path}")

    def rank(d):
        n = d.name.lower()
        if "mxfp8" in n:
            return 0
        if "mxfp4" in n:
            return 1
        try:
            q = json.loads((d / "config.json").read_text()).get("quantization")
        except Exception:
            q = None
        return 2 if q else 3

    return str(sorted(cands, key=rank)[0])


class Engine:
    def __init__(self, model_dir: str):
        self.model_dir = resolve_model_dir(model_dir)
        self.model, self.processor, self.config = _load(self.model_dir)
        self.lm = self.model.language_model
        self.tok = self.processor.tokenizer
        self.eos = _eos_set(self.processor, self.config)
        self.has_mtp = hasattr(self.lm, "mtp") and self.lm.mtp is not None
        self.lock = threading.Lock()           # one GPU → serialize generation
        self.model_id = os.path.basename(os.path.normpath(self.model_dir))

    # ---- prompt build ----
    def build_prompt(self, messages, tools):
        tmpl_msgs, images = normalize_messages(messages)
        prompt = apply_chat_template_messages(self.processor, tmpl_msgs, tools=tools)
        if images:
            proc = self.processor(text=prompt, images=images, return_tensors="np")
            ids = mx.array(proc["input_ids"])
            pv = mx.array(proc["pixel_values"])
            grid = mx.array(proc["image_grid_thw"]) if "image_grid_thw" in proc else None
            return ids, pv, grid, prompt
        ids = mx.array([self.tok.encode(prompt)])
        return ids, None, None, prompt

    # ---- streaming decode ----
    def stream(self, messages, tools, *, decoder, max_tokens, temperature, top_p, top_k, stops):
        """Yield ('reasoning'|'content'|'tool_call', payload) events, then ('done', meta)."""
        ids, pv, grid, prompt = self.build_prompt(messages, tools)
        prompt_len = ids.shape[1]
        start_think = THINK_OPEN in [int(x) for x in ids[0, -6:].tolist()]
        use_mtp = (decoder == "mtp") and self.has_mtp and (not temperature or temperature <= 0)
        last = len(self.lm.model.layers) - 1
        cache = self.lm.make_cache()

        # prefill (vision goes through the multimodal top-level forward)
        if pv is not None:
            out = self.model(ids, pixel_values=pv, image_grid_thw=grid, cache=cache,
                             capture_layer_ids=[last])
        else:
            out = self.lm(ids, cache=cache, capture_layer_ids=[last])
        hidden = out.hidden_states[0][:, -1:, :]
        cur = sample_token(out.logits[0, -1, :], temperature, top_p, top_k)
        mx.eval(cur)

        mode = "reasoning" if start_think else "content"
        r_toks, c_toks = [], []
        emit_r, c_cursor = 0, 0
        n_tool = 0
        produced = 0
        finish = "stop"

        def feed(tok):
            nonlocal mode, emit_r, c_cursor, n_tool, finish
            ev = []
            if mode == "reasoning":
                if tok == THINK_CLOSE:
                    mode = "content"
                    return ev
                r_toks.append(tok)
                full = self.tok.decode(r_toks)
                if len(full) > emit_r:
                    ev.append(("reasoning", full[emit_r:]))
                    emit_r = len(full)
                return ev
            # content mode
            c_toks.append(tok)
            full = self.tok.decode(c_toks)
            cur_pos = c_cursor
            while True:
                nxt = full.find("<tool_call>", cur_pos)
                if nxt == -1:
                    if len(full) > cur_pos:
                        ev.append(("content", full[cur_pos:]))
                        cur_pos = len(full)
                    break
                if nxt > cur_pos:
                    ev.append(("content", full[cur_pos:nxt]))
                end = full.find("</tool_call>", nxt)
                if end == -1:
                    cur_pos = nxt              # hold back the partial tool_call
                    break
                end += len("</tool_call>")
                tc = parse_tool_call(full[nxt + len("<tool_call>"):end - len("</tool_call>")])
                if tc:
                    ev.append(("tool_call", (n_tool, tc)))
                    n_tool += 1
                    finish = "tool_calls"
                cur_pos = end
            c_cursor = cur_pos
            return ev

        def stop_hit():
            if not stops:
                return False
            txt = self.tok.decode(c_toks)
            return any(s in txt for s in stops)

        done = cur in self.eos                      # never emit an eos token
        with self.lock:
            if not done:
                yield from feed(cur)
                produced = 1
            if use_mtp:
                while not done and produced < max_tokens and not stop_hit():
                    mtp_cache = self.lm.make_mtp_cache()
                    d = int(mx.argmax(self.lm.mtp_forward(hidden, mx.array([[cur]]), mtp_cache)[0, -1, :]).item())
                    vout = self.lm(mx.array([[cur, d]]), cache=cache, capture_layer_ids=[last])
                    vh = vout.hidden_states[0]
                    r = int(mx.argmax(vout.logits[0, 0, :]).item())
                    if d == r:                       # draft accepted -> +d, +s
                        s = int(mx.argmax(vout.logits[0, 1, :]).item())
                        if d in self.eos:
                            done = True
                            break
                        yield from feed(d)
                        produced += 1
                        self.lm.rollback_speculative_cache(cache, vout.gdn_states, 1, 2)
                        cur, hidden = s, vh[:, 1:2, :]
                        if s in self.eos or produced >= max_tokens:
                            done = True
                            break
                        yield from feed(s)
                        produced += 1
                    else:                            # draft rejected -> +r
                        self.lm.rollback_speculative_cache(cache, vout.gdn_states, 0, 2)
                        if r in self.eos:
                            done = True
                            break
                        yield from feed(r)
                        produced += 1
                        cur, hidden = r, vh[:, 0:1, :]
            else:
                while not done and produced < max_tokens and not stop_hit():
                    out = self.lm(mx.array([[cur]]), cache=cache)
                    cur = sample_token(out.logits[0, -1, :], temperature, top_p, top_k)
                    if cur in self.eos:
                        done = True
                        break
                    yield from feed(cur)
                    produced += 1

        if finish == "stop" and produced >= max_tokens and not done:
            finish = "length"
        yield ("done", {"finish": finish, "prompt_tokens": prompt_len,
                        "completion_tokens": produced, "mode": "mtp" if use_mtp else "vanilla"})


# ===========================================================================
# HTTP / SSE layer
# ===========================================================================

ENGINE: Engine = None
API_KEY: str = None


def _now():
    return int(time.time())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _authed(self) -> bool:
        if not API_KEY:
            return True
        hdr = self.headers.get("Authorization", "")
        return hdr.startswith("Bearer ") and hdr[7:].strip() == API_KEY

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            if not self._authed():
                return self._json(401, {"error": {"message": "unauthorized"}})
            return self._json(200, {"object": "list", "data": [
                {"id": ENGINE.model_id, "object": "model", "created": _now(),
                 "owned_by": "mlx-mtp", "mtp": ENGINE.has_mtp}]})
        if self.path.rstrip("/") in ("", "/health", "/healthz"):
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        if not self._authed():
            return self._json(401, {"error": {"message": "unauthorized"}})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad request: {e}"}})

        messages = req.get("messages", [])
        tools = req.get("tools")
        stream = bool(req.get("stream", False))
        samp = req.get("sampling") or {}
        decoder = (req.get("decoder") or req.get("decoding")
                   or os.environ.get("MLX_MTP_DECODER")
                   or _env_decoder()).lower()
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 1024)
        temperature = float(req.get("temperature", samp.get("temperature", 0.0)) or 0.0)
        top_p = float(req.get("top_p", samp.get("top_p", 1.0)) or 1.0)
        top_k = int(req.get("top_k", samp.get("top_k", 0)) or 0)
        stops = req.get("stop")
        if isinstance(stops, str):
            stops = [stops]
        cid = "chatcmpl-" + uuid.uuid4().hex
        gen = ENGINE.stream(messages, tools, decoder=decoder, max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p, top_k=top_k, stops=stops)
        try:
            if stream:
                self._stream(cid, gen, req)
            else:
                self._blocking(cid, gen)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            gen.close()      # release the generation lock even on client disconnect

    # ---- streaming SSE ----
    def _stream(self, cid, gen, req):
        self.close_connection = True               # SSE: read-until-close framing
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        want_usage = bool((req.get("stream_options") or {}).get("include_usage"))

        def send(delta, finish=None, usage=None):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": _now(),
                     "model": ENGINE.model_id,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                chunk["usage"] = usage
            self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()

        send({"role": "assistant"})
        meta = None
        for kind, payload in gen:
            if kind == "reasoning":
                send({"reasoning_content": payload})
            elif kind == "content":
                send({"content": payload})
            elif kind == "tool_call":
                idx, tc = payload
                send({"tool_calls": [{"index": idx, "id": tc["id"], "type": "function",
                                      "function": tc["function"]}]})
            elif kind == "done":
                meta = payload
        usage = {"prompt_tokens": meta["prompt_tokens"], "completion_tokens": meta["completion_tokens"],
                 "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"]}
        send({}, finish=meta["finish"], usage=usage if want_usage else None)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- blocking ----
    def _blocking(self, cid, gen):
        reasoning, content, tool_calls, meta = [], [], [], None
        for kind, payload in gen:
            if kind == "reasoning":
                reasoning.append(payload)
            elif kind == "content":
                content.append(payload)
            elif kind == "tool_call":
                tool_calls.append(payload[1])
            elif kind == "done":
                meta = payload
        msg = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": _now(), "model": ENGINE.model_id,
            "choices": [{"index": 0, "message": msg, "finish_reason": meta["finish"]}],
            "usage": {"prompt_tokens": meta["prompt_tokens"],
                      "completion_tokens": meta["completion_tokens"],
                      "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"]},
        })


def _env_decoder() -> str:
    kind = (os.environ.get("MLX_VLM_DRAFT_KIND") or "").lower()
    if kind in ("mtp", "dflash", "draft", "speculative"):
        return "mtp"          # native speculative is the pure-mlx default
    if kind in ("standard", "vanilla", "none", ""):
        return "mtp" if not kind else "vanilla"
    return "mtp"


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog="mlx-mtp-serve", description="OpenAI-compatible mlx-mtp inference server")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-concurrent-requests", type=int, default=1)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args(argv)

    global ENGINE, API_KEY
    API_KEY = a.api_key
    print(f">> mlx-mtp serve: loading {a.model_dir} ...", flush=True)
    t0 = time.perf_counter()
    ENGINE = Engine(a.model_dir)
    print(f">> loaded {ENGINE.model_id} in {time.perf_counter()-t0:.1f}s | "
          f"MTP={ENGINE.has_mtp} | default decoder={_env_decoder()}", flush=True)

    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f">> listening on http://{a.host}:{a.port}/v1  (GET /v1/models, POST /v1/chat/completions)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
