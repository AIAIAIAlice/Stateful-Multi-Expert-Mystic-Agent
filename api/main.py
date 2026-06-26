from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = PROJECT_ROOT / "frontend"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yhj_agent.orchestrator import LangGraphOrchestrator


HOST = "127.0.0.1"
PORT = 8001

ORCHESTRATOR = LangGraphOrchestrator()
TASK_STORE: dict[str, dict[str, Any]] = {}
TASK_STORE_LOCK = threading.Lock()


def _run_task_background(
    task_id: str,
    fn: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    print(f"[TASK {task_id}] started {fn.__name__}", flush=True)
    try:
        result = fn(*args, **kwargs)
        with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                **TASK_STORE.get(task_id, {}),
                "status": "completed",
                "result": result,
                "completed_at": time.time(),
            }
        print(f"[TASK {task_id}] completed", flush=True)
    except Exception as exc:
        print(f"[TASK {task_id}] failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        with TASK_STORE_LOCK:
            TASK_STORE[task_id] = {
                **TASK_STORE.get(task_id, {}),
                "status": "failed",
                "result": {"error": f"{type(exc).__name__}: {exc}"},
                "completed_at": time.time(),
            }


def _cleanup_old_tasks() -> None:
    cutoff = time.time() - 1800
    with TASK_STORE_LOCK:
        expired_task_ids = [
            task_id
            for task_id, task in TASK_STORE.items()
            if task.get("created_at", cutoff) < cutoff
        ]
        for task_id in expired_task_ids:
            del TASK_STORE[task_id]


class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/app.js":
            self._send_file(STATIC_DIR / "app.js", "text/javascript; charset=utf-8")
        elif self.path == "/styles.css":
            self._send_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
        elif self.path == "/api/health":
            self._send_json({"status": "ok"})
        elif self.path.startswith("/api/sessions/"):
            session_id = unquote(self.path.removeprefix("/api/sessions/")) or "demo-session"
            self._send_json(ORCHESTRATOR.get_session(session_id))
        elif self.path.startswith("/api/tasks/"):
            self._handle_get_task()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/tasks":
            self._handle_create_task()
        elif self.path == "/api/turns":
            self._handle_turns()
        elif self.path == "/api/resume":
            self._handle_resume()
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} {format % args}", flush=True)

    def _handle_create_task(self) -> None:
        try:
            payload = self._read_json_body()
            task_type = str(payload.get("type", "turn"))
            task_id = uuid.uuid4().hex[:12]

            with TASK_STORE_LOCK:
                TASK_STORE[task_id] = {
                    "status": "running",
                    "result": None,
                    "created_at": time.time(),
                }

            if task_type == "turn":
                message = str(payload.get("message", "")).strip()
                if not message:
                    self._send_json({"error": "message 不能为空"}, status_code=400)
                    return
                fn = ORCHESTRATOR.run_turn
                kwargs = {
                    "session_id": str(payload.get("session_id", "demo-session")),
                    "user_input": message,
                    "user_id": str(payload.get("user_id", "")),
                }
            elif task_type == "resume":
                answer = str(payload.get("answer", "")).strip()
                if not answer:
                    self._send_json({"error": "answer 不能为空"}, status_code=400)
                    return
                fn = ORCHESTRATOR.resume_turn
                kwargs = {
                    "session_id": str(payload.get("session_id", "demo-session")),
                    "answer": answer,
                }
            else:
                self._send_json({"error": f"unknown task type: {task_type}"}, status_code=400)
                return

            thread = threading.Thread(
                target=_run_task_background,
                args=(task_id, fn, (), kwargs),
                daemon=True,
            )
            thread.start()
            _cleanup_old_tasks()
            self._send_json({"task_id": task_id})
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是合法 JSON"}, status_code=400)
        except Exception as exc:
            self._send_json({"error": f"服务端执行失败：{type(exc).__name__}: {exc}"}, status_code=500)

    def _handle_get_task(self) -> None:
        task_id = unquote(self.path.removeprefix("/api/tasks/"))
        with TASK_STORE_LOCK:
            task = TASK_STORE.get(task_id)
        if task is None:
            self._send_json({"error": "task not found"}, status_code=404)
            return
        self._send_json(task)

    def _handle_turns(self) -> None:
        try:
            payload = self._read_json_body()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"error": "message 不能为空"}, status_code=400)
                return
            result = ORCHESTRATOR.run_turn(
                session_id=str(payload.get("session_id", "demo-session")),
                user_input=message,
                user_id=str(payload.get("user_id", "")),
            )
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是合法 JSON"}, status_code=400)
        except Exception as exc:
            self._send_json({"error": f"服务端执行失败：{type(exc).__name__}: {exc}"}, status_code=500)

    def _handle_resume(self) -> None:
        try:
            payload = self._read_json_body()
            session_id = str(payload.get("session_id", "demo-session"))
            answer = str(payload.get("answer", "")).strip()
            if not answer:
                self._send_json({"error": "answer 不能为空"}, status_code=400)
                return
            result = ORCHESTRATOR.resume_turn(session_id=session_id, answer=answer)
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是合法 JSON"}, status_code=400)
        except Exception as exc:
            self._send_json({"error": f"服务端执行失败：{type(exc).__name__}: {exc}"}, status_code=500)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(raw_body or "{}")
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _check_environment() -> None:
    in_venv = sys.prefix != sys.base_prefix or "venv" in str(Path(sys.executable)).lower()
    if not in_venv:
        print("=" * 60, flush=True)
        print("  [WARNING] 未检测到虚拟环境", flush=True)
        print(f"  当前 Python: {sys.executable}", flush=True)
        print("  建议使用: .venv-win\\Scripts\\python.exe api\\main.py", flush=True)
        print("  或使用: start.bat 一键启动", flush=True)
        print("=" * 60, flush=True)

    missing_modules = []
    for module_name in ("pydantic", "chromadb", "langgraph"):
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(module_name)
    if missing_modules:
        print(f"[ERROR] 缺少依赖: {', '.join(missing_modules)}", flush=True)
        print("请运行: uv sync", flush=True)
        sys.exit(1)


def main() -> None:
    _check_environment()
    server = ThreadingHTTPServer((HOST, PORT), DemoRequestHandler)
    print("=" * 60, flush=True)
    print("  API 服务已启动", flush=True)
    print(f"  地址: http://{HOST}:{PORT}", flush=True)
    print(f"  健康检查: http://{HOST}:{PORT}/api/health", flush=True)
    print(f"  Python: {sys.executable}", flush=True)
    print("=" * 60, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
