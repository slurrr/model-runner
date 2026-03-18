from __future__ import annotations

import ast
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo


def _resolve_path(path: str, config_path: str | None = None) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if config_path:
        cfg_dir = os.path.dirname(config_path)
        candidate = os.path.abspath(os.path.join(cfg_dir, expanded))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(expanded)


def _truncate_text(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    marker = "...(truncated)"
    keep = max(1, limit - len(marker))
    return text[:keep] + marker


class _SafeCalcEvaluator:
    _MAX_EXPR_LEN = 200
    _MAX_DEPTH = 32
    _MAX_ABS_EXPONENT = 12
    _MAX_ABS_NUMBER = 10**12

    def eval(self, expression: str) -> str:
        expr = (expression or "").strip()
        if not expr:
            raise ValueError("expression is required")
        if len(expr) > self._MAX_EXPR_LEN:
            raise ValueError(f"expression too long (max {self._MAX_EXPR_LEN} chars)")
        tree = ast.parse(expr, mode="eval")
        value = self._visit(tree.body, depth=0)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _visit(self, node: ast.AST, *, depth: int) -> float | int:
        if depth > self._MAX_DEPTH:
            raise ValueError("expression too deep")

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError("only numeric literals are allowed")
            return self._validate_number(node.value)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._visit(node.operand, depth=depth + 1)
            out = +value if isinstance(node.op, ast.UAdd) else -value
            return self._validate_number(out)

        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left = self._visit(node.left, depth=depth + 1)
            right = self._visit(node.right, depth=depth + 1)
            if isinstance(node.op, ast.Add):
                return self._validate_number(left + right)
            if isinstance(node.op, ast.Sub):
                return self._validate_number(left - right)
            if isinstance(node.op, ast.Mult):
                return self._validate_number(left * right)
            if isinstance(node.op, ast.Div):
                return self._validate_number(left / right)
            if isinstance(node.op, ast.FloorDiv):
                return self._validate_number(left // right)
            if isinstance(node.op, ast.Mod):
                return self._validate_number(left % right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > self._MAX_ABS_EXPONENT:
                    raise ValueError("exponent too large")
                return self._validate_number(left**right)

        raise ValueError("unsupported expression")

    def _validate_number(self, value: float | int) -> float | int:
        if isinstance(value, bool):
            raise ValueError("unsupported value")
        if not math.isfinite(float(value)):
            raise ValueError("non-finite result")
        if abs(float(value)) > self._MAX_ABS_NUMBER:
            raise ValueError("result too large")
        return value


_SAFE_CALC = _SafeCalcEvaluator()


@dataclass
class ToolDefinition:
    name: str
    schema: dict[str, object]
    handler: Callable[[dict[str, object]], str] | None
    source: str


@dataclass
class ToolRuntime:
    backend_name: str
    enabled: bool
    mode: str
    schema_file: str
    tool_choice: str
    allow: list[str]
    deny: list[str]
    max_calls_per_turn: int
    timeout_s: float
    max_result_chars: int
    supported_backend: bool
    available_tools: dict[str, ToolDefinition]
    exposed_tools: dict[str, ToolDefinition]

    def exposed_schema(self) -> list[dict[str, object]]:
        return [tool.schema for tool in self.exposed_tools.values()]

    def tool_names(self) -> list[str]:
        return list(self.exposed_tools.keys())

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self.exposed_tools.get((name or "").strip())

    def describe(self, *, verbose: bool = False) -> dict[str, object]:
        data: dict[str, object] = {
            "backend": self.backend_name,
            "supported_backend": self.supported_backend,
            "enabled": self.enabled,
            "mode": self.mode,
            "schema_file": self.schema_file or "",
            "tool_choice": self.tool_choice or "",
            "allow": list(self.allow),
            "deny": list(self.deny),
            "max_calls_per_turn": self.max_calls_per_turn,
            "timeout_s": self.timeout_s,
            "max_result_chars": self.max_result_chars,
            "tool_names": self.tool_names(),
        }
        if verbose:
            data["available_tools"] = {
                name: {
                    "source": tool.source,
                    "executable": bool(tool.handler),
                }
                for name, tool in self.available_tools.items()
            }
        return data

    def execute(
        self,
        *,
        name: str,
        arguments_raw: str,
        denied_by_policy: bool = False,
    ) -> dict[str, object]:
        item: dict[str, object] = {
            "name": name,
            "arguments_raw": arguments_raw,
            "arguments_json": None,
            "status": "received",
            "result": None,
            "error": None,
        }
        if denied_by_policy:
            item["status"] = "denied_by_policy"
            item["error"] = "Tool call denied by policy."
            item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
            return item

        tool = self.get_tool(name)
        if tool is None:
            item["status"] = "tool_unavailable"
            item["error"] = f"Tool '{name}' is unavailable."
            item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
            return item

        try:
            parsed = json.loads(arguments_raw or "{}")
        except Exception as exc:
            item["status"] = "invalid_arguments"
            item["error"] = f"Invalid tool arguments JSON: {exc}"
            item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
            return item

        if not isinstance(parsed, dict):
            item["status"] = "invalid_arguments"
            item["error"] = "Tool arguments must decode to a JSON object."
            item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
            return item

        item["arguments_json"] = parsed
        if self.mode != "execute":
            item["status"] = "dry_run"
            return item

        if tool.handler is None:
            item["status"] = "tool_unavailable"
            item["error"] = f"Tool '{name}' is not executable in this repo."
            item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
            return item

        def _call() -> str:
            return str(tool.handler(parsed))

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_call)
            try:
                raw_result = future.result(timeout=max(0.1, self.timeout_s))
            except FutureTimeoutError:
                future.cancel()
                item["status"] = "timeout"
                item["error"] = f"Tool '{name}' timed out after {self.timeout_s:.1f}s."
                item["result"] = _truncate_text(str(item["error"]), self.max_result_chars)
                return item
            except Exception as exc:
                item["status"] = "execute_error"
                item["error"] = str(exc)
                item["result"] = _truncate_text(str(exc), self.max_result_chars)
                return item

        item["status"] = "executed"
        item["result"] = _truncate_text(raw_result, self.max_result_chars)
        return item


def _builtin_tool_schemas() -> list[ToolDefinition]:
    def calc_tool(arguments: dict[str, object]) -> str:
        return _SAFE_CALC.eval(str(arguments.get("expression", "") or ""))

    def get_time_tool(arguments: dict[str, object]) -> str:
        tz = str(arguments.get("tz", "America/Denver") or "America/Denver")
        now = datetime.now(ZoneInfo(tz))
        return now.isoformat()

    def echo_tool(arguments: dict[str, object]) -> str:
        return str(arguments.get("text", "") or "")

    return [
        ToolDefinition(
            name="calc",
            source="builtin",
            handler=calc_tool,
            schema={
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Evaluate a simple arithmetic expression safely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Arithmetic expression to evaluate."}
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
        ToolDefinition(
            name="get_time",
            source="builtin",
            handler=get_time_tool,
            schema={
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Return the current time for an IANA timezone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tz": {
                                "type": "string",
                                "description": "IANA timezone name like America/Denver.",
                                "default": "America/Denver",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
        ),
        ToolDefinition(
            name="echo",
            source="builtin",
            handler=echo_tool,
            schema={
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Return the provided text unchanged.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
        ),
    ]


def _load_schema_file(schema_file: str, *, config_path: str | None) -> list[ToolDefinition]:
    if not schema_file:
        return []
    resolved = _resolve_path(schema_file, config_path=config_path)
    with open(resolved, "rb") as fh:
        data = json.loads(fh.read().decode("utf-8"))
    tools_obj = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(tools_obj, list):
        raise RuntimeError(f"Tool schema file must contain a tools list: {resolved}")
    out: list[ToolDefinition] = []
    for item in tools_obj:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid tool schema entry in {resolved}: expected object")
        if item.get("type") != "function":
            raise RuntimeError(f"Unsupported tool schema type in {resolved}: {item.get('type')!r}")
        function = item.get("function")
        if not isinstance(function, dict):
            raise RuntimeError(f"Invalid function tool schema in {resolved}")
        name = str(function.get("name", "") or "").strip()
        if not name:
            raise RuntimeError(f"Function tool missing name in {resolved}")
        out.append(ToolDefinition(name=name, schema=item, handler=None, source=resolved))
    return out


def build_tool_runtime(args, backend_name: str) -> ToolRuntime:
    mode = str(getattr(args, "tools_mode", "dry_run") or "dry_run").strip().lower()
    if mode not in {"off", "dry_run", "execute"}:
        raise RuntimeError(f"Invalid tools.mode: {mode!r}")
    enabled = bool(getattr(args, "tools_enabled", False)) and mode != "off"
    supported_backend = backend_name in {"openai", "vllm"}
    schema_file = str(getattr(args, "tools_schema_file", "") or "").strip()
    allow = [str(x).strip() for x in (getattr(args, "tools_allow", []) or []) if str(x).strip()]
    deny = [str(x).strip() for x in (getattr(args, "tools_deny", []) or []) if str(x).strip()]

    tool_defs = _builtin_tool_schemas() + _load_schema_file(
        schema_file,
        config_path=getattr(args, "_config_path", None),
    )

    collisions: dict[str, list[str]] = {}
    for tool in tool_defs:
        collisions.setdefault(tool.name, []).append(tool.source)
    dupes = sorted(name for name, sources in collisions.items() if len(sources) > 1)
    if dupes:
        parts = [f"{name} ({', '.join(collisions[name])})" for name in dupes]
        raise RuntimeError("Tool schema name collision(s): " + "; ".join(parts))

    available = {tool.name: tool for tool in sorted(tool_defs, key=lambda item: item.name)}
    allow_lower = {name.lower() for name in allow}
    deny_lower = {name.lower() for name in deny}
    exposed: dict[str, ToolDefinition] = {}
    for name, tool in available.items():
        lowered = name.lower()
        if allow_lower and lowered not in allow_lower:
            continue
        if lowered in deny_lower:
            continue
        exposed[name] = tool

    return ToolRuntime(
        backend_name=backend_name,
        enabled=enabled,
        mode=mode,
        schema_file=_resolve_path(schema_file, config_path=getattr(args, "_config_path", None)) if schema_file else "",
        tool_choice=str(getattr(args, "tools_tool_choice", "") or "").strip(),
        allow=allow,
        deny=deny,
        max_calls_per_turn=max(1, int(getattr(args, "tools_max_calls_per_turn", 3) or 3)),
        timeout_s=float(getattr(args, "tools_timeout_s", 10) or 10),
        max_result_chars=max(1, int(getattr(args, "tools_max_result_chars", 8000) or 8000)),
        supported_backend=supported_backend,
        available_tools=available,
        exposed_tools=exposed,
    )
