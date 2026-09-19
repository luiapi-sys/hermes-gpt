#!/usr/bin/env python3
"""Agent Parley -> Hermes GPT Mission Control adapter.

This adapter intentionally stops at the Mission Control decision boundary.
It may create/update mission planning state and request placement/budget/recovery
proposals, but it never dispatches a worker and never approves a mission.

Input is a JSON document with optional keys:
  mission_create: {...tool arguments...}
  mission_update: {...tool arguments...}
  plan_create: {...tool arguments...}
  plan_decompose: {...tool arguments...}
  plan_set_status: {...tool arguments...}
  budget_set: {...tool arguments...}
  budget_check: {...tool arguments...}
  placement_scores: [{...tool arguments...}, ...]
  placement_candidates: [{...tool arguments...}, ...]
  failure_classify: [{...tool arguments...}, ...]
  controller_reconcile: [{...tool arguments...}, ...]

By default every tool exposing a ``dry_run`` argument is forced to dry-run.
Pass --apply to allow non-dry-run Mission Control writes. This still does not
add any dispatch/approval tool to the allowlist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOL_MAP: dict[str, tuple[str, bool]] = {
    "mission_create": ("hermes_mission_create", False),
    "mission_update": ("hermes_mission_update", False),
    "plan_create": ("hermes_plan_create", False),
    "plan_decompose": ("hermes_plan_decompose", False),
    "plan_set_status": ("hermes_plan_set_status", False),
    "budget_set": ("hermes_budget_set", False),
    "budget_check": ("hermes_budget_check", False),
    "placement_scores": ("hermes_placement_score", True),
    "placement_candidates": ("hermes_placement_candidates", True),
    "failure_classify": ("hermes_failure_classify", True),
    "controller_reconcile": ("hermes_controller_reconcile", True),
}

# Safety property: no A2A/fleet dispatch, runner start, approval, merge, or shell
# tool is reachable through this adapter.
ALLOWED_TOOLS = frozenset(tool for tool, _ in TOOL_MAP.values())


@dataclass(frozen=True)
class Step:
    key: str
    tool: str
    arguments: dict[str, Any]


def load_request(path: str) -> dict[str, Any]:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def build_steps(request: dict[str, Any]) -> list[Step]:
    unknown = sorted(set(request) - set(TOOL_MAP))
    if unknown:
        raise ValueError(f"unknown request keys: {', '.join(unknown)}")

    steps: list[Step] = []
    for key, (tool, repeated) in TOOL_MAP.items():
        if key not in request:
            continue
        payload = request[key]
        items = payload if repeated else [payload]
        if repeated and not isinstance(payload, list):
            raise ValueError(f"{key} must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"{key} entries must be objects")
            if tool not in ALLOWED_TOOLS:
                raise ValueError(f"tool {tool!r} is outside the decision boundary")
            steps.append(Step(key=key, tool=tool, arguments=dict(item)))
    if not steps:
        raise ValueError("request contains no Mission Control steps")
    return steps


def _tool_schemas(tools_result: Any) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in getattr(tools_result, "tools", []):
        schemas[tool.name] = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {}) or {}
    return schemas


def prepare_arguments(step: Step, schema: dict[str, Any], *, apply: bool) -> dict[str, Any]:
    args = dict(step.arguments)
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if "dry_run" in props:
        args["dry_run"] = not apply
    # Never manufacture an approval/confirmation gate. The caller must provide
    # an explicitly documented confirmation field if a selected Mission tool
    # requires one.
    return args


def _result_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None)
    if content is None:
        return str(result)
    out: list[Any] = []
    for item in content:
        text = getattr(item, "text", None)
        out.append(text if text is not None else str(item))
    return out


async def run(request: dict[str, Any], *, url: str, token: str | None, apply: bool) -> dict[str, Any]:
    # Lazy import keeps build_steps/prepare_arguments unit-testable without an
    # active MCP runtime.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"Authorization": f"Bearer {token}"} if token else None
    steps = build_steps(request)
    records: list[dict[str, Any]] = []

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            schemas = _tool_schemas(await session.list_tools())
            missing = sorted({step.tool for step in steps} - set(schemas))
            if missing:
                raise RuntimeError("Mission Control tools unavailable: " + ", ".join(missing))

            for index, step in enumerate(steps, start=1):
                args = prepare_arguments(step, schemas[step.tool], apply=apply)
                result = await session.call_tool(step.tool, arguments=args)
                records.append(
                    {
                        "index": index,
                        "key": step.key,
                        "tool": step.tool,
                        "arguments": args,
                        "is_error": bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
                        "result": _result_payload(result),
                    }
                )
                if records[-1]["is_error"]:
                    break

    return {
        "engine": "agent-parley/hermes-gpt-mission-control",
        "mode": "apply" if apply else "dry-run",
        "dispatch_performed": False,
        "approval_performed": False,
        "steps": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="JSON request file, or '-' for stdin")
    parser.add_argument(
        "--url",
        default=os.environ.get("HERMES_GPT_MCP_URL", "http://127.0.0.1:7677/mcp"),
        help="Hermes GPT streamable HTTP MCP endpoint",
    )
    parser.add_argument(
        "--token-env",
        default="HERMES_GPT_TOKEN",
        help="environment variable containing the bearer token (never printed)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="allow Mission Control writes by setting supported dry_run arguments false",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        request = load_request(args.request)
        token = os.environ.get(args.token_env) or None
        payload = asyncio.run(run(request, url=args.url, token=token, apply=args.apply))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
