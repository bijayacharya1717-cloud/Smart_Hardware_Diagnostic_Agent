"""Run a single Gemini diagnostic pass over the three project MCP servers.

Usage: python diagnostic_agent.py "Temperature rises abnormally" [--failure-time "2026-10-08 13:15:00"]
Set GEMINI_API_KEY, QDRANT_URL, and QDRANT_API_KEY in the project .env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from google.genai import types
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


async def connect(stack: AsyncExitStack, name: str, script: Path, extra_env: dict | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    params = StdioServerParameters(command=sys.executable, args=[str(script)], env=env)
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def call(session, tool: str, **arguments):
    result = await session.call_tool(tool, arguments)
    if result.is_error:
        return {"error": f"{tool} MCP call failed", "details": [getattr(c, "text", "") for c in result.content]}
    blocks = [getattr(c, "text", "") for c in result.content]
    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except (json.JSONDecodeError, TypeError):
            parsed.append(block)
    return parsed[0] if len(parsed) == 1 else parsed


async def gather_evidence(description: str, failure_time: str | None):
    async with AsyncExitStack() as stack:
        datasheet = await connect(stack, "datasheet_rag", ROOT / "diagnostic_rag" / "search_server.py", {"RAG_SERVER": "datasheet"})
        schematic = await connect(stack, "schematic_rag", ROOT / "diagnostic_rag" / "search_server.py", {"RAG_SERVER": "schematic"})
        sensors = await connect(stack, "sensor_tools", ROOT / "sensor_tool" / "sensor_tool.py")

        # Sensor schema is discovered from the data through tool errors rather than assumed.
        window_start = window_end = None
        if failure_time:
            from datetime import datetime, timedelta
            when = datetime.fromisoformat(failure_time)
            window_start = (when - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            window_end = (when + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")

        anomaly_results = {}
        for metric in ("temperature", "pressure", "humidity", "current", "voltage"):
            anomaly_results[metric] = await call(
                sensors,
                "detect_anomaly",
                metric=metric,
                start_time=window_start,
                end_time=window_end,
            )

        sensor_details = {"anomalies": anomaly_results}
        if failure_time:
            # Pull summaries over the same hour used for anomaly analysis.
            sensor_details["window"] = {
                metric: await call(sensors, "query_sensor_log", start_time=window_start, end_time=window_end, metric=metric)
                for metric in ("temperature", "pressure", "humidity", "current", "voltage")
            }

        ds = await call(datasheet, "search_datasheets", query=f"{description}; operating limits, electrical specifications, fault causes", limit=5)
        sch = await call(schematic, "search_schematics", query=f"{description}; relevant pins, motor driver, supply wiring, signal connections", limit=5)
        return {"sensor_tools": sensor_details, "datasheet_rag": ds, "schematic_rag": sch}


def _evidence_catalog(evidence: dict) -> tuple[dict, list[str]]:
    catalog, retrieval_context = {}, []
    counter = 1
    for group in ("sensor_tools", "datasheet_rag", "schematic_rag"):
        items = evidence[group]
        if group == "sensor_tools":
            for kind, data in items.items():
                text = f"{kind}: {json.dumps(data, ensure_ascii=False)}"
                catalog[f"E{counter}"] = {"source": "sensor log / sensor tool", "evidence_type": kind, "content": text}
                retrieval_context.append(text)
                counter += 1
        else:
            for item in items.get("evidence", []) if isinstance(items, dict) else []:
                ref = f"E{counter}"
                catalog[ref] = {"source": item.get("source"), "page": item.get("page"), "type": item.get("type"), "content": item.get("content"), "image_path": item.get("image_path")}
                retrieval_context.append(f"{ref}: {item.get('content')} (source={item.get('source')}, page={item.get('page')}, type={item.get('type')})")
                counter += 1
    return catalog, retrieval_context


def _local_evidence_report(description: str, failure_time: str | None, evidence: dict, catalog: dict, errors: list[str]) -> str:
    """Produce a conservative cited report when every Gemini endpoint is unavailable."""
    sensor = evidence.get("sensor_tools", {})
    anomaly_results = sensor.get("anomalies", {})
    window_results = sensor.get("window", {})
    refs = {entry.get("evidence_type"): ref for ref, entry in catalog.items() if entry.get("evidence_type")}

    lines = [
        "LOCAL EVIDENCE REPORT — not AI-generated",
        (
            "Gemini could not complete a response on any configured model. This is a deterministic summary of retrieved evidence, not an AI-generated root-cause diagnosis."
            if errors else
            "Gemini was not called. This is a deterministic summary of retrieved evidence, not an AI-generated root-cause diagnosis."
        ),
        f"Reported failure: {description}",
        f"Failure time: {failure_time or 'not provided'}",
        "",
        "Sensor findings:",
    ]

    anomaly_ref = refs.get("anomalies")
    for metric, result in anomaly_results.items():
        citation = f" [{anomaly_ref}]" if anomaly_ref else ""
        if not isinstance(result, dict):
            lines.append(f"- {metric}: tool returned an unreadable result.{citation}")
        elif result.get("error"):
            lines.append(f"- {metric}: unavailable — {result['error']}.{citation}")
        elif result.get("anomaly_count", 0):
            lines.append(
                f"- {metric}: {result['anomaly_count']} statistical anomaly reading(s); peak {result.get('peak_value')} "
                f"at {result.get('peak_at')} (analysis window {result.get('window_start')} to {result.get('window_end')}).{citation}"
            )
        else:
            lines.append(
                f"- {metric}: no readings exceeded the configured statistical anomaly threshold "
                f"in {result.get('readings_checked', 0)} checked reading(s). This does not establish that values were within hardware limits.{citation}"
            )

    window_ref = refs.get("window")
    if window_results:
        lines.extend(["", "Sensor summaries:"])
        for metric, result in window_results.items():
            citation = f" [{window_ref}]" if window_ref else ""
            if not isinstance(result, dict):
                lines.append(f"- {metric}: no readable summary.{citation}")
            elif result.get("error"):
                lines.append(f"- {metric}: unavailable — {result['error']}.{citation}")
            else:
                unit = result.get("unit", "unit not provided")
                lines.append(
                    f"- {metric}: min {result.get('min')} {unit}, max {result.get('max')} {unit}, "
                    f"mean {result.get('mean')} {unit}; maximum at {result.get('max_at')}.{citation}"
                )

    for group, title in (("datasheet_rag", "Retrieved datasheet passages"), ("schematic_rag", "Retrieved diagram-caption passages")):
        result = evidence.get(group, {})
        passages = result.get("evidence", []) if isinstance(result, dict) else []
        lines.extend(["", f"{title}:"])
        if not passages:
            lines.append("- No matching passages were returned.")
            continue
        for item in passages[:5]:
            ref = next((key for key, value in catalog.items() if value.get("source") == item.get("source") and value.get("page") == item.get("page") and value.get("content") == item.get("content")), None)
            citation = f" [{ref}]" if ref else ""
            source = item.get("source", "unknown source")
            page = item.get("page", "unknown page")
            text = str(item.get("content", "")).replace("\n", " ").strip()
            lines.append(f"- Candidate from {source}, page {page}: {text}{citation}")
        if group == "schematic_rag":
            lines.append("These captions are retrieval candidates; they do not prove that the described board is installed in the reported system.")

    motor_report = any(word in description.lower() for word in ("motor", "stall", "stalls"))
    lines.extend(["", "Conclusion:"])
    if motor_report:
        lines.append("The available evidence does not establish why the motor stalls. The log has no motor-current or motor-voltage channels, and the retrieved documents do not by themselves confirm the installed motor or driver. Do not infer overcurrent, undervoltage, thermal shutdown, or a wiring fault from this evidence alone.")
        lines.extend(["", "Next measurements:", "- Capture synchronized motor current, motor supply voltage, motor speed, and driver fault/temperature data during the stall.", "- Confirm the motor, driver, and controller models and retrieve their matching datasheets and wiring schematic.", "- Inspect or test energized equipment only using qualified personnel and the equipment's safety procedures."])
    else:
        lines.append("The evidence summary does not determine a root cause. Compare measured values with a verified datasheet for the confirmed hardware and collect measurements specific to the failure mechanism before assigning a cause.")
    lines.extend(["", "Gemini service errors:", *[f"- {error}" for error in errors]])
    return "\n".join(lines)


class _GeminiUnavailable(RuntimeError):
    """All configured Gemini model endpoints were unavailable."""


def _mcp_server_config() -> dict:
    """Register the three local stdio MCP servers for one FastMCP client."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    rag_server = ROOT / "diagnostic_rag" / "search_server.py"
    return {
        "mcpServers": {
            "datasheet_rag": {
                "command": sys.executable,
                "args": [str(rag_server)],
                "env": {**env, "RAG_SERVER": "datasheet"},
            },
            "schematic_rag": {
                "command": sys.executable,
                "args": [str(rag_server)],
                "env": {**env, "RAG_SERVER": "schematic"},
            },
            "sensor_tools": {
                "command": sys.executable,
                "args": [str(ROOT / "sensor_tool" / "sensor_tool.py")],
                "env": env,
            },
        }
    }


def _gemini_generate(client, contents, config, preferred_model: str):
    model_order = list(dict.fromkeys([
        preferred_model,
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ]))
    unavailable = []
    for model in model_order:
        try:
            print(f"[diagnostic] Waiting for Gemini ({model})...", file=sys.stderr, flush=True)
            return client.models.generate_content(model=model, contents=contents, config=config), model
        except ServerError as exc:
            # Try another configured model on transient service and deadline errors.
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            unavailable.append(f"{model}: {exc}")
    raise _GeminiUnavailable(" | ".join(unavailable))


def _mcp_payload(result) -> object:
    """Convert an MCP result into JSON-friendly data for Gemini and citations."""
    if result.is_error:
        return {"error": True, "content": [getattr(block, "text", "") for block in result.content]}
    if result.structured_content is not None:
        return result.structured_content
    blocks = [getattr(block, "text", "") for block in result.content]
    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except (json.JSONDecodeError, TypeError):
            parsed.append(block)
    return parsed[0] if len(parsed) == 1 else parsed


async def _ask_with_mcp_tools(description: str, failure_time: str | None) -> dict:
    """Let Gemini choose and call tools exposed by all three MCP servers."""
    print("[diagnostic] Starting the three MCP servers...", file=sys.stderr, flush=True)
    # Bound MCP startup and request waits so a child server cannot leave the CLI
    # apparently idle forever.
    mcp_client = Client(_mcp_server_config(), timeout=30, init_timeout=30)
    genai_client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    system_instruction = f"""You are a careful hardware diagnostic agent investigating: {description}
Failure time: {failure_time or 'not supplied'}.
You can call tools from sensor_tools, datasheet_rag, and schematic_rag. Choose tools based on the failure; do not call every tool automatically. For a time-specific failure, inspect sensor readings/anomalies around that time first, then search relevant datasheet limits and relevant circuit/pin connections as needed. You may call multiple tools and continue for several rounds.
For a hardware-failure diagnosis, use the sensor tools to establish available channels and inspect relevant measurements before drawing conclusions. Search datasheets/schematics only when useful to the identified or candidate hardware; a schematic search with no motor/driver connection is not relevant evidence.
The sensor CSV currently contains temperature, pressure, and humidity; current and voltage may be absent. Unknown-metric tool errors mean the channel is unavailable, not that its value is zero.
Use only tool results and the user description for factual claims. Cite tool evidence IDs like [E1]. Treat retrieved hardware documents as candidates unless the user confirms those exact parts are installed. Omit unrelated documents. Do not claim causation from temporal coincidence, do not rank unsupported causes, and do not invent units or specifications.
Separate measurements, matching document facts, unconfirmed possibilities, and missing evidence. If the evidence cannot establish a root cause, say so. Recommend only safe, general next steps; do not prescribe electrical tests for unidentified hardware."""

    async with mcp_client:
        print("[diagnostic] MCP servers connected; discovering available tools...", file=sys.stderr, flush=True)
        mcp_tools = await mcp_client.list_tools()
        print(f"[diagnostic] Discovered {len(mcp_tools)} MCP tools.", file=sys.stderr, flush=True)
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters_json_schema=tool.input_schema,
            )
            for tool in mcp_tools
        ]
        tool_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=declarations)],
        )
        contents = [types.Content(role="user", parts=[types.Part(text=description + (f"\nFailure time: {failure_time}" if failure_time else ""))])]
        evidence = {}
        next_evidence_id = 1
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        response = None
        selected_model = preferred_model

        for _round in range(8):
            response, selected_model = _gemini_generate(genai_client, contents, tool_config, preferred_model)
            function_calls = response.function_calls or []
            if not function_calls:
                break

            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)

            for function_call in function_calls:
                arguments = dict(function_call.args or {})
                print(f"[diagnostic] Gemini selected MCP tool: {function_call.name}", file=sys.stderr, flush=True)
                try:
                    tool_result = await mcp_client.call_tool(
                        function_call.name,
                        arguments,
                        raise_on_error=False,
                    )
                    payload = _mcp_payload(tool_result)
                except Exception as exc:
                    payload = {"error": str(exc)}

                evidence_id = f"E{next_evidence_id}"
                next_evidence_id += 1
                evidence_record = {
                    "evidence_id": evidence_id,
                    "tool": function_call.name,
                    "arguments": arguments,
                    "result": payload,
                }
                evidence[evidence_id] = evidence_record
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=function_call.name,
                            response={"result": evidence_record},
                        )],
                    )
                )
        else:
            # Give Gemini one tool-free chance to summarize when it exhausts the tool-round limit.
            final_config = types.GenerateContentConfig(system_instruction=system_instruction)
            response, selected_model = _gemini_generate(genai_client, contents, final_config, selected_model)

    return {
        "failure_description": description,
        "failure_time": failure_time,
        "model": selected_model,
        "generation_mode": "gemini_mcp_tool_calling",
        "diagnosis": response.text or "Gemini returned no text.",
        "evidence": evidence,
        "raw_tool_evidence": evidence,
    }


async def diagnose(description: str, failure_time: str | None = None, use_gemini: bool = True) -> dict:
    if not use_gemini:
        evidence = await gather_evidence(description, failure_time)
        catalog, _ = _evidence_catalog(evidence)
        report = _local_evidence_report(description, failure_time, evidence, catalog, [])
        return {
            "failure_description": description,
            "failure_time": failure_time,
            "model": None,
            "generation_mode": "local_evidence_only",
            "diagnosis": report,
            "evidence": catalog,
            "raw_tool_evidence": evidence,
        }
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required unless --local-only is used")
    try:
        return await _ask_with_mcp_tools(description, failure_time)
    except _GeminiUnavailable as exc:
        evidence = await gather_evidence(description, failure_time)
        catalog, _ = _evidence_catalog(evidence)
        unavailable = str(exc).split(" | ")
        report = _local_evidence_report(description, failure_time, evidence, catalog, unavailable)
        return {
            "failure_description": description,
            "failure_time": failure_time,
            "model": None,
            "generation_mode": "local_evidence_fallback",
            "diagnosis": report,
            "evidence": catalog,
            "raw_tool_evidence": evidence,
            "generation_errors": unavailable,
        }
    except Exception as exc:
        raise RuntimeError(
            "The MCP/Gemini integration stopped before producing a diagnosis. "
            "Check the [diagnostic] progress line above; Gemini requests have a 30-second "
            f"per-request timeout. Original error: {type(exc).__name__}: {exc}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", help="Observed failure description")
    parser.add_argument("--failure-time", help="ISO timestamp, e.g. 2026-10-08 13:15:00")
    parser.add_argument("--json", action="store_true", help="Print diagnosis and evidence as JSON")
    parser.add_argument("--local-only", action="store_true", help="Skip Gemini and print a conservative evidence report")
    args = parser.parse_args()
    try:
        result = asyncio.run(diagnose(args.description, args.failure_time, use_gemini=not args.local_only))
    except Exception as exc:
        details = "".join(traceback.format_exception(exc))
        parser.exit(1, f"Diagnostic failed. Full exception details follow:\n{details}\n")
    if hasattr(sys.stdout, "reconfigure"):
        # Retrieved PDF text can contain symbols the default Windows code page cannot print.
        sys.stdout.reconfigure(errors="backslashreplace")
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else result["diagnosis"])


if __name__ == "__main__":
    main()
