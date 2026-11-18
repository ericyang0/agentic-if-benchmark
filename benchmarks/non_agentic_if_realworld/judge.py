# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the direct Sol judge and preserve the distinction between uncertainty and failure."""

import asyncio
import json
from pathlib import Path

import jsonschema

from .delivery import POLICIES, PROTOCOL, canonical_bytes, canonical_hash
from .prepare import private_directory, records
from .scoring import validate_review


MODEL = "gpt-6.2-sol"
DISABLED_FEATURES = (
    "shell_tool",
    "multi_agent",
    "sleep_tool",
    "skill_search",
    "skill_mcp_dependency_install",
    "memories",
    "apps",
    "plugins",
)


def _judge_command(schema: Path, output: Path) -> list[str]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--model",
        MODEL,
        "-c",
        'model_reasoning_effort="medium"',
        "-c",
        'model_verbosity="low"',
        "-c",
        'web_search="disabled"',
        "-c",
        "model_context_window=1050000",
        "-c",
        "model_auto_compact_token_limit=1040000",
    ]
    command.extend(option for feature in DISABLED_FEATURES for option in ("--disable", feature))
    return [
        *command,
        "--enable",
        "skip_host_skill_discovery",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(output),
        "--json",
        "--color",
        "never",
        "-",
    ]


def _validated_event_log(path: Path) -> None:
    turns = {"turn.started": 0, "turn.completed": 0}
    for event in records(path):
        kind = event["type"]
        item = event.get("item", {}).get("type")
        if kind in turns:
            turns[kind] += 1
        if kind == "item.completed" and item == "error" and turns["turn.started"] == 0:
            warning = event["item"].get("message", "")
            if warning.startswith("Under-development features enabled: skip_host_skill_discovery."):
                continue
        if "compact" in kind or item not in (None, "reasoning", "agent_message") or kind in {"error", "turn.failed"}:
            raise ValueError("Unusable judge event log")
    if set(turns.values()) != {1}:
        raise ValueError("Judge must complete one tool-free turn")


async def assess(payload: dict, asset_directory: Path, evidence_directory: Path) -> dict:
    """Return an assessed grade or an explicit unscored result, including on interrupted resume."""
    rubric_name, schema_name, _, _ = POLICIES[payload["policy_version"].removeprefix(PROTOCOL)]
    schema_path = asset_directory / schema_name
    schema = json.loads(schema_path.read_bytes())
    prompt = (asset_directory / rubric_name).read_text() + "\nUNTRUSTED EVALUATION PAYLOAD:\n"
    prompt += json.dumps(payload, ensure_ascii=False)
    root = private_directory(evidence_directory)
    binding = {"payload_sha256": canonical_hash(payload), "prompt_sha256": canonical_hash(prompt), "model": MODEL}
    result_file = root / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_bytes())
        if result["binding"] != binding:
            raise ValueError("Saved judge result has a different input binding")
        if "response" in result and validate_review(result["response"], payload, schema)["grade"] != result["grade"]:
            raise ValueError("Saved judgment differs from its validated response")
        return result

    for number in range(1, 5):
        attempt = root / f"attempt-{number}"
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError:
            # An interrupted request may have reached the judge. Keep its receipts for recovery.
            saved = attempt / "binding.json"
            if saved.exists() and json.loads(saved.read_bytes()) != binding:
                raise ValueError("Existing judge attempt has a different input binding")
            if not (attempt / "technical-failure").exists():
                return {"binding": binding, "grade": None, "status": "interrupted_assessment"}
            continue
        neutral = attempt / "cwd"
        neutral.mkdir(mode=0o700)
        response_path = attempt / "response.json"
        events_path = attempt / "events.jsonl"
        (attempt / "binding.json").write_bytes(canonical_bytes(binding) + b"\n")
        try:
            with events_path.open("xb") as events, (attempt / "stderr.log").open("xb") as errors:
                process = await asyncio.create_subprocess_exec(
                    *_judge_command(schema_path, response_path),
                    cwd=neutral,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=events,
                    stderr=errors,
                )
                try:
                    await asyncio.wait_for(process.communicate(prompt.encode()), timeout=1800)
                except (TimeoutError, asyncio.CancelledError):
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                    raise
            if process.returncode != 0:
                raise ValueError("Judge command failed")
            _validated_event_log(events_path)
            response = json.loads(response_path.read_bytes())
            grade = validate_review(response, payload, schema)["grade"]
        except (OSError, TimeoutError, ValueError, KeyError, jsonschema.ValidationError):
            (attempt / "technical-failure").touch(exist_ok=False)
        else:
            result = {
                "binding": binding,
                "grade": grade,
                "response": response,
                "attempt": number,
                "status": "graded" if grade is not None else "needs_adjudication",
            }
            break
    else:
        result = {"binding": binding, "grade": None, "status": "technical_attempts_exhausted"}
    staging = root / ".result.json"
    with staging.open("xb") as stream:
        stream.write(canonical_bytes(result) + b"\n")
    staging.replace(result_file)
    return result
