# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the selected Sol scoring profile and preserve its assessment receipts."""

import asyncio
import json
import os
import shutil
from pathlib import Path

import jsonschema
from aiohttp import ClientError, ClientTimeout

from nemo_gym.client import request

from .delivery import POLICIES, PROTOCOL, canonical_bytes, canonical_hash
from .prepare import private_directory, records, write_private_bytes
from .scoring import validate_review


MODEL = "gpt-6.2-sol"
DIRECT_CODEX_PROFILE = "direct_codex_v1"
INFERENCE_HUB_PROFILE = "inference_hub_v1"
HUB_ENDPOINT = "https://inference-api.nvidia.com/v1/chat/completions"
HUB_MODEL = "openai/openai/gpt-6.2-sol"
HUB_KEY_ENV = "NON_AGENTIC_IF_JUDGE_API_KEY"
REASONING = "medium"
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


def judge_settings(profile: str) -> dict[str, str]:
    """Identify the judge independently of the credential used to access it."""
    if profile == DIRECT_CODEX_PROFILE:
        return {"profile": profile, "model": MODEL, "reasoning_effort": REASONING, "verbosity": "low"}
    if profile == INFERENCE_HUB_PROFILE:
        return {"profile": profile, "model": HUB_MODEL, "reasoning_effort": REASONING, "endpoint": HUB_ENDPOINT}
    raise ValueError(f"Unknown Non-Agentic IF judge profile: {profile}")


def check_judge_configuration(profile: str) -> dict[str, str]:
    """Check the selected transport before generating policy responses."""
    settings = judge_settings(profile)
    if profile == DIRECT_CODEX_PROFILE:
        if shutil.which("codex") is None:
            raise ValueError("Direct GPT-6.2 Sol grading requires the Codex CLI")
    elif os.getenv("NON_AGENTIC_IF_JUDGE_ENDPOINT", HUB_ENDPOINT) != HUB_ENDPOINT or not os.getenv(HUB_KEY_ENV):
        raise ValueError(f"Inference Hub grading requires the approved endpoint and {HUB_KEY_ENV}")
    return settings


async def _hub_review(prompt: str, schema: dict, attempt: Path) -> dict:
    body = {
        "model": HUB_MODEL,
        "reasoning_effort": REASONING,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "non_agentic_if_strict_review", "strict": True, "schema": schema},
        },
    }
    response = await request(
        "POST",
        HUB_ENDPOINT,
        headers={"Authorization": "Bearer " + os.environ[HUB_KEY_ENV]},
        json=body,
        timeout=ClientTimeout(total=1800, connect=60),
        allow_redirects=False,
        _max_num_tries=1,
        _resend_after_disconnect=False,
    )
    async with response:
        raw = await response.read()
        status = response.status
    (attempt / "hub-response.json").write_bytes(raw)
    (attempt / "http-status").write_text(f"{status}\n")
    if status != 200:
        raise ValueError(f"Inference Hub returned HTTP {status}")
    completion = json.loads(raw)
    if completion["model"] != HUB_MODEL:
        raise ValueError("Inference Hub returned an unexpected judge model")
    (choice,) = completion["choices"]
    message = choice["message"]
    if (
        choice["finish_reason"] != "stop"
        or choice.get("native_finish_reason") == "repetition"
        or choice.get("stop_reason") == "repetition_detected"
        or message["role"] != "assistant"
        or any(message.get(field) for field in ("refusal", "tool_calls", "function_call"))
    ):
        raise ValueError("Incomplete or unsupported judge response")
    return json.loads(message["content"])


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


async def _codex_review(prompt: str, schema_path: Path, response_path: Path) -> dict:
    attempt = response_path.parent
    neutral = attempt / "cwd"
    neutral.mkdir(mode=0o700)
    events_path = attempt / "events.jsonl"
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
    return json.loads(response_path.read_bytes())


async def assess(
    payload: dict, asset_directory: Path, evidence_directory: Path, *, profile: str = DIRECT_CODEX_PROFILE
) -> dict:
    """Return an assessed grade or an explicit unscored result, including on interrupted resume."""
    rubric_name, schema_name, _, _ = POLICIES[payload["policy_version"].removeprefix(PROTOCOL)]
    schema_path = asset_directory / schema_name
    schema = json.loads(schema_path.read_bytes())
    prompt = (asset_directory / rubric_name).read_text() + "\nUNTRUSTED EVALUATION PAYLOAD:\n"
    prompt += json.dumps(payload, ensure_ascii=False)
    root = private_directory(evidence_directory)
    binding = {
        "payload_sha256": canonical_hash(payload),
        "prompt_sha256": canonical_hash(prompt),
        **judge_settings(profile),
    }
    result_file = root / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_bytes())
        if result["binding"] != binding:
            raise ValueError("Saved judge result has a different input binding")
        if "response" in result:
            validated = validate_review(result["response"], payload, schema)
            if any(result[field] != validated[field] for field in ("grade", "status")):
                raise ValueError("Saved judgment differs from its validated response")
        return result

    for number in range(1, 5):
        attempt = root / f"attempt-{number}"
        response_path = attempt / "response.json"
        saved = attempt / "binding.json"
        recovering = False
        try:
            attempt.mkdir(mode=0o700)
        except FileExistsError:
            # An interrupted request may have reached the judge. Keep its receipts for recovery.
            if not saved.is_file():
                return {"binding": binding, "grade": None, "status": "interrupted_assessment"}
            if json.loads(saved.read_bytes()) != binding:
                raise ValueError("Existing judge attempt has a different input binding")
            if (attempt / "technical-failure").exists():
                continue
            if not response_path.is_file():
                return {"binding": binding, "grade": None, "status": "interrupted_assessment"}
            recovering = True
        else:
            write_private_bytes(saved, canonical_bytes(binding) + b"\n")
        try:
            if recovering:
                if profile == DIRECT_CODEX_PROFILE:
                    _validated_event_log(attempt / "events.jsonl")
                response = json.loads(response_path.read_bytes())
            elif profile == INFERENCE_HUB_PROFILE:
                try:
                    response = await _hub_review(prompt, schema, attempt)
                except (ClientError, TimeoutError):
                    return {"binding": binding, "grade": None, "status": "interrupted_assessment"}
                write_private_bytes(response_path, canonical_bytes(response) + b"\n")
            else:
                response = await _codex_review(prompt, schema_path, response_path)
            validated = validate_review(response, payload, schema)
        except (OSError, TimeoutError, TypeError, ValueError, KeyError, jsonschema.ValidationError):
            if recovering:
                return {"binding": binding, "grade": None, "status": "interrupted_assessment"}
            (attempt / "technical-failure").touch(exist_ok=False)
            status_path = attempt / "http-status"
            if status_path.is_file():
                status = int(status_path.read_text())
                if 300 <= status < 500 and status not in (408, 429):
                    result = {"binding": binding, "grade": None, "status": "judge_rejected_request", "attempt": number}
                    break
                if status == 429 and number < 4:
                    await asyncio.sleep(65)
        else:
            if validated["status"] == "validation_error":
                (attempt / "technical-failure").touch(exist_ok=False)
                continue
            result = {
                "binding": binding,
                "grade": validated["grade"],
                "response": response,
                "attempt": number,
                "status": validated["status"],
            }
            break
    else:
        result = {"binding": binding, "grade": None, "status": "technical_attempts_exhausted"}
    write_private_bytes(result_file, canonical_bytes(result) + b"\n")
    return result
