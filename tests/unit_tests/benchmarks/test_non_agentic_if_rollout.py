# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError

from benchmarks.non_agentic_if_realworld import judge as judge_runtime
from benchmarks.non_agentic_if_realworld import rollout_driver
from benchmarks.non_agentic_if_realworld.delivery import canonical_bytes, canonical_hash


@pytest.mark.asyncio
async def test_first_complete_stream_is_scored_once_and_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = [{"role": "user", "name": "planner", "content": "Say Cedar."}]
    baseline = {"model": "frozen-model", "messages": messages, "temperature": 0.7, "seed": 17, "stream": False}
    row = {
        "id": "independent-synthetic-case",
        "baseline_request": baseline,
        "baseline_provenance": {"input_sha256": canonical_hash({"messages": messages})},
        "membership": {"original_split": "train", "reserved_sep4": False, "benchmark_role": "development_exposed"},
    }
    wire = {**baseline, "model": "policy", "max_tokens": 32768, "stream": True}
    event = {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion.chunk",
        "created": 2026,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Cedar.", "reasoning_content": "Private reasoning."},
                "finish_reason": "stop",
            }
        ],
    }
    raw = b"data: " + canonical_bytes(event) + b"\n\ndata: [DONE]\n\n"
    raw_hash = hashlib.sha256(raw).hexdigest()
    receipt = {"request": wire, "response_sha256": raw_hash}
    capture_hash = canonical_hash(receipt)
    (tmp_path / f"{raw_hash}.sse").write_bytes(raw)
    (tmp_path / f"{capture_hash}.json").write_bytes(canonical_bytes(receipt))
    completion = {
        "id": event["id"],
        "object": "chat.completion",
        "created": 2026,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>Private reasoning.</think>Cedar.",
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                },
                "finish_reason": "stop",
            }
        ],
        "metrics": {"upstream_chat_capture_sha256": capture_hash, "upstream_chat_stream_sha256": raw_hash},
    }
    generate = AsyncMock(return_value=completion)
    judge = AsyncMock(return_value={"grade": "pass", "status": "graded"})
    monkeypatch.setattr(rollout_driver, "_completion", generate)
    monkeypatch.setattr(rollout_driver, "assess", judge)
    engine = AsyncMock()
    engine.create_tokenize.return_value = {"count": 100}
    kwargs = {
        "client": AsyncMock(),
        "engine": engine,
        "model": "policy",
        "context": 131072,
        "manifest_hash": "a" * 64,
        "caps": [32768, 65536, 131072],
    }
    criteria = [{"criterion_id": "synthetic-criterion", "mechanical_checks": []}]
    result = await rollout_driver._evaluate(row, criteria, tmp_path / "work", tmp_path, tmp_path, **kwargs)
    resumed = await rollout_driver._evaluate(row, criteria, tmp_path / "work", tmp_path, tmp_path, **kwargs)

    assert result == resumed
    assert result["reward"] == 1
    assert result["mask_sample"] is False
    assert "Private reasoning." in json.dumps(result["response"])
    assert any(item.get("generation_token_ids") == [3, 4] for item in result["response"]["output"])
    assert any(item.get("generation_log_probs") == [-0.1, -0.2] for item in result["response"]["output"])
    generate.assert_awaited_once()
    judge.assert_awaited_once()
    projection = judge.call_args.args[0]
    assert projection["input_request"]["messages"] == messages
    assert projection["new_response"] == {"content": "Cedar.", "refusal": ""}


@pytest.mark.asyncio
async def test_hub_judge_validates_and_resumes_with_its_own_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {"type": "object", "required": ["item_id", "criteria"]}
    (tmp_path / "judge.schema.json").write_text(json.dumps(schema))
    (tmp_path / "rubric.md").write_text("Assess the single visible instruction.")
    payload = {
        "item_id": "synthetic",
        "policy_version": "non-agentic-if-strict-response-grading-v2.2",
        "input_request": {"messages": [{"role": "user", "content": "Say Cedar."}]},
        "new_response": {"content": "Cedar.", "refusal": ""},
        "criteria": [{"criterion_id": "cedar", "instruction_anchors": ["Say Cedar."]}],
    }
    review = {
        "item_id": "synthetic",
        "criteria": [
            {
                "criterion_id": "cedar",
                "anchor_index": 0,
                "assessment_status": "decided",
                "grade": "pass",
                "applicability": "active",
                "materiality": "none",
                "evidence_kind": "compliance",
                "response_field": "content",
                "response_quote": "Cedar.",
                "exception_input_quote": "",
                "review_reason": "",
            }
        ],
    }
    envelope = {
        "model": judge_runtime.HUB_MODEL,
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(review)}}],
    }
    response = AsyncMock()
    response.status = 200
    response.read.return_value = canonical_bytes(envelope)
    transport = AsyncMock(return_value=response)
    monkeypatch.setenv(judge_runtime.HUB_KEY_ENV, "test-token")
    monkeypatch.setattr(judge_runtime, "request", transport)
    work = tmp_path / "hub"

    result = await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE)
    assert result["grade"] == "pass"
    assert result["binding"]["profile"] == judge_runtime.INFERENCE_HUB_PROFILE
    assert await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE) == result
    with pytest.raises(ValueError, match="different input binding"):
        await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.DIRECT_CODEX_PROFILE)
    transport.assert_awaited_once()
    assert transport.call_args.args == ("POST", judge_runtime.HUB_ENDPOINT)
    assert transport.call_args.kwargs["allow_redirects"] is False
    assert transport.call_args.kwargs["json"]["reasoning_effort"] == "medium"
    assert transport.call_args.kwargs["json"]["response_format"]["json_schema"]["schema"] == schema

    transport.reset_mock(side_effect=True)
    transport.side_effect = ClientConnectionError("ambiguous dispatch")
    work = tmp_path / "disconnect"
    result = await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE)
    assert result["grade"] is None
    assert result["status"] == "interrupted_assessment"
    assert await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE) == result
    transport.assert_awaited_once()
