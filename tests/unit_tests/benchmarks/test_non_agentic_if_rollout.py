# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
