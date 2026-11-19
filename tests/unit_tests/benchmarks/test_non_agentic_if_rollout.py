# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError, ConnectionTimeoutError
from httpx import ASGITransport, AsyncClient

from benchmarks.non_agentic_if import judge as judge_runtime
from benchmarks.non_agentic_if import rollout_driver
from benchmarks.non_agentic_if.delivery import canonical_bytes, canonical_hash, visible_delivery
from benchmarks.non_agentic_if.scoring import validate_review
from environment_servers.non_agentic_if.app import NonAgenticIFEnvironmentServer
from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.models.model_call_capture import reconstruct_upstream_chat_completion


@pytest.mark.parametrize(
    "field,marker", [("native_finish_reason", "repetition"), ("stop_reason", "repetition_detected")]
)
def test_provider_repetition_is_not_scored_as_a_complete_answer(field: str, marker: str) -> None:
    chunk = {"object": "chat.completion.chunk", "id": "synthetic", "model": "policy", "created": 2026}
    events = [
        chunk | {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Cedar"}, field: marker}]},
        chunk | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", field: "other"}]},
    ]
    raw = b"".join(b"data: " + canonical_bytes(event) + b"\n\n" for event in events) + b"data: [DONE]\n\n"
    with pytest.raises(ValueError, match="repetition"):
        visible_delivery(reconstruct_upstream_chat_completion(raw), {})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation", "conv_fixture"),
        ("conversation", {"id": "conv_fixture"}),
        ("conversation", ""),
        ("previous_response_id", "resp_fixture"),
        ("prompt", {"id": "prompt_fixture"}),
        ("instructions", "Use a different system instruction."),
        ("tools", [{"type": "function", "function": {"name": "fixture"}}]),
    ],
)
def test_frozen_policy_fields_reject_provider_context_and_tools(field: str, value: object) -> None:
    messages = [{"role": "user", "content": "Say Cedar."}]
    row = {
        "baseline_request": {"messages": messages, "model": "fixture", field: value},
        "baseline_provenance": {"input_sha256": canonical_hash({"messages": messages})},
    }
    with pytest.raises(ValueError, match="provider-managed context or tool grants"):
        rollout_driver._policy_request(row, "test-model")
    del row["baseline_request"][field]
    assert rollout_driver._policy_request(row, "test-model") == {"messages": messages, "model": "test-model"}


@pytest.mark.parametrize("unsent", [True, False])
async def test_policy_dispatch_resumes_only_if_known_unsent(tmp_path, monkeypatch, unsent: bool) -> None:
    error_type = ConnectionTimeoutError if unsent else ClientConnectionError
    response = AsyncMock()
    response.raise_for_status = lambda: None
    client = SimpleNamespace(request=AsyncMock(side_effect=[error_type("disconnected"), response]))
    monkeypatch.setattr(rollout_driver, "get_response_json", AsyncMock(return_value={"id": "synthetic"}))
    attempt = tmp_path / "attempt"
    request = {"model": "policy", "seed": 17, "max_tokens": 32768}
    with pytest.raises(RuntimeError, match="before dispatch" if unsent else "ambiguous policy dispatch"):
        await rollout_driver._completion(client, request, attempt)
    if unsent:
        assert await rollout_driver._completion(client, request, attempt) == {"id": "synthetic"}
        assert await rollout_driver._completion(client, request, attempt) == {"id": "synthetic"}
        assert client.request.await_count == 2
    else:
        with pytest.raises(RuntimeError, match="ambiguous policy dispatch"):
            await rollout_driver._completion(client, request, attempt)
        client.request.assert_awaited_once()
    for call in client.request.await_args_list:
        assert call.kwargs["json"] == request
        assert call.kwargs["_max_num_tries"] == 1
        assert call.kwargs["_resend_after_disconnect"] is False


async def test_environment_reaggregates_saved_outcomes_without_running_tasks() -> None:
    app = NonAgenticIFEnvironmentServer.model_construct().setup_webserver()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        results = [
            {
                TASK_INDEX_KEY_NAME: index,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "task_id": f"synthetic-{index}",
                "reward": float(grade == "pass"),
                "mask_sample": grade is None,
                "grade": grade,
            }
            for index, grade in enumerate(("pass", "fail", None))
        ]
        response = await client.post("/aggregate_metrics", json={"verify_responses": results})
        response.raise_for_status()
        summary = response.json()
        assert summary["key_metrics"]["composite/complete_accuracy"] is None
        assert summary["agent_metrics"]["composite/accuracy_among_scored"] == 0.5
        assert summary["agent_metrics"]["count/unattempted"] == 568
        assert summary["agent_metrics"]["count/unscored"] == 1
        assert len(summary["group_level_metrics"]) == 2
        assert (await client.post("/run", json={})).status_code == 409


@pytest.mark.parametrize("status, model", [(200, judge_runtime.HUB_MODEL), (200, "another-model"), (401, "")])
async def test_hub_preflight_checks_access_without_a_judge_dispatch(monkeypatch, status: int, model: str) -> None:
    monkeypatch.setenv(judge_runtime.HUB_KEY_ENV, "test-token")
    response = AsyncMock(status=status)
    response.read.return_value = canonical_bytes({"data": [{"id": model}]})
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(judge_runtime, "request", request)
    if model == judge_runtime.HUB_MODEL:
        await judge_runtime.check_judge_access(judge_runtime.INFERENCE_HUB_PROFILE)
    else:
        with pytest.raises(ValueError, match="access"):
            await judge_runtime.check_judge_access(judge_runtime.INFERENCE_HUB_PROFILE)
    request.assert_awaited_once()
    assert request.call_args.args[0] == "GET"
    assert request.call_args.args[1].endswith("/v1/models")


async def test_direct_judge_checks_login_before_generation(monkeypatch) -> None:
    start = AsyncMock(return_value=SimpleNamespace(wait=AsyncMock(return_value=1)))
    monkeypatch.setattr(judge_runtime.asyncio, "create_subprocess_exec", start)
    with pytest.raises(ValueError, match="authenticated Codex CLI"):
        await judge_runtime.check_judge_access(judge_runtime.DIRECT_CODEX_PROFILE)
    assert start.call_args.args == ("codex", "login", "status")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"upload_rollouts": True}, "keep rollouts private"),
        ({"task_ids": ["synthetic"]}, "with --limit"),
        ({"task_ids_fpath": "synthetic.txt"}, "with --limit"),
        ({"limit": 0}, "positive --limit"),
    ],
)
async def test_unsafe_run_settings_fail_before_model_setup(overrides: dict, error: str) -> None:
    defaults = {
        "upload_rollouts": False,
        "num_repeats": 1,
        "fan_out": False,
        "responses_create_params": {},
        "task_ids": None,
        "task_ids_fpath": None,
        "limit": None,
    }
    with pytest.raises(ValueError, match=error):
        await rollout_driver.run_rollout_collection(SimpleNamespace(**(defaults | overrides)), {})


@pytest.mark.asyncio
@pytest.mark.parametrize("grade", ["pass", "fail"])
async def test_first_complete_stream_is_scored_once_and_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grade: str
) -> None:
    messages = [{"role": "user", "name": "planner", "content": "Say Cedar."}]
    baseline = {"model": "frozen-model", "messages": messages, "temperature": 0.7, "seed": 17, "stream": False}
    row = {
        "id": "independent-synthetic-case",
        "baseline_request": baseline,
        "baseline_provenance": {"input_sha256": canonical_hash({"messages": messages})},
        "membership": {"original_split": "train", "reserved_sep4": False, "benchmark_role": "development_exposed"},
    }
    wire = {**baseline, "model": "policy", "max_tokens": 65536, "stream": True}
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
    incomplete_event = event | {"choices": [event["choices"][0] | {"finish_reason": "length"}]}
    incomplete_raw = b"data: " + canonical_bytes(incomplete_event) + b"\n\ndata: [DONE]\n\n"
    incomplete_hash = hashlib.sha256(incomplete_raw).hexdigest()
    incomplete_receipt = {"request": wire | {"max_tokens": 32768}, "response_sha256": incomplete_hash}
    incomplete_capture = canonical_hash(incomplete_receipt)
    (tmp_path / f"{incomplete_hash}.sse").write_bytes(incomplete_raw)
    (tmp_path / f"{incomplete_capture}.json").write_bytes(canonical_bytes(incomplete_receipt))
    incomplete = completion | {
        "metrics": {
            "upstream_chat_capture_sha256": incomplete_capture,
            "upstream_chat_stream_sha256": incomplete_hash,
        },
    }
    completions = iter([incomplete, completion])

    async def deliver(_client, _request, directory):
        response = next(completions)
        directory.mkdir()
        (directory / "completion.json").write_bytes(canonical_bytes(response))
        return response

    generate = AsyncMock(side_effect=deliver)
    judge = AsyncMock(return_value={"grade": grade, "status": "graded"})
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
    assert result["reward"] == float(grade == "pass")
    assert result["mask_sample"] is False
    assert "Private reasoning." in json.dumps(result["response"])
    assert any(item.get("generation_token_ids") == [3, 4] for item in result["response"]["output"])
    assert any(item.get("generation_log_probs") == [-0.1, -0.2] for item in result["response"]["output"])
    assert [call.args[1]["max_tokens"] for call in generate.await_args_list] == [32768, 65536]
    assert all(call.args[1]["seed"] == 17 for call in generate.await_args_list)
    judge.assert_awaited_once()
    projection = judge.call_args.args[0]
    assert projection["input_request"]["messages"] == messages
    assert projection["new_response"] == {"content": "Cedar.", "refusal": ""}

    capture_path = tmp_path / f"{raw_hash}.sse"
    capture_path.write_bytes(raw + b"corruption")
    with pytest.raises(ValueError, match="original receipts"):
        await rollout_driver._evaluate(row, criteria, tmp_path / "work", tmp_path, tmp_path, **kwargs)
    capture_path.write_bytes(raw)
    saved_path = tmp_path / "work" / "result.json"
    saved = json.loads(saved_path.read_bytes())
    saved["result"]["output"]["reward"] = 1 - result["reward"]
    saved_path.write_bytes(canonical_bytes(saved))
    with pytest.raises(ValueError, match="original result or run binding"):
        await rollout_driver._evaluate(row, criteria, tmp_path / "work", tmp_path, tmp_path, **kwargs)
    assert generate.await_count == 2
    judge.assert_awaited_once()


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
    (work / "result.json").unlink()
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

    transport.reset_mock(side_effect=True)
    invalid = dict(review, criteria=[review["criteria"][0] | {"response_quote": "Pine."}])
    invalid_envelope = dict(
        envelope,
        choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(invalid)}}],
    )
    response.read.side_effect = [canonical_bytes(invalid_envelope), canonical_bytes(envelope)]
    result = await judge_runtime.assess(
        payload, tmp_path, tmp_path / "invalid", profile=judge_runtime.INFERENCE_HUB_PROFILE
    )
    assert result["grade"] == "pass"
    assert result["attempt"] == 2
    assert transport.await_count == 2

    transport.reset_mock()
    response.read.side_effect = None
    work = tmp_path / "access-recovery"
    for status in (401, 403, 404, 401):
        response.status = status
        response.read.return_value = b"{}"
        with pytest.raises(RuntimeError, match="denied judge access"):
            await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE)
        assert not (work / "result.json").exists()
    (work / "attempt-4" / "technical-failure").unlink()
    response.status = 200
    response.read.return_value = canonical_bytes(envelope)
    result = await judge_runtime.assess(payload, tmp_path, work, profile=judge_runtime.INFERENCE_HUB_PROFILE)
    assert result["grade"] == "pass" and result["attempt"] == 5
    assert transport.await_count == 5


@pytest.mark.parametrize("invalid_quote", [False, True])
def test_independent_failure_survives_an_invalid_criterion(invalid_quote: bool) -> None:
    payload = {
        "item_id": "synthetic",
        "input_request": {"messages": [{"role": "user", "content": "Say Cedar and name its colour."}]},
        "new_response": {"content": "Pine, green.", "refusal": ""},
        "criteria": [
            {"criterion_id": "cedar", "instruction_anchors": ["Say Cedar"]},
            {
                "criterion_id": "colour",
                "instruction_anchors": ["name its colour"],
                "depends_on": [] if invalid_quote else ["cedar"],
            },
        ],
    }
    evidence = {
        "anchor_index": 0,
        "assessment_status": "decided",
        "applicability": "active",
        "response_field": "content",
        "review_reason": "",
    }
    results = validate_review(
        {
            "item_id": "synthetic",
            "criteria": [
                evidence
                | {
                    "criterion_id": "cedar",
                    "grade": "fail",
                    "materiality": "major",
                    "evidence_kind": "direct_conflict",
                    "response_quote": "Pine",
                },
                evidence
                | {
                    "criterion_id": "colour",
                    "grade": "pass",
                    "materiality": "none",
                    "evidence_kind": "compliance",
                    "response_quote": "purple" if invalid_quote else "green",
                },
            ],
        },
        payload,
        {"type": "object"},
    )
    assert results["grade"] == "fail"
    assert results["status"] == "graded"
    assert [criterion["grade"] for criterion in results["criteria"]] == ["fail", None]


async def test_run_materializes_selected_tasks_before_dispatch_and_checks_resume(tmp_path, monkeypatch) -> None:
    pin = {"cases": 2, "artifacts": {"release_manifest.json": "a" * 64}}
    asset = tmp_path / "assets"
    asset.mkdir(mode=0o700)
    (asset / "source_manifest.json").write_text(json.dumps(pin))
    (asset / "release_manifest.json").write_text(json.dumps({"profile": {"context_bounded_output_caps": [32768]}}))
    ids = ("first-fictional", "second-fictional")
    prepared = [
        {"task_id": task, "responses_create_params": {"input": []}, "verifier_metadata": {"task_id": task}}
        for task in ids
    ]
    for name, rows in {
        "requests.jsonl": [{"id": task} for task in ids],
        "criteria.jsonl": [{"id": task, "criteria": []} for task in ids],
        "prepared.jsonl": prepared,
    }.items():
        (asset / name).write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
    output = tmp_path / "run" / "rollouts.jsonl"
    inventory = rollout_driver.materialized_path_for(output)
    config = SimpleNamespace(
        upload_rollouts=False,
        num_repeats=1,
        fan_out=False,
        responses_create_params={},
        task_ids=None,
        task_ids_fpath=None,
        limit=1,
        resume_from_cache=True,
        num_samples_in_parallel=1,
        disable_aggregation=True,
        input_jsonl_fpath=asset / "prepared.jsonl",
        output_jsonl_fpath=output,
    )
    policy = {
        "base_url": "http://localhost:8000/v1",
        "model": "policy",
        "api_key": "test-token",
        "upstream_chat_stream": True,
        "return_token_id_information": True,
        "upstream_chat_capture_dir": str(tmp_path / "captures"),
        "max_num_tries": 1,
        "max_http_attempts": 1,
    }
    monkeypatch.setattr(rollout_driver, "ROOT", asset)
    monkeypatch.setattr(rollout_driver, "digest", lambda _: "b" * 64)
    monkeypatch.setattr(rollout_driver, "fetch_release", lambda *_: asset)
    monkeypatch.setattr(rollout_driver, "prepared_rows", lambda *_: iter(prepared))
    monkeypatch.setattr(rollout_driver, "check_judge_configuration", lambda _: {})
    monkeypatch.setattr(rollout_driver.RolloutCollectionHelper, "setup_server_client", lambda self: None)
    engine = SimpleNamespace(
        create_models=AsyncMock(return_value={"data": [{"id": "policy", "max_model_len": 65536}]})
    )
    monkeypatch.setattr(rollout_driver, "NeMoGymAsyncOpenAI", lambda **_: engine)

    async def evaluate(row, *_args, **_kwargs):
        items = list(rollout_driver.records(inventory))
        assert [item["task_id"] for item in items] == [ids[0]]
        assert row["id"] == ids[0]
        return {"task_id": row["id"], "grade": "pass", "reward": 1, "mask_sample": False}

    evaluation = AsyncMock(side_effect=evaluate)
    monkeypatch.setattr(rollout_driver, "_evaluate", evaluation)
    resolved = {"policy_model": {"responses_api_models": {"vllm_model": policy}}}
    await rollout_driver.run_rollout_collection(config, resolved)
    (item,) = list(rollout_driver.records(output))
    assert item[TASK_INDEX_KEY_NAME] == 0 and item[ROLLOUT_INDEX_KEY_NAME] == 0
    assert inventory.stat().st_mode & 0o777 == 0o600
    inventory.write_text("{}\n")
    with pytest.raises(ValueError, match="inventory mismatch"):
        await rollout_driver.run_rollout_collection(config, resolved)
    evaluation.assert_awaited_once()
