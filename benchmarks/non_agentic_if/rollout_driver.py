# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect one assistant turn with the stock vLLM model server and score the selected delivery."""

import asyncio
import fcntl
import json
import os
from collections import Counter
from pathlib import Path

from aiohttp import ClientError

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient, get_response_json
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter, reconstruct_upstream_chat_completion
from nemo_gym.rollouts import RolloutCollectionHelper

from .delivery import canonical_bytes, canonical_hash, judge_projection, visible_delivery
from .judge import DIRECT_CODEX_PROFILE, assess, check_judge_configuration
from .prepare import (
    DEFAULT_DATA_DIR,
    ROOT,
    digest,
    fetch_release,
    prepared_rows,
    private_directory,
    records,
    write_private_bytes,
)


def _stored_capture(completion: dict, captures: Path) -> tuple[dict, bytes, str]:
    metrics = completion["metrics"]
    capture_sha = metrics["upstream_chat_capture_sha256"]
    raw_sha = metrics["upstream_chat_stream_sha256"]
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in (capture_sha, raw_sha)):
        raise ValueError("Invalid upstream capture handle")
    receipt = captures / f"{capture_sha}.json"
    path = captures / f"{raw_sha}.sse"
    if digest(receipt) != capture_sha or digest(path) != raw_sha:
        raise ValueError("Upstream capture bytes do not match the served Chat Completion")
    evidence = json.loads(receipt.read_bytes())
    if evidence["response_sha256"] != raw_sha:
        raise ValueError("Upstream request is bound to a different response")
    return evidence["request"], path.read_bytes(), capture_sha


async def _completion(client: ServerClient, request: dict, directory: Path) -> dict | None:
    received = directory / "completion.json"
    if received.exists():
        return json.loads(received.read_bytes())
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        return None
    write_private_bytes(directory / "request.json", canonical_bytes(request) + b"\n")
    response = await client.request(
        server_name="policy_model",
        url_path="/v1/chat/completions",
        method="POST",
        json=request,
        _resend_after_disconnect=False,
    )
    async with response:
        response.raise_for_status()
        completion = await get_response_json(response)
    write_private_bytes(received, canonical_bytes(completion) + b"\n")
    return completion


async def _evaluate(
    row: dict,
    criteria: list[dict],
    work: Path,
    captures: Path,
    dataset: Path,
    *,
    client: ServerClient,
    engine: NeMoGymAsyncOpenAI,
    model: str,
    context: int,
    manifest_hash: str,
    caps: list[int],
    judge_profile: str = DIRECT_CODEX_PROFILE,
) -> dict:
    result_path = work / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_bytes())
    work.mkdir(mode=0o700, exist_ok=True)
    output = {"task_id": row["id"], "reward": 0.0, "mask_sample": True, "status": "unscored_delivery"}
    base_request = row["baseline_request"] | {"model": model}
    fields = ("model", "messages", "chat_template_kwargs")
    try:
        profile_path = work / "profile.json"
        if profile_path.exists():
            count = json.loads(profile_path.read_bytes())["count"]
        else:
            tokens = await engine.create_tokenize(**{key: base_request[key] for key in fields if key in base_request})
            count = tokens["count"]
            if type(count) is not int or count < 0:
                raise ValueError("Missing verified prompt token count")
            write_private_bytes(profile_path, canonical_bytes({"count": count}) + b"\n")
        selected = None
        for cap in dict.fromkeys(min(value, context - count) for value in caps):
            if cap <= 0:
                break
            request = base_request | {"max_tokens": cap, "stream": False}
            completion = await _completion(client, request, work / f"cap-{cap}")
            if completion is None:
                raise ValueError("An upstream model request may still be interrupted")
            wire, raw, capture_hash = _stored_capture(completion, captures)
            if any(wire.get(key) != value for key, value in request.items() if key != "stream"):
                raise ValueError("Stock model server changed a frozen request field")
            version, _ = visible_delivery(reconstruct_upstream_chat_completion(raw), row["membership"])
            selected = (request, wire, raw, completion, capture_hash)
            if version in {"v2.2", "v2.4-native-action"}:
                break
        if selected is not None:
            request, wire, raw, completion, capture_hash = selected
            converter = ChatCompletionsConverter(return_token_id_information=True)
            parameters = converter.chat_completion_to_responses_create_params(
                NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(request)
            )
            parameters.max_output_tokens = request["max_tokens"]
            response = converter.chat_completion_to_response(
                parameters, NeMoGymChatCompletion.model_validate(completion), preserve_envelope_id=True
            )
            output["response"] = response.model_dump(mode="json")
            selection = {"item_id": row["id"], "capture_sha256": capture_hash, "profile": manifest_hash}
            selection_hash = canonical_hash(selection)
            write_private_bytes(work / "selection.json", canonical_bytes(selection) + b"\n")
            payload, delivery_receipt = judge_projection(
                row=row,
                criteria=criteria,
                wire_request=wire,
                raw=raw,
                manifest_sha256=manifest_hash,
                selection_sha256=selection_hash,
                capture_sha256=capture_hash,
                source_verifier_sha256=digest(ROOT / "delivery.py"),
            )
            if delivery_receipt is not None:
                write_private_bytes(work / "delivery.json", canonical_bytes(delivery_receipt) + b"\n")
            grade = await assess(payload, dataset, work / "judge", profile=judge_profile)
            output.update(
                reward=float(grade["grade"] == "pass"),
                mask_sample=grade["grade"] is None,
                status=grade["status"],
                grade=grade["grade"],
            )
    except (ClientError, TimeoutError, ValueError, KeyError) as error:
        output["failure_reason"] = type(error).__name__
    if output["mask_sample"]:
        output["failure_kind"] = "non_agentic_if:" + output["status"]
    write_private_bytes(result_path, canonical_bytes(output) + b"\n")
    return output


async def run_rollout_collection(rollout_collection_config, global_config_dict) -> None:
    config = rollout_collection_config
    if config.upload_rollouts:
        raise ValueError("Non-Agentic IF requires upload_rollouts=false to keep rollouts private")
    if config.num_repeats != 1 or config.fan_out or config.responses_create_params:
        raise ValueError("Frozen evaluation requires one repeat without fan-out or request overrides")
    if config.task_ids is not None or config.task_ids_fpath is not None:
        raise ValueError("Select a bounded Non-Agentic IF run with --limit")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("Non-Agentic IF requires a positive --limit")
    judge_profile = os.getenv("NON_AGENTIC_IF_JUDGE_PROFILE", DIRECT_CODEX_PROFILE)
    judge_settings = check_judge_configuration(judge_profile)
    policy = global_config_dict["policy_model"]["responses_api_models"]["vllm_model"]
    if not policy["upstream_chat_stream"] or not policy["return_token_id_information"]:
        raise ValueError("Configure verified upstream Chat delivery and token capture")
    if policy.get("refit_output_budget_to_window") or policy.get("refit_output_budget_to_prompt"):
        raise ValueError("The model server must preserve the selected output cap")
    captures = private_directory(Path(policy["upstream_chat_capture_dir"]))
    pin = json.loads((ROOT / "source_manifest.json").read_bytes())
    dataset = fetch_release(Path(os.environ.get("NON_AGENTIC_IF_DATA_DIR") or DEFAULT_DATA_DIR), pin)
    output_file = Path(config.output_jsonl_fpath)
    directory = private_directory(output_file.parent)
    output_file = directory / output_file.name
    with (directory / ".non-agentic-if.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output_file.exists() and not config.resume_from_cache:
            raise ValueError("Use a new private directory or resume the existing evaluation")
        expected = list(prepared_rows(dataset, pin))
        prepared = list(records(Path(config.input_jsonl_fpath)))
        if [{key: row[key] for key in expected[0]} for row in prepared] != expected:
            raise ValueError("Prepared rollout input differs from the frozen population")
        requested_ids = (
            {row["task_id"] for row in expected[: config.limit]}
            if config.limit
            else {row["task_id"] for row in expected}
        )
        requests = [row for row in records(dataset / "requests.jsonl") if row["id"] in requested_ids]
        criteria = {row["id"]: row["criteria"] for row in records(dataset / "criteria.jsonl")}
        engine = NeMoGymAsyncOpenAI(base_url=policy["base_url"], api_key=policy["api_key"], max_http_attempts=1)
        model = policy["model"]
        helper = RolloutCollectionHelper()
        client = helper.setup_server_client()
        served = [row for row in (await engine.create_models())["data"] if row["id"] == model]
        if len(served) != 1 or type(served[0].get("max_model_len")) is not int or served[0]["max_model_len"] <= 0:
            raise ValueError("The served model must advertise its context length")
        manifest = json.loads((dataset / "release_manifest.json").read_bytes())
        binding = {
            "release": pin["artifacts"]["release_manifest.json"],
            "model": {key: served[0].get(key) for key in ("id", "root", "owned_by", "max_model_len")},
            "requested_ids": sorted(requested_ids),
            "release_profile": manifest["profile"],
            "judge_profile": judge_settings,
            "runtime": {
                name: digest(ROOT / name) for name in ("rollout_driver.py", "delivery.py", "scoring.py", "judge.py")
            },
        }
        manifest_hash = canonical_hash(binding)
        identity = directory / "run-binding.json"
        if identity.exists() and json.loads(identity.read_bytes()) != binding:
            raise ValueError("Existing run has a different model, dataset or sample binding")
        if identity.exists() and not config.resume_from_cache:
            raise ValueError("Use a fresh run directory or resume the existing evaluation")
        write_private_bytes(identity, canonical_bytes(binding) + b"\n")
        slots = asyncio.Semaphore(config.num_samples_in_parallel or 1)

        async def evaluate(row: dict) -> dict:
            async with slots:
                return await _evaluate(
                    row,
                    criteria[row["id"]],
                    directory / canonical_hash(row["id"]),
                    captures,
                    dataset,
                    client=client,
                    engine=engine,
                    model=model,
                    context=served[0]["max_model_len"],
                    manifest_hash=manifest_hash,
                    caps=manifest["profile"]["context_bounded_output_caps"],
                    judge_profile=judge_profile,
                )

        results = await asyncio.gather(*(evaluate(row) for row in requests))
        write_private_bytes(output_file, b"".join(canonical_bytes(row) + b"\n" for row in results))
        counts = Counter(row.get("grade") for row in results)
        decided = counts["pass"] + counts["fail"]
        aggregate = {
            "membership/expected": pin["cases"],
            "membership/selected": len(results),
            "count/pass": counts["pass"],
            "count/fail": counts["fail"],
            "count/unscored": counts[None],
            "count/unattempted": pin["cases"] - len(results),
            "composite/complete_accuracy": counts["pass"] / pin["cases"] if decided == pin["cases"] else None,
            "composite/accuracy_among_scored": counts["pass"] / decided if decided else None,
        }
        document = {
            "agent_ref": {"name": "non_agentic_if_agent"},
            "_ng_environment_server": "non_agentic_if_environment",
            "agent_metrics": aggregate,
            "key_metrics": {"composite/complete_accuracy": aggregate["composite/complete_accuracy"]},
        }
        write_private_bytes(
            output_file.with_name(f"{output_file.stem}_aggregate_metrics.json"), canonical_bytes([document]) + b"\n"
        )
