# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect one assistant turn with the stock vLLM model server and score the selected delivery."""

import asyncio
import fcntl
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientConnectorError, ClientError, ConnectionTimeoutError

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient, get_response_json
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter, reconstruct_upstream_chat_completion
from nemo_gym.rollouts import RolloutCollectionHelper
from nemo_gym.utils import materialized_path_for

from .delivery import canonical_bytes, canonical_hash, judge_projection, visible_delivery
from .judge import DIRECT_CODEX_PROFILE, assess, check_judge_access, check_judge_configuration
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


FROZEN_CHAT_FIELDS = {
    "chat_template_kwargs",
    "max_tokens",
    "messages",
    "model",
    "reasoning_effort",
    "seed",
    "stream",
    "temperature",
    "top_p",
}


def _policy_request(row: dict, model: str) -> dict:
    frozen = row["baseline_request"]
    if frozen.keys() - FROZEN_CHAT_FIELDS:
        raise ValueError("Frozen policy requests cannot include provider-managed context or tool grants")
    if canonical_hash({"messages": frozen["messages"]}) != row["baseline_provenance"]["input_sha256"]:
        raise ValueError("Policy messages differ from the frozen input")
    return {**frozen, "model": model}


def _saved_result(path: Path, task_id: str, run_hash: str, captures: Path) -> dict:
    record = json.loads(path.read_bytes())
    saved = record.get("result") if isinstance(record, dict) else None
    result = saved.get("output") if isinstance(saved, dict) else None
    if (
        not isinstance(result, dict)
        or saved.get("run") != run_hash
        or result.get("task_id") != task_id
        or record.get("sha256") != canonical_hash(saved)
    ):
        raise ValueError("Saved rollout differs from its original result or run binding")
    for root, receipts in ((path.parent, saved["work_receipts"]), (captures, saved["captures"])):
        for name, expected in receipts.items():
            receipt = (root / name).resolve()
            if not receipt.is_relative_to(root.resolve()) or not receipt.is_file() or digest(receipt) != expected:
                raise ValueError("Saved rollout evidence differs from its original receipts")
    return result


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


async def _completion(client: ServerClient, request: dict, directory: Path) -> dict:
    received = directory / "completion.json"
    recorded = directory / "request.json"
    not_sent = directory / "not-dispatched"
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        if not recorded.is_file() or json.loads(recorded.read_bytes()) != request:
            raise RuntimeError("Reconcile the saved policy request before resuming")
        if received.is_file():
            return json.loads(received.read_bytes())
        if not not_sent.exists():
            raise RuntimeError("Reconcile the ambiguous policy dispatch before resuming")
        not_sent.unlink()
    else:
        write_private_bytes(recorded, canonical_bytes(request) + b"\n")
    try:
        response = await client.request(
            server_name="policy_model",
            url_path="/v1/chat/completions",
            method="POST",
            json=request,
            allow_redirects=False,
            _max_num_tries=1,
            _resend_after_disconnect=False,
        )
        async with response:
            response.raise_for_status()
            completion = await get_response_json(response)
    except (ClientConnectorError, ConnectionTimeoutError) as error:
        not_sent.touch()
        raise RuntimeError(
            "Policy server connection failed before dispatch. Resume the same run after recovery."
        ) from error
    except (ClientError, TimeoutError, ValueError) as error:
        raise RuntimeError("Reconcile the ambiguous policy dispatch before resuming") from error
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
        return _saved_result(result_path, row["id"], manifest_hash, captures)
    work.mkdir(mode=0o700, exist_ok=True)
    output = {"task_id": row["id"], "reward": 0.0, "mask_sample": True, "status": "unscored_delivery"}
    work_receipts = []
    capture_receipts = {}
    base_request = _policy_request(row, model)
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
            work_receipts.extend(["selection.json", f"cap-{request['max_tokens']}/completion.json"])
            stream_hash = completion["metrics"]["upstream_chat_stream_sha256"]
            capture_receipts = {f"{capture_hash}.json": capture_hash, f"{stream_hash}.sse": stream_hash}
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
                work_receipts.append("delivery.json")
            grade = await assess(payload, dataset, work / "judge", profile=judge_profile)
            if grade["status"] == "interrupted_assessment":
                raise RuntimeError("Reconcile the ambiguous judge dispatch before resuming")
            if (work / "judge/result.json").is_file():
                work_receipts.append("judge/result.json")
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
    saved = {
        "output": output,
        "run": manifest_hash,
        "work_receipts": {name: digest(work / name) for name in work_receipts},
        "captures": capture_receipts,
    }
    write_private_bytes(result_path, canonical_bytes({"result": saved, "sha256": canonical_hash(saved)}) + b"\n")
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
    if policy.get("max_num_tries") != 1 or policy.get("max_http_attempts") != 1 or policy.get("sampling_overrides"):
        raise ValueError("Non-Agentic IF requires one policy attempt and frozen sampling controls")
    if policy.get("extra_body"):
        raise ValueError("Non-Agentic IF requires the frozen policy request without Model Server extra_body overrides")
    endpoint = urlsplit(policy["base_url"])
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or any((endpoint.username, endpoint.password, endpoint.query, endpoint.fragment))
    ):
        raise ValueError("Configure a policy base URL without embedded credentials")
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
        selected = prepared[: config.limit]
        helper = RolloutCollectionHelper()
        rows = helper.preprocess_examples(selected, agent_map={"_default": "non_agentic_if_agent"})
        by_id = {row["task_id"]: {**row, "_ng_environment_server": "non_agentic_if_environment"} for row in rows}
        inventory = materialized_path_for(output_file)
        inventory_bytes = b"".join(canonical_bytes(row) + b"\n" for row in by_id.values())
        if inventory.exists():
            if not config.resume_from_cache or inventory.read_bytes() != inventory_bytes:
                raise ValueError("Materialized rollout inventory mismatch")
        elif (directory / "run-binding.json").exists() or output_file.exists():
            raise ValueError("Missing materialized rollout inventory. Reconcile this run before resuming.")
        else:
            write_private_bytes(inventory, inventory_bytes)
        requests = [row for row in records(dataset / "requests.jsonl") if row["id"] in by_id]
        criteria = {row["id"]: row["criteria"] for row in records(dataset / "criteria.jsonl")}
        engine = NeMoGymAsyncOpenAI(base_url=policy["base_url"], api_key=policy["api_key"], max_http_attempts=1)
        model = policy["model"]
        client = helper.setup_server_client()
        served = [row for row in (await engine.create_models())["data"] if row["id"] == model]
        if len(served) != 1 or type(served[0].get("max_model_len")) is not int or served[0]["max_model_len"] <= 0:
            raise ValueError("The served model must advertise its context length")
        manifest = json.loads((dataset / "release_manifest.json").read_bytes())
        binding = {
            "release": pin["artifacts"]["release_manifest.json"],
            "model": {key: served[0].get(key) for key in ("id", "root", "owned_by", "max_model_len")},
            "policy_base_url": policy["base_url"],
            "requested_ids": sorted(by_id),
            "release_profile": manifest["profile"],
            "judge_profile": judge_settings,
            "runtime": {
                name: digest(ROOT / name)
                for name in ("rollout_driver.py", "delivery.py", "scoring.py", "judge.py", "reporting.py")
            },
        }
        manifest_hash = canonical_hash(binding)
        identity = directory / "run-binding.json"
        if identity.exists() and json.loads(identity.read_bytes()) != binding:
            raise ValueError("Existing run has a different model, dataset or sample binding")
        if identity.exists() and not config.resume_from_cache:
            raise ValueError("Use a fresh run directory or resume the existing evaluation")
        write_private_bytes(identity, canonical_bytes(binding) + b"\n")
        completed = {}
        for row in requests:
            result_path = directory / canonical_hash(row["id"]) / "result.json"
            if result_path.is_file():
                completed[row["id"]] = _saved_result(result_path, row["id"], manifest_hash, captures)
        if len(completed) < len(requests):
            await check_judge_access(judge_profile)
        slots = asyncio.Semaphore(config.num_samples_in_parallel or 1)

        async def evaluate(row: dict) -> dict:
            if row["id"] in completed:
                return by_id[row["id"]] | completed[row["id"]]
            async with slots:
                result = await _evaluate(
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

                return by_id[row["id"]] | result

        results = await asyncio.gather(*(evaluate(row) for row in requests))
        write_private_bytes(output_file, b"".join(canonical_bytes(row) + b"\n" for row in results))
        if not config.disable_aggregation:
            await helper._call_aggregate_metrics(results, list(by_id.values()), output_file)
