# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive the frozen judge projection from verified policy delivery bytes."""

import hashlib
import json

from nemo_gym.models import reconstruct_upstream_chat_completion


PROTOCOL = "non-agentic-if-strict-response-grading-"
DELIVERY = "nonagentic-if-delivery-"
EMPTY_RULE = "sse-choice0-visible-delta-concatenation-v1"
POLICIES = {
    "v2.2": ("rubric.md", "judge.schema.json", None, None),
    "v2.3": (
        "non_agentic_if_replay_judge_v2_3.md",
        "judge.schema.json",
        "v2.3-bounded-visible",
        "verified-bounded-visible-delivery-v1",
    ),
    "v2.4-native-action": (
        "non_agentic_if_replay_judge_v2_4.md",
        "native.schema.json",
        "v2.4-native-action",
        "verified-native-action-delivery-v1",
    ),
    "v2.5": (
        "non_agentic_if_replay_judge_v2_5_empty_bounded.md",
        "judge.schema.json",
        "v2.5-empty-bounded",
        "verified-empty-bounded-delivery-v2.5",
    ),
    "v2.6": (
        "non_agentic_if_replay_judge_v2_6_empty_stop.md",
        "judge.schema.json",
        "v2.6-empty-stop",
        "verified-empty-stop-delivery-v2.6",
    ),
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def visible_delivery(completion: dict, membership: dict) -> tuple[str, dict]:
    """Select the delivery policy without exposing reasoning or executing native actions."""
    choice = completion["choices"][0]
    finish, message = choice["finish_reason"], choice["message"]
    content, refusal = message.get("content"), message.get("refusal")
    if any(value is not None and not isinstance(value, str) for value in (content, refusal)):
        raise ValueError("Delivery contains unsupported visible fields")
    calls = message.get("tool_calls")
    if calls:
        if (
            finish not in {"stop", "tool_calls"}
            or membership["original_split"] != "train"
            or membership["reserved_sep4"] is not False
            or membership["benchmark_role"] != "development_exposed"
        ):
            raise ValueError("Tool call is outside the frozen delivery eligibility")
        if any(
            call["type"] != "function"
            or not isinstance(call["id"], str)
            or not call["id"]
            or any(not isinstance(call["function"][key], str) for key in ("name", "arguments"))
            for call in calls
        ) or len({call["id"] for call in calls}) != len(calls):
            raise ValueError("Invalid inert function call")
        return "v2.4-native-action", {"content": content, "refusal": refusal, "tool_calls": calls}
    if finish not in {"stop", "length"}:
        raise ValueError("Unsupported finish reason")
    visible = {"content": content or "", "refusal": refusal or ""}
    if any(value.strip() for value in visible.values()):
        return ("v2.2" if finish == "stop" else "v2.3"), visible
    return ("v2.6" if finish == "stop" else "v2.5"), visible


def judge_projection(
    *,
    row: dict,
    criteria: list[dict],
    wire_request: dict,
    raw: bytes,
    manifest_sha256: str,
    selection_sha256: str,
    capture_sha256: str,
    source_verifier_sha256: str,
) -> tuple[dict, dict | None]:
    """Create the independent scoring payload from an exact selected request and upstream stream."""
    request = {"messages": wire_request["messages"]}
    baseline_input = {"messages": row["baseline_request"]["messages"]}
    input_hash = canonical_hash(request)
    if request != baseline_input or input_hash != row["baseline_provenance"]["input_sha256"]:
        raise ValueError("Policy input differs from the frozen messages")
    response = reconstruct_upstream_chat_completion(raw)
    if response["model"] != wire_request["model"]:
        raise ValueError("Unverified upstream policy identity")
    version, visible = visible_delivery(response, row["membership"])
    finish = response["choices"][0]["finish_reason"]
    item_id = (
        row["id"] if version != "v2.4-native-action" else canonical_hash({"id": row["id"], "run": manifest_sha256})
    )
    response_hash = canonical_hash(visible)
    binding = {
        "id": item_id,
        "input_sha256": input_hash,
        "response_sha256": response_hash,
        "benchmark_manifest_sha256": manifest_sha256,
    }
    if not criteria or any(criterion["mechanical_checks"] for criterion in criteria):
        raise ValueError("Expected frozen criteria without mechanical checks")
    payload = {
        "item_id": item_id,
        "policy_version": PROTOCOL + version,
        "payload_verified": True,
        "response_status": "complete",
        "finish_reason": finish,
        "input_request": request,
        "new_response": visible,
        "criteria": criteria,
        "mechanical_diagnostics": {
            criterion["criterion_id"]: [] for criterion in criteria if version != "v2.4-native-action"
        },
        "input_sha256": input_hash,
        "response_sha256": response_hash,
        "benchmark_manifest_sha256": manifest_sha256,
        "inference_binding": binding,
    }
    if version == "v2.2":
        return payload, None
    _, _, contract, receipt_schema = POLICIES[version]
    receipt = {
        "schema_id": receipt_schema,
        "delivery_contract_id": DELIVERY + contract,
        "selection_manifest_sha256": selection_sha256,
        "source_inference_sha256": capture_sha256,
        "source_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_request_sha256": canonical_hash(wire_request),
        "input_sha256": input_hash,
        "visible_response_sha256": response_hash,
        "actual_finish_reason": finish,
        "http_status": 200,
        "transport_kind": "sse",
        "terminal_marker_present": True,
        "capture_complete": True,
        "assistant_choice_index": 0,
        "model_identity_verified": True,
        "full_input_binding_verified": True,
        "visible_extraction_verified": True,
        "source_verifier_version": "non-agentic-if-gym-delivery-v1",
        "source_verifier_sha256": source_verifier_sha256,
    }
    if version != "v2.4-native-action":
        receipt.update(budget_field="max_tokens", request_output_budget=wire_request["max_tokens"])
    if version == "v2.6":
        empty_kind = "zero_length" if visible == {"content": "", "refusal": ""} else "whitespace_only"
        receipt.update(visible_extraction_rule=EMPTY_RULE, visible_empty_kind=empty_kind)
    receipt_hash = hashlib.sha256(canonical_bytes(receipt) + b"\n").hexdigest()
    payload.update(
        response_status={"v2.4-native-action": "complete", "v2.6": "verified_empty_stop"}.get(
            version, "bounded_visible"
        ),
        delivery_contract_id=receipt["delivery_contract_id"],
        delivery_receipt_sha256=receipt_hash,
        selection_manifest_sha256=selection_sha256,
        generation_request_sha256=receipt["source_request_sha256"],
    )
    if version == "v2.6":
        payload["delivery_fact"] = {
            "kind": "verified_complete_empty_stop",
            "empty_kind": empty_kind,
            "visible_response_sha256": response_hash,
            "proof_receipt_sha256": receipt_hash,
            "visible_extraction_rule": EMPTY_RULE,
        }
    return payload, receipt
