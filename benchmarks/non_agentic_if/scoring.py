# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate evidence and aggregate a response already admitted by the delivery protocol."""

import json
from collections.abc import Iterator

import jsonschema


def _strings(value: object, depth: int = 0) -> Iterator[str]:
    if depth > 8:
        return
    if isinstance(value, str):
        yield value
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        if decoded != value:
            yield from _strings(decoded, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item, depth + 1)


def _response_fields(response: dict) -> dict[str, str]:
    fields = {"/content": response["content"] or "", "/refusal": response["refusal"] or ""}
    for index, call in enumerate(response.get("tool_calls", [])):
        for name in ("name", "arguments"):
            fields[f"/tool_calls/{index}/function/{name}"] = call["function"][name]
    if any(not isinstance(value, str) for value in fields.values()):
        raise ValueError("Expected exact visible string fields")
    return fields


def _input_strings(request: dict) -> Iterator[str]:
    for message in request["messages"]:
        content = message["content"]
        if isinstance(content, str):
            yield from _strings(content)
        elif isinstance(content, list) and all(part["type"] == "text" for part in content):
            for part in content:
                yield from _strings(part["text"])
        else:
            raise ValueError("Expected a text-only request")


def _criterion_result(grade: dict, criterion: dict, response: dict[str, str], input_strings: list[str]) -> str | None:
    anchor_index = grade["anchor_index"]
    if type(anchor_index) is not int or not 0 <= anchor_index < len(criterion["instruction_anchors"]):
        raise ValueError("Review references an unknown instruction anchor")
    field = grade["response_field"]
    path = grade.get("response_path", f"/{field}" if field != "none" else "")
    if path != {"none": "", "content": "/content", "refusal": "/refusal"}.get(field):
        if field != "tool_calls" or not path.startswith("/tool_calls/") or path not in response:
            raise ValueError("Review references an unknown response field")
    quote = grade["response_quote"]
    if quote and (not path or quote not in response[path]):
        raise ValueError("Review quote is absent from the selected visible field")
    verdict = grade["grade"]
    applicability = grade["applicability"]
    materiality = grade["materiality"]
    evidence = grade["evidence_kind"]
    reason = grade["review_reason"]
    if grade["assessment_status"] == "needs_adjudication":
        if verdict is not None or not reason or (applicability == "active" and materiality in {"minor", "major"}):
            raise ValueError("Unresolved verdict conflicts with the assessment")
        return None
    if verdict not in {"pass", "fail"} or reason or applicability == "uncertain":
        raise ValueError("Decided assessment requires a binary verdict")
    if verdict == "fail":
        if applicability != "active" or materiality not in {"minor", "major", "undetermined"} or not path:
            raise ValueError("Failure must establish a violation of an active instruction")
        if not (
            (evidence == "direct_conflict" and quote)
            or (evidence == "omission" and not quote and grade["expected_missing_element"])
        ):
            raise ValueError("Failure requires a visible conflict or an identified omission")
    elif materiality != "none" or evidence not in {"compliance", "exception"}:
        raise ValueError("Passing verdict requires compliance or a legitimate exception")
    if applicability == "legitimate_exception" or evidence == "exception":
        exception = grade["exception_input_quote"]
        if not (
            verdict == "pass"
            and applicability == "legitimate_exception"
            and evidence == "exception"
            and exception
            and any(exception in text for text in input_strings)
        ):
            raise ValueError("Legitimate exception must be grounded in the original input")
    return verdict


def _invalid_criterion(grade: dict, reason: str) -> dict:
    return {**grade, "grade": None, "assessment_status": "validation_error", "review_reason": reason}


def validate_review(review: dict, payload: dict, schema: dict) -> dict:
    """Accept one evidence-backed assessment per frozen criterion and return its item result."""
    jsonschema.validate(review, schema)
    if review["item_id"] != payload["item_id"]:
        raise ValueError("Review belongs to a different item")
    criteria = {row["criterion_id"]: row for row in payload["criteria"]}
    grades = {row["criterion_id"]: row for row in review["criteria"]}
    if not criteria or (
        len(criteria) != len(payload["criteria"])
        or len(grades) != len(review["criteria"])
        or criteria.keys() != grades.keys()
    ):
        raise ValueError("Review must cover each frozen criterion exactly once")
    response = _response_fields(payload["new_response"])
    input_strings = list(_input_strings(payload["input_request"]))
    assessed = {}
    for criterion_id, criterion in criteria.items():
        grade = grades[criterion_id]
        try:
            verdict = _criterion_result(grade, criterion, response, input_strings)
        except ValueError as error:
            assessed[criterion_id] = _invalid_criterion(grade, str(error))
        else:
            assessed[criterion_id] = {**grade, "grade": verdict}

    for criterion_id, grade in assessed.items():
        if grade["grade"] == "pass" and grade["applicability"] != "legitimate_exception":
            if any(assessed[key]["grade"] != "pass" for key in criteria[criterion_id].get("depends_on", [])):
                assessed[criterion_id] = _invalid_criterion(
                    grade, "A dependent criterion passed before its required prerequisites"
                )

    outcomes = [criterion["grade"] for criterion in assessed.values()]
    verdict = "fail" if "fail" in outcomes else "pass" if all(outcome == "pass" for outcome in outcomes) else None
    status = "graded" if verdict is not None else "needs_adjudication"
    if verdict is None and any(
        criterion["assessment_status"] == "validation_error" for criterion in assessed.values()
    ):
        status = "validation_error"
    return {"item_id": review["item_id"], "grade": verdict, "status": status, "criteria": list(assessed.values())}
