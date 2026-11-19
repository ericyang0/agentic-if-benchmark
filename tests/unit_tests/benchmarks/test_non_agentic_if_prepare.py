# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.non_agentic_if import prepare


def test_preparation_keeps_private_messages_out_of_collated_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership = {"id": "synthetic-case", "original_split": "train"}
    messages = [
        {"role": "system", "content": "Discuss fictional maps."},
        {"role": "assistant", "content": "We can start with the north shore."},
        {"role": "assistant", "content": "Or use the old ferry station."},
        {"role": "user", "name": "cartographer", "content": [{"type": "text", "text": "Start at the ferry."}]},
    ]
    for filename, row in {
        "membership.jsonl": membership,
        "requests.jsonl": {
            "id": membership["id"],
            "membership": membership,
            "baseline_request": {"messages": messages},
        },
        "criteria.jsonl": {"id": membership["id"], "criteria": [{"criterion_id": "synthetic-criterion"}]},
    }.items():
        (tmp_path / filename).write_text(json.dumps(row) + "\n")

    pin = {"cases": 1, "criteria": 1}
    (row,) = prepare.prepared_rows(tmp_path, pin)
    assert row["responses_create_params"] == {"input": []}
    assert row["verifier_metadata"] == {"task_id": membership["id"]}
    (tmp_path / "source_manifest.json").write_text(json.dumps(pin))
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    cache = tmp_path / "prepared.jsonl"
    cache.write_text(json.dumps(row | {"responses_create_params": {"input": messages}}) + "\n")
    assert not prepare.is_prepared_data_current(cache)
    cache.write_text(json.dumps(row) + "\n")
    assert prepare.is_prepared_data_current(cache)

    (tmp_path / "membership.jsonl").write_text(json.dumps({**membership, "original_split": "test"}) + "\n")
    with pytest.raises(ValueError, match="Request membership mismatch"):
        list(prepare.prepared_rows(tmp_path, {"cases": 1, "criteria": 1}))


def test_unapproved_registry_cannot_replace_a_cache_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "cache"
    directory.mkdir(mode=0o700)
    target = directory / "requests.jsonl"
    target.write_text("unverified")
    monkeypatch.setattr(prepare, "get_global_config_dict", lambda: {"mlflow_tracking_uri": "https://example.test"})
    with pytest.raises(ValueError, match="approved private registry"):
        prepare.fetch_release(directory, {"artifacts": {"requests.jsonl": "0" * 64}, "tracking_uri_sha256": "a" * 64})
    assert target.read_text() == "unverified"
