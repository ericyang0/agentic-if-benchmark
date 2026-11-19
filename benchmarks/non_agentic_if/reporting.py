# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

from nemo_gym.config import AggregateMetrics
from nemo_gym.metrics import compute_aggregate_metrics

from .prepare import ROOT


def aggregate(rows: list[dict]) -> AggregateMetrics:
    """Report the selected outcomes against the full frozen cohort."""
    expected = json.loads((ROOT / "source_manifest.json").read_bytes())["cases"]
    task_ids = {row.get("task_id") for row in rows}
    if None in task_ids or len(task_ids) != len(rows) or len(rows) > expected:
        raise ValueError("Expected one result per selected benchmark task")
    for row in rows:
        grade = row.get("grade")
        if (
            grade not in {"pass", "fail", None}
            or row.get("mask_sample") is not (grade is None)
            or row.get("reward") != float(grade == "pass")
        ):
            raise ValueError("Inconsistent benchmark grade, mask or reward")
    counts = Counter(row.get("grade") for row in rows)
    passed, failed = counts["pass"], counts["fail"]
    decided = passed + failed
    metrics = {
        "membership/expected": expected,
        "membership/selected": len(rows),
        "count/pass": passed,
        "count/fail": failed,
        "count/unscored": counts[None],
        "count/unattempted": expected - len(rows),
        "composite/complete_accuracy": passed / expected if decided == expected else None,
        "composite/accuracy_among_scored": passed / decided if decided else None,
    }
    result = compute_aggregate_metrics(rows)
    result.agent_metrics.update(metrics)
    result.key_metrics = {"composite/complete_accuracy": metrics["composite/complete_accuracy"]}
    return result
