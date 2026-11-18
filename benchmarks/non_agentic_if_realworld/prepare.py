# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the frozen Non-Agentic IF dataset from the approved private registry."""

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from nemo_gym.config import DownloadJsonlDatasetGitlabConfig, get_global_config_dict
from nemo_gym.data import download_jsonl_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path.home() / ".cache/nemo-gym/non_agentic_if_realworld/1.0.0"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def private_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if any((parent / ".git").is_file() or (parent / ".git/HEAD").is_file() for parent in (path, *path.parents)):
        raise ValueError("Keep private benchmark artifacts outside Git")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("The benchmark cache must be owner-only")
    return path


def fetch_release(directory: Path, pin: dict) -> Path:
    directory = private_directory(directory)
    fd = os.open(directory / ".download.lock", os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, expected in pin["artifacts"].items():
            if Path(name).name != name:
                raise ValueError("Unexpected artifact name")
            target = directory / name
            if target.is_file() and not target.is_symlink() and digest(target) == expected:
                continue
            uri = get_global_config_dict().get("mlflow_tracking_uri")
            if (
                not isinstance(uri, str)
                or hashlib.sha256(uri.rstrip("/").encode()).hexdigest() != pin["tracking_uri_sha256"]
            ):
                raise ValueError("Configure the approved private registry URI before downloading")
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as stream:
                staging = Path(stream.name)
            try:
                download_jsonl_dataset(
                    DownloadJsonlDatasetGitlabConfig(
                        dataset_name=pin["dataset_name"],
                        version=pin["version"],
                        artifact_fpath=name,
                        output_fpath=str(staging),
                    )
                )
                if digest(staging) != expected:
                    raise ValueError(f"Benchmark hash mismatch: {name}")
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
    return directory


def records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prepared_rows(directory: Path, pin: dict) -> Iterator[dict]:
    criteria = list(records(directory / "criteria.jsonl"))
    membership = list(records(directory / "membership.jsonl"))
    requests = list(records(directory / "requests.jsonl"))
    members = {row["id"]: row for row in membership}
    ids = {row["id"] for row in requests}
    if not len(members) == len(membership) == len(ids) == len(requests) == len(criteria) == pin["cases"]:
        raise ValueError("Unexpected benchmark population")
    if ids != members.keys() or ids != {row["id"] for row in criteria}:
        raise ValueError("Benchmark requests, criteria and membership disagree")
    if sum(len(row["criteria"]) for row in criteria) != pin["criteria"]:
        raise ValueError("Unexpected number of benchmark criteria")
    for row in requests:
        item_id = row["id"]
        if row["membership"] != members[item_id]:
            raise ValueError("Request membership mismatch")
        # The evaluation driver must send the original Chat request for exact history.
        messages = row["baseline_request"]["messages"]
        inputs = [
            {
                **message,
                "content": [{"type": "input_text", "text": part["text"]} for part in message["content"]],
            }
            if isinstance(message["content"], list)
            else message
            for message in messages
        ]
        yield {
            "task_id": item_id,
            "responses_create_params": {"input": inputs},
            "verifier_metadata": {"task_id": item_id},
        }


def prepare() -> Path:
    pin = json.loads((ROOT / "source_manifest.json").read_text())
    directory = fetch_release(Path(os.environ.get("NON_AGENTIC_IF_DATA_DIR") or DEFAULT_DATA_DIR), pin)
    data = ROOT / "data"
    if (data.exists() or data.is_symlink()) and (not data.is_symlink() or data.resolve() != directory):
        raise ValueError("Benchmark data directory already points elsewhere")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            for row in prepared_rows(directory, pin):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(directory / "prepared.jsonl")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if not data.is_symlink():
        data.symlink_to(directory, target_is_directory=True)
    return data / "prepared.jsonl"


if __name__ == "__main__":
    prepare()
