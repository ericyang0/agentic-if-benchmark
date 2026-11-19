# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, HTTPException

from benchmarks.non_agentic_if.reporting import aggregate
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.server_runtime import SimpleServer
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


class NonAgenticIFEnvironmentServer(SimpleServer):
    """Report saved results. The rollout driver owns all generation and grading."""

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        app.post("/run")(self.run)
        return app

    async def run(self) -> None:
        raise HTTPException(409, "Run Non-Agentic IF with gym eval run --benchmark non_agentic_if.")

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        try:
            return aggregate(body.verify_responses)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error


if __name__ == "__main__":
    NonAgenticIFEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NonAgenticIFEnvironmentServer.run_webserver()
