# Non-Agentic Instruction Following

The private release contains 571 historical regression cases and 686 atomic
criteria. Every case is evaluation-only and development-exposed. One case was
excluded before launch because its criterion could not be resolved reliably.
Historical grades retain their original judge model and protocol.

## Prepare and run

Obtain read access to the approved private registry through your organization's
access process. Install Gym with the `mlflow` extra and set the approved
`mlflow_tracking_uri` and `mlflow_tracking_token` in your private configuration.
Preparation verifies the release and every artifact before producing dataset
rows. The prepared rows contain case identifiers with empty inputs. The
rollout driver reads the exact messages from the verified private cache.
Use the standard server-starting `gym eval run` path to invoke that driver.
Supply owner-only cache and capture directories outside Git using
`NON_AGENTIC_IF_DATA_DIR` and `NON_AGENTIC_IF_CAPTURE_DIR`.

```bash
gym eval prepare --benchmark non_agentic_if
gym eval run --benchmark non_agentic_if \
  --limit 5 --output /path/to/owner-only-run/rollouts.jsonl
```

Use the standard vLLM model server with an endpoint that supplies its context
length, tokenizer count, token IDs and log probabilities. The rollout driver
sends the original Chat Completions history to preserve named participants and
adjacent assistant turns. It uses seed 17, temperature 0.7 and context-bounded
caps of 32,768, 65,536 and 131,072 tokens. The verified upstream stream, full
Responses output, and visible judge projection are stored separately. Tool
calls stay inert. This benchmark fixes the request controls, so run without
additional overrides or repeats.

The rollout driver calls the stock Model Server's Chat route directly to
preserve the frozen messages and request controls. There is no agent session
or tool grant. The Environment Server reaggregates saved outcomes. Gym
records one materialized input row per selected task before policy dispatch.

For new EFB scores, select `NON_AGENTIC_IF_JUDGE_PROFILE=inference_hub_v1` and
supply `NON_AGENTIC_IF_JUDGE_API_KEY` through your runtime secret configuration.
The profile uses `openai/openai/gpt-6.2-sol` with medium reasoning and strict JSON
over NVIDIA Inference Hub. The default standalone profile, `direct_codex_v1`,
requires an authenticated Codex CLI. Start a fresh run directory when switching
profiles. Each run records the active judge profile alongside the release.

The judge receives the frozen criteria and applicable delivery-specific rubric.
Every criterion must pass for a scored pass. A valid
violation fails the item. Unresolved assessments and technical problems are
reported as unscored. The complete accuracy metric requires valid grades for
all 571 cases. Scores from bounded smoke runs carry the selected and scored
denominators.

The archived GLM-5.3 run on this cohort has 256 passes, 315 failures and zero
unscored cases out of 571, for a mean binary reward of 0.4483. It used seed 17
and temperature 0.7. These grades describe the historical evaluation. Measure
the current Gym and Inference Hub profile with a separate run.

Keep rollout artifacts and judge receipts private. A new provider, judge model,
or scoring protocol needs a separate run. Resume can reuse a verified response
or retry a connection that failed before dispatch. Reconcile ambiguous policy
dispatches in the private run directory before resuming.
