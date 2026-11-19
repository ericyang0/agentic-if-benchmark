# Non-Agentic IF review audit

This audit covers the 19 inline threads and the task schema question in
[Agentic IF onboarding #5013](https://github.com/NVIDIA-NeMo/Gym/pull/5013),
plus the lifecycle review and delivery checks in
[the stock-client follow-up #6160](https://github.com/NVIDIA-NeMo/Gym/pull/6160).

## Onboarding review

| Reviewer finding | Non-Agentic IF evidence |
|---|---|
| [1. Internal bug reference](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365234608) | The benchmark configuration contains no internal bug references. |
| [2. Configuration instead of environment variables](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365236523) | Gym resolves policy settings through its standard CLI and Hydra configuration. EFB binds its deployed endpoint per recipe. Environment variables supply the selected judge profile, key, and private storage paths. |
| [3. Standard evaluation command](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365250221) | `gym eval run --benchmark non_agentic_if` invokes the configured rollout driver. Synthetic checks exercise the normal CLI. |
| [4. Development-only resource documentation](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365254719) | The benchmark has a user runbook and uses no benchmark-specific Resources Server. |
| [5. Shared packaging and lint overrides](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365256771) | The integration uses the existing Gym package and lint rules. Its reporting server declares only a Gym dependency. |
| [6. Private collation files](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365367363) | The whole prepared-data directory points outside Git. Collated rows contain a task identifier and empty input. Exact requests stay in the verified private cache. |
| [7. Provider conversation context](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365367370) | The rollout driver checks a closed set of frozen Chat fields and the exact message digest. Tests reject conversation IDs, previous response IDs, prompt and instruction overrides, and tool grants. The driver also rejects Model Server `extra_body` overrides. |
| [8. Materialized input inventory](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4365367371) | The driver writes one mode-0600 inventory row per selected task before dispatch, checks it on resume, and refuses a changed inventory. Synthetic run health verified one expected task and zero missing. |
| [9. Reaggregation loses benchmark metrics](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4384188732) | Initial collection and `gym eval aggregate` both call the benchmark Environment Server. Synthetic tests reconcile pass, fail, unscored and unattempted counts and leave complete accuracy null for partial coverage. |
| [10. Real-task reward evidence](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4384188737) | The runbook records the historical GLM-5.3 profile with 256 passes and 315 failures across 571 scored cases. A deployed run using the current judge profile remains pending. |
| [11. Frozen policy provenance](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4384188743) | The public manifest pins the private release and its policy artifacts by hash. The release records the historical generation and scoring manifest hashes. Runtime code implements the contract in Gym. |
| [12. Published discovery](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4384188750) | The Gym environment catalog links to the runbook and records the benchmark's private, development-exposed, evaluation-only status. |
| [13. Policy endpoint and model precedence](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407411778) | The benchmark has no competing endpoint alias. The driver checks the resolved endpoint and served model, and binds them to the run. EFB's configuration resolves to its deployed GLM endpoint. |
| [14. Modifying the shared legacy relay](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407411783) | Reporting uses a benchmark-local Environment Server. It rejects direct generation requests and uses the existing aggregate metrics contract. |
| [15. Transport failures advance delivery](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407411795) | A verified local connection failure resumes the same cap. An ambiguous dispatch stops for reconciliation. A valid terminal SSE event survives an HTTP body error. |
| [16. Judge access failures spend policy or grading attempts](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407411801) | Judge access is checked before generation. Hub access denials stop the run and preserve the saved policy answer. Non-2xx replies are outside the grading budget, with a separate bound on transport retries. Tests recover after four denied requests. |
| [17. Handwritten aggregate artifact](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4407411806) | Both commands use Gym aggregation with its standard group and reward statistics. The CLI synthetic run reproduced the full aggregate artifact. |
| [18. Cached equality depends on fresh observations](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4486927970) | Completed resume checks the saved task, run, result, and selected receipt digests. Tests reject an edited reward and corrupted upstream bytes. Completed resume performs no additional policy or judge dispatch. |
| [19. Capture identity reused across retries](https://github.com/NVIDIA-NeMo/Gym/pull/5013#discussion_r4486927979) | Upstream captures are content addressed by exact request and stream digest. The same cap can be sent again only after a verified local connection failure before dispatch. Later token caps have distinct request bindings. |
| [Task Data schema](https://github.com/NVIDIA-NeMo/Gym/pull/5013#issuecomment-6377342888) | `gym env validate` checks the manifest and metadata-only rows. The driver validates prepared rows against the frozen cohort before model dispatch, including IDs, input shape, and verifier metadata. |

## Stock-client follow-up

| Reviewer concern | Non-Agentic IF evidence |
|---|---|
| [Agent sessions, cleanup, tool grants](https://github.com/NVIDIA-NeMo/Gym/pull/6160#issuecomment-6880439707) | The driver sends exact Chat messages directly through the stock Model Server to preserve the frozen history. It creates no agent sessions or tools. The reporting server serves aggregation separately. |
| Incomplete delivery and first complete selection | The tests exercise a capped delivery followed by a complete one. The selected answer stays fixed whether its grade passes or fails. |
| Refitting and automatic retries | The driver checks single-attempt settings and rejects sampling and body overrides. Frozen caps and seed are verified against upstream receipts. |
| Token retention and judge projection | An end-to-end synthetic CLI run retained reasoning, token IDs, and log probabilities. The judge received the separate visible projection. |
| Provider repetition | Tests cover `native_finish_reason=repetition` and `stop_reason=repetition_detected`, including a marker on an earlier stream chunk. |
| External-contributor CI gate | Hosted Gym CI requires maintainer authorization of the submitted head. The EFB configuration has its own CI and deployed smoke. |
