# AlphaGo audit — 6 November 2026

This audit covers the AlphaGo BRAIN research workstation at commit `6c54511`. It checks how robust the app is against BRAIN and the AI providers, how well it generates alphas that pass submission, and what the SPC and AQP competitions need. Every finding below was fixed in this round unless the **Status** says otherwise. The last section lists what can only be settled on a live BRAIN account.

**Method.**
- Read the BRAIN client, sync, research, generator, evaluator, repair, LLM and UI layers.
- Ran the test suite against a mocked BRAIN.
- Checked current external facts online: OpenRouter's free-model line-up (November 2026) and BRAIN API rate-limit behaviour. No public documentation exists for BRAIN's rate limits, the SPC submission API or the AQP rules.
- Verified the OpenRouter ranking against OpenRouter's live catalogue.

**Result.**
- **Tests:** 272 pass, against a mocked BRAIN.
- **Lint:** clean under a new project ruff config.
- **Not yet done:** none of this has been run against a real BRAIN account (see the last section).

---

## 1. BRAIN API robustness

| # | Finding | Severity | Status |
|---|---|---|---|
| 1.1 | **Field sync failed with `GET /data-fields → 429: API rate limit exceeded`.** A full catalogue is 173 datasets and about 278,000 fields at 50 per request, so thousands of requests. They ran 4 at a time with no pacing. Every 429 was retried on its own clock, so the four workers kept the limit tripped. After 6 retries one request failed, and `gather_all` cancelled the whole sync. | High | **Fixed.** New `brain/ratelimit.py`: catalogue calls start at 2 per second, halve on every 429 and recover after 25 successes in a row. A 429 opens a cooldown for its whole class of requests, using `Retry-After` or an exponential wait. It allows up to 12 rate-limit retries per request, separate from the transient-error budget. Default concurrency went from 4 to 2. |
| 1.2 | One failing dataset aborted the field sync for all datasets. | High | **Fixed.** Failures are collected and retried once at the end, one request at a time. The job fails only if every dataset failed. |
| 1.3 | An interrupted sync started from zero. | Medium | **Fixed.** Datasets synced in the last 72 hours are skipped unless `full` is set (`brain.fields_resync_hours`). |
| 1.4 | No way to see the throttling state. | Low | **Fixed.** A live **BRAIN request pacing** card on Health shows the rate, cooldown and 429 count. |
| 1.5 | Simulation-start 429s ("concurrent limit") had to stay out of the new pacing. | — | **Checked.** `POST /simulations` does not retry 429s, so the gate never slows simulation slots. |

## 2. LLM providers

| # | Finding | Severity | Status |
|---|---|---|---|
| 2.1 | OpenRouter could only be reached through the OpenAI-compatible provider, which needs a hand-picked model. Older guides recommend IDs that now return 404 (`gpt-oss-120b:free`, `qwen3-coder:free`, `llama-3.3-70b-instruct:free`), and the free line-up changes every few weeks. | High | **Fixed.** New `llm/openrouter.py` provider. It reads OpenRouter's live `/models` list and keeps free, text-only models. It ranks them by size, context window, structured-output support, reasoning support and retirement date, and adjusts each model's rank by its error share in your own call log over the last 2 days. Auto-pick per role: descriptions go to the runner-up, which spreads per-model limits. |
| 2.2 | A retired or busy free model failed the call. | High | **Fixed.** Each request carries two backup models in OpenRouter's `models` fallback list. On a 404, 429, 502 or 503 the model rests (90 s, or a day for a 404) and the next ranked model is tried at once. |
| 2.3 | OpenRouter's account-wide daily free cap was retried like a transient error. | Medium | **Fixed.** It is reported as a cap, so the router fails over to your fallback provider. |
| 2.4 | Ranking checked against the live catalogue on 6 November 2026: 457 models listed, 13 free text models ranked, led by Nemotron 3 Ultra and Gemma 4. Music and other non-text models are excluded. | — | Verified. |

## 3. Alpha generation: why few alphas reached "submittable"

The repair loop (`engine/refine.py`) was already strong: targeted fixes, sign flips and correlation routes. The weakness was **upstream**. Search tasks enumerated every template over every field (up to 60 fields per dataset). On a typical dataset most fields carry no usable signal, so most of the budget went to noise before anything reached the repair loop.

| # | Finding | Severity | Status |
|---|---|---|---|
| 3.1 | Budget spent evenly over every field, signal or not. | High | **Fixed.** **Field screening** in research tasks: one probe per field (its time-series rank over the category's main horizon). Only fields with a Sharpe of at least 0.5 in either direction are kept: at most 35% of the dataset's fields, at least 4. Negative fields are used sign-flipped. The probe is the first expression the search would have run anyway, so screening costs no extra simulations. |
| 3.2 | The same windows (66, 126, 22, 252) for all data. Fundamentals need long horizons to keep turnover, and so fitness, in range; price/volume needs short ones. | Medium | **Fixed.** Windows are chosen by data category: 252/126/66/22 for slow data, 22/66/5/126 for fast data. |
| 3.3 | Template families were picked uniformly, with no learning. | Medium | **Fixed.** `engine/families.py` keeps your hit rate per family (hit = promising or better). Research tasks and template ideas order families by Thompson sampling. A table on Health shows the record. |
| 3.4 | Few templates in the shapes that most often clear Fitness: slow, rank-bounded and peer-relative. | Medium | **Fixed.** Added `zscore_delta`, `mean_diff`, `smoothed_rank`, `group_neut_rank` and `group_rank_long`. Search Lab also runs group-relative templates on its best fields. |
| 3.5 | The AI ideation prompt asked for "valid, diverse" ideas but never stated BRAIN's pass bar. | Medium | **Fixed.** The prompt now states Sharpe ≥ 1.25 and Fitness ≥ 1.0 with the fitness formula, the turnover band, what suits slow versus fast data, weight bounding for CONCENTRATED_WEIGHT and the sub-universe check, and the self-correlation limit. |
| 3.6 | Autopilot and template ideas ignored what research tasks had learned about a dataset. | Medium | **Fixed.** `screened_fields()` feeds the screened, signed fields to template ideas (when there are at least 3) and lists them in the AI's dataset context. |

## 4. SPC (Systematic Predictions Challenge)

| # | Finding | Severity | Status |
|---|---|---|---|
| 4.1 | SPC Lab could only check pasted output and back-test it. No generation, no submission. | High | **Fixed.** **Generate prompt**: the AI writes a rules-based prompt with exact counts, the exact JSON output spec and the character limit. **Dry-run** runs it on your provider 2–5 times. **Run all checks** applies the gates: prompt written, character limit, output spec, no placeholders, every run valid, stability across runs (worst overlap ≥ 30%), and optionally Test PnL with ≥ 90% of tickers priced. **Submit to BRAIN** stays locked until every blocking gate passes, then asks for confirmation. The prompt version is saved, marked submitted and stored with BRAIN's reply. |
| 4.2 | BRAIN publishes no SPC submission API. | — | **Handled without guessing.** You paste one request captured from your browser (**Copy as cURL** or **Copy as fetch**), and the app derives the path, method, competition id and a body template with typed placeholders. **Find my competitions** lists ids from `/users/self/competitions`. **Open:** needs one manual capture on your account. |
| 4.3 | Dry runs use your AI provider, not BRAIN's model, so they test clarity and stability, not BRAIN's picks. | — | Documented in the UI and the guide. |

## 5. AQP (Alpha Quality Project)

| # | Finding | Severity | Status |
|---|---|---|---|
| 5.1 | No support for the monthly AQP cycle: up to 3 USA alphas tagged `AQP` + YYYYMM, never reused across cycles, scored 80% on the description (match, uniqueness, hypothesis) and 20% on OS performance. | High | **Fixed.** New `engine/aqp.py` and the **Alpha Quality Project** page. It lists eligible alphas and ranks them by distinctiveness (self-correlation, field popularity, richness) plus OS evidence (OS Sharpe, else IS Sharpe × your predicted retention). **Suggest best 3** keeps the picks to different ideas. |
| 5.2 | The 80% description part needs text that matches the code exactly. | High | **Fixed.** A structured AI description (idea, mechanism, data, construction, novelty, behaviour, risks) is graded twice: a strict match check (every field, non-arithmetic operator, window, grouping and the neutralization must be described) and an AI judge for match, uniqueness and hypothesis quality. It rewrites until the grade stops improving and shows an estimated score out of 100. |
| 5.3 | Applying must respect the cycle rules. | High | **Fixed.** `apply()` refuses a 4th alpha in the cycle, an alpha from an earlier cycle, non-USA or unsubmitted alphas, and descriptions under 200 characters. It patches tags and description in one call, after confirmation. |

## 6. Code health

| # | Finding | Severity | Status |
|---|---|---|---|
| 6.1 | No lint config: ruff 0.17's defaults reported 377 issues. | Low | **Fixed.** `[tool.ruff]` in `pyproject.toml` with line length 125, matching the code, and the E/F/W/I/UP/B rule families. Every reported issue was fixed or explicitly ignored; B905 (`zip(strict=)`) is off, because adding it would change behaviour where lengths differ on purpose. |
| 6.2 | pytest-asyncio warned about an unset fixture loop scope. | Low | **Fixed.** |
| 6.3 | The project's `.venv` pointed at a Python on another machine and could not run. | Medium | **Fixed.** Rebuilt from your Python 3.11. |
| 6.4 | Tests for the new behaviour. | — | **Added.** Rate limiting (429 recovery, failure isolation, resume, AIMD), OpenRouter (catalogue ranking, fallback routing, rotation, daily cap), screening, family learning, screened-field reuse, SPC gates, generation, dry run, capture import, gated submit, and AQP eligibility, diverse pick, match check, rewrite loop and apply rules. Smoke tests cover the SPC and AQP pages. |

## 7. What only a live account can settle

These cannot be confirmed offline. Each is set up so you can check it in a few minutes.

1. **Whether screening raises your pass rate.** Run one Search Lab task on a dataset you know. The task message reports how many fields screening kept; compare its promising/strong counts with an earlier task on the same dataset (Research History).
2. **BRAIN's actual rate limit.** Watch **Health → BRAIN request pacing** during a full sync. If you see many 429s, set `brain.catalog_rps` to 1.
3. **The SPC endpoint.** Capture one submission (SPC Lab → BRAIN submission endpoint and gates → Import). After that, submissions go from the app.
4. **AQP.** BRAIN's AQP FAQ is not public. If it adds rules beyond tag, count, USA-only and no-reuse, tell me and I'll add them to `apply()`. The description structure follows the scoring criteria BRAIN announced.
5. **OpenRouter.** Add a key and click **Save & test**. The ranking shown on the LLM page is today's live list.
