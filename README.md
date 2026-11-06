# AlphaGo: a WorldQuant BRAIN alpha research desktop app

AlphaGo is a Windows desktop app for BRAIN consultants. It finds data, generates FASTEXPR alphas with an LLM of your choice (Gemini, Claude, OpenAI or a local Ollama model) and templates, and simulates them through the official BRAIN API. It then optimises near-misses, runs BRAIN's own submission checks, and submits **only when you confirm**.

It is a local, single-user app. Your BRAIN password and LLM API keys are kept in Windows Credential Manager, and all research data stays in a local SQLite database.

**New to AlphaGo?** Start with the step-by-step [User guide](USER_GUIDE.md), with screenshots of every page. This README is the reference.

---

## 1. Install and run

**Requirements**

- Windows 10/11 with the Microsoft Edge WebView2 runtime (preinstalled on current Windows).
- Python 3.11 or newer (as `pyproject.toml` requires). It was built and tested on 3.13.

```powershell
cd AlphaGo
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m alphago            # native window
.venv\Scripts\python.exe -m alphago --web      # same app in your browser at http://127.0.0.1:8090
.venv\Scripts\python.exe -m alphago --headless # the engine without a window (UI at http://127.0.0.1:8090)
```

- The UI listens on 127.0.0.1 only. To open it from another PC, add `--host 0.0.0.0 --token <secret>`, then open `http://<pc>:8090/?token=<secret>` once. The app refuses to listen on another address without a token.
- Only one AlphaGo engine runs per PC (window, browser or background), so the same quota is never spent twice.
- **Sync with BRAIN → Run in the background → Start with Windows** registers a Task Scheduler entry that starts `--headless` at logon, as you, so it can read your keys. While simulations are queued or running, the PC is kept awake (`service.keep_awake`).

**Build a standalone .exe** (no Python needed on the target PC):

```powershell
.venv\Scripts\python.exe scripts\build_exe.py             # -> dist\AlphaGo\AlphaGo.exe
.venv\Scripts\python.exe scripts\build_exe.py --console   # keeps a console window for debugging
```

- Zip the whole `dist\AlphaGo` folder to move it to another PC.
- `AlphaGo.exe --web` works there too.

**Where your data lives**

| What | Where |
|---|---|
| Database (`alphago.db`), settings (`settings.json`), logs (`logs\alphago.log`) | `%APPDATA%\AlphaGo` |
| BRAIN email, password and LLM API keys (Gemini, Anthropic, OpenAI) | Windows Credential Manager, service `AlphaGo` |

- To use a different folder, set the `ALPHAGO_HOME` environment variable. This is handy for a test profile.
- Back up the database by copying `alphago.db` while the app is closed.

---

## 2. First run (about 10 minutes)

1. **Sync with BRAIN**: sign in with your BRAIN email and password.
   - If BRAIN asks for biometric verification (Persona), click **Open verification page**. Finish it in the browser, then click **Continue**.
   - Your account's allowed settings (regions, universes, delays, neutralizations) load automatically after sign-in.
   - The header shows the session countdown and today's simulations left.
2. **Sync with BRAIN**: choose your main scope (for example USA / 1 / TOP3000) and click **Sync everything for scope**. This syncs operators, data categories, datasets and fields.
   - Then click **Sync my alphas** and **Pyramids & Genius**.
   - Field sync fetches up to 2 pages at a time (`catalog_concurrency` under `brain` in `settings.json`; 1 = one at a time), paced at 2 requests a second to start (`catalog_rps`). A full catalogue is thousands of 50-field pages, which used to trip BRAIN's `429 API rate limit exceeded`. Now every 429 pauses all catalogue requests for BRAIN's `Retry-After`, halves the pace and retries, and the pace creeps back up after a run of successes.
   - The sync **resumes**: datasets synced in the last 72 hours (`fields_resync_hours`) are skipped, so a rerun after an interruption carries on where it stopped. A dataset that still fails is retried once at the end and never stops the others. Watch it on **Tasks**.
3. **LLM Integration**: choose the **Active provider**, then click **Save & test**. One provider is active at a time and every AI feature uses it.
   - **Gemini**: paste a Google AI Studio key (https://aistudio.google.com/apikey).
   - **Claude**: paste an Anthropic API key (https://console.anthropic.com). Auto uses Claude Opus 5.5 for every role; Sonnet 5.5 or Haiku 5.5 cost less.
   - **OpenAI / compatible**: paste an OpenAI key, or change the server URL to Groq, DeepSeek, LM Studio, vLLM or any other server with the OpenAI chat API (local servers usually need no key).
   - **OpenRouter (free models)**: paste a key from https://openrouter.ai/keys. There is no fixed model list: the free models are read live from OpenRouter and ranked by size, context window, JSON support, retirement date and how each one has behaved in the app. Each call carries two backup models, and a retired (404) or busy (429/503) model is skipped at once. Free use is capped per account per day (about 50 requests, or 1,000 with $10 of credit), so add a fallback provider. Switch **Free models only** off to rank paid models too.
   - **Ollama (local)**: install Ollama (https://ollama.com), run `ollama pull <model>`, and keep the default URL `http://localhost:11434`. Set **Context window** high enough for the knowledge pack plus a dataset's fields (15–20k tokens). Local calls don't count toward the daily budget.
   - The test loads the models the provider offers, and the app picks one each for ideation, refinement and descriptions (OpenAI needs your pick). You can change the picks.
   - Cloud providers are paced automatically: the app slows down when a provider returns 429.
4. **Simulation Matrix**: replace the expressions with `rank(-ts_delta(close, 5))`, keep one neutralization and set **Decays** to `0`, so the preview reads 1 simulation. Click **Validate**, then **Run matrix**.
   - The result appears in the grid and on **Alphas**. This proves the whole pipeline works for your account.

---

## 3. What each tab does

| Tab | Purpose |
|---|---|
| **Dashboard** | Today at a glance: submittable alphas, simulations left, sent today, queued and busy slots, the AI research counters, and the live activity feed. |
| **Simulation Matrix** | One or more expressions × several settings (universe, neutralization, decay, truncation and more). Shows the quota cost first. Runs are packed into multi-simulations of 10 across the 8 slots. *Template* mode fills `{field}`, `{window}` and `{group}`. The heatmap shows whether an alpha is a robust plateau or a lucky spike. |
| **Dataset Intelligence** | Datasets rated by AI for **Single / Double / Interaction** research, with top recommendations, a themes chart and usage/theme filters, blended with your own hit rate. Per dataset: AI brief, fields, idea generation, **Research task…**. |
| **Alpha Autopilot** | Three modes: **Manual**, **Assisted** (default; proposed tasks wait for approval) and **Full Autopilot** (opt-in, within a daily simulation cap). Plans research tasks with a reason, shows the current task's stage bar and what was left out on purpose. The v1 AI idea batches (with approval) are in a collapsible section. |
| **Genius AI** | Your quarterly Genius progress: signals, pyramids lit, combined performance and the tie-breakers. Includes a pyramid map, pyramid candidates, and an AI-written plan of concrete goals you can launch as targeted Autopilot runs. |
| **Osmosis** | **Prepare data** (submitted alphas + daily PnL), then **Make best plan** for all scopes: leaves out weak alphas (last-2y Sharpe < 0.5 or turnover > 50%), picks the alphas whose daily PnL combine best (Max 15/20/30/40), 100,000 points per scope. Combo Sharpe, PnL chart, editable points, CSV, confirmed push. Lock shown in local time. |
| **Osmosis Review** | Points now on BRAIN, mistakes per scope, **Predict with AI** and **Explain my mistakes** (numbers only, English or Hinglish), plus allocation history. |
| **Power Pool** | Monthly pick of the best Power-Pool-shaped submitted alphas per scope (max per scope, rating floor, PnL correlation < 0.5), then confirmed bulk tag changes. Ratings, Calmar, 2-year Sharpe, last-year PnL, sparklines. The v1 eligibility scanner stays in a collapsible section. |
| **Out of Sample** | Estimated OS outlook from in-sample daily PnL: Holds (last-2y vs earlier Sharpe), risk 0–100, Robust / Likely fine / Fading / Overfit risk, Est. OS Sharpe, main reason. |
| **SPC Lab** | Systematic Predictions Challenge, end to end. **Generate prompt** (the AI writes a rules-based prompt with the exact output spec from your idea), **Dry-run** it on your AI provider several times, **Check** the JSON against editable form rules, **Test PnL** on Yahoo Finance prices, and check stability across runs. **Run all checks** shows the gates (character limit, output spec, no placeholders, valid output, runs agree), and **Submit to BRAIN** unlocks only when every gate passes and asks you to confirm. Also: prompt library and versions, and saved runs to re-test. |
| **Alpha Quality Project** | The monthly AQP cycle (tag `AQP` + year + month, for example `AQP202611`). Lists submitted USA alphas never tagged in an earlier cycle, ranked by distinctiveness plus out-of-sample evidence. **Suggest best 3** picks different ideas. **Write & grade with AI** writes a structured hypothesis description, then grades it two ways: a strict check that every field, operator, window and grouping is described, and an AI judge for match, uniqueness and hypothesis quality. It rewrites until the grade stops improving and estimates the score (80% description, 20% performance). **Tag & save on BRAIN** applies the tag and description after you confirm, refusing a 4th alpha or one used in an earlier cycle. |
| **Super Alphas** | Follows your Super Alpha guide: rename about 20 ACTIVE alphas to a code (for example `321`), select them with `name == '321'`, combine with `1`, preview, simulate, write the two descriptions (100+ characters each), then a confirmed submit. |
| **Tasks** | Research tasks (Search Lab, Double dataset · Pair, Interaction). Each first **screens the fields**: one simulation per field, keeping only fields that showed a signal, sign-flipped when negative. Templates run with windows suited to the data category, in the order your own hit history ranks their families. Shows cores, progress, best Sharpe, pause/stop/delete, **Run All / Stop All**, the Autopilot's daily plan, plus the simulation queue and background jobs. |
| **Alphas** | Every alpha, from the app or synced. Double-click for details: metrics, check badges, PnL and yearly charts, self-correlation, optimise, properties and AI description, `/check`, and **Submit…**. |
| **Research History** | Autopilot runs and the idea journal (hypothesis → verdict). Failed shapes become the AI's "don't repeat" memory. |
| **Analytics** | Simulations per day, success rate, Sharpe distribution, most frequent failing checks, yield by generator and dataset, and LLM usage. |
| **Quota Planner** | What 100 simulations have bought so far, per dataset and strategy: promising, strong, *passed BRAIN's check* and submitted per 100, with 80% ranges. Suggests a split of a quota (most where submittable alphas came from, 20% exploring) and can create the research tasks as *planned*, waiting for your approval. |
| **Health** | Per AI provider and day: calls, rate limits, timeouts, unavailable models, server errors, latency, tokens; per model error rates; BRAIN throttles (429), concurrency rejections, 5xx and network errors, quota used; **which repairs work** per failing check; how close the self-correlation estimate has been to BRAIN. |
| **Audit Log** | Every action the app took on your BRAIN account (simulations sent, checks, submissions, property/tag/Osmosis changes, sign-ins), who triggered it (you, a confirmed job, task N, Autopilot) and how many simulations it spent. Filters and CSV export; append-only. |
| **LLM Integration** | Active provider (Gemini, Claude, OpenAI / compatible, Ollama), its key or server, models per role, thinking level, call budget, rate-limit status, **fallback providers**, the **AI self-check** switch, the **model tournament** (scoreboard, lock the winner) and the prompt-template editor with **versions** (results per version, diff, restore). |
| **Sync with BRAIN** | Login (including Persona), allowed settings, and catalogue, alphas and programs syncs, with last-synced times. **Email alerts** (SMTP) and **Run in the background** (Start with Windows, keep awake). |

**Verdicts used throughout**

| Verdict | Meaning |
|---|---|
| `rejected` | Error, or far below the Sharpe limit. |
| `weak` | Below the bar for further work. |
| `promising` | At least 70% of the Sharpe limit, with failures that can be fixed. |
| `strong` | All in-sample checks pass, correlation not yet checked (or a pass went stale: you submitted another alpha in that region/delay since, so it needs a fresh check). |
| `ready` | BRAIN's `/check` passes, correlation included. A **fragile** badge means AlphaGo's own robustness test failed (neighbouring settings keep under 80% of the Sharpe); BRAIN would still accept it — you decide. |
| `submitted` | Already submitted. |

Limits always come from BRAIN's own check results. The app's built-in thresholds are only a fallback, and the limits BRAIN last reported for your account are given to the AI.

**How a near miss becomes ready** (research tasks and AI batches)

1. **Repair loop.** For each near miss the app measures which checks fail, by how much and why (`engine/playbook.py`): Fitness is split into turnover, thin returns or a weak signal using BRAIN's formula, and yearly PnL shows a fading edge or one bad year. Moves get values worked out for that alpha, and moves that cannot help are not simulated. Moves are ranked by how much distance to passing they should remove and by how often they have actually worked (Health → Which repairs work). Children that got clearly closer are repaired again for up to `research.repair_depth` rounds. Repairs use `research.optimize_share` (20%) of a task's simulations and run after every wave. A **Repair near misses** task (Tasks, Alphas) does the same for alphas you already have, within a budget you set: each round goes to whichever alpha is closest to passing, across all of them.
2. **BRAIN check on every strong alpha** (up to `research.check_cap`, 15 per task; a check costs no simulations), most different first: lowest estimated self-correlation (from daily PnL against your submitted alphas) and no two near-identical PnL twins in a row.
3. **After a pass:** robustness test (3 simulations; a badge, not a demotion), walk-forward (recent 20% of in-sample ÷ the first 80%), a search-adjusted (deflated) Sharpe from how many ideas the search tried, and — once 20 of your submitted alphas have BRAIN OS numbers — a predicted OS/IS ratio. All advisory.
4. **A correlation failure** gets several rounds aimed at what it collides with (`engine/decorrelate.py`): beat the correlated alpha's Sharpe by 10% (a community-reported rule, unverified; BRAIN's check decides), remove the overlap with `regression_neut`, swap in less-crowded fields, or change horizon or conditioning. Any other failure from BRAIN's check goes back to the repair loop.

Before simulating, generated ideas that copy a submitted alpha's shape and fields are left out, each economic idea (momentum, reversal, value, …) is capped per day at `research.cluster_share` of the simulations, and the AI critiques its own batch (`llm.self_check`).

---

## 4. A typical day

1. **Dashboard**: check your quota and anything left in the queue.
2. **Dataset Intelligence**: pick one or two high-opportunity datasets, ideally ones that would light a new pyramid (see **Genius AI**).
3. **Alpha Autopilot** (Assisted): start a run on that dataset. Review each proposed batch, edit or drop ideas, then approve.
4. **Alphas**: filter by verdict `ready` or `strong`. Open the best ones, run **Check** and review self-correlation.
5. **Submit…**: the dialog shows every check and today's count (limit: 4 regular, 1 super). Nothing is sent until you click **Submit**.
6. On Sunday, before the Osmosis lock, use **Osmosis** to rebalance points.

Practitioner data (about 830 simulations per live alpha) shows that **correlation, not generation, is the bottleneck**. Changing windows or neutralization rarely breaks correlation; a new dataset or a new economic mechanism does. The Autopilot's dataset choice and its "family cap" per expression skeleton are built around this.

---

## 5. Compliance and safety

- **API only.** The app never scrapes the website. It honours `Retry-After`, reduces concurrency on 429, and keeps a daily quota reserve (default 200 simulations, set in `settings.json`).
- **Nothing is submitted without your click.** Super Alpha renames, Osmosis pushes and Power Pool bulk tag changes each ask for confirmation too. SPC prompts are submitted only from **Submit to BRAIN**, after every check passes and you confirm. AQP tags and descriptions are applied only from **Tag & save on BRAIN**, after you confirm. Tag buttons and the AI description buttons write to BRAIN as soon as you click them (see section 6 of the [User guide](USER_GUIDE.md#6-safety-settings-and-troubleshooting)).
- **Assisted mode is the default.** Full Autopilot is capped at a daily number of simulations (default 500), and the top-bar **Stop all** button pauses everything.
- **Background:** BRAIN's *Consultant Do's and Don'ts* lists "using automation to generate Alpha ideas" as a research don't. There is also a community report of a fully automated account being locked.
  - Full Autopilot is behind a warning you must accept.
  - Please read the current Do's and Don'ts on the platform. Which mode you run is your decision.
- **Credentials.** They are stored only in Windows Credential Manager and are never written to logs or the database.
- **Accountability.** Every action on your BRAIN account is in the **Audit Log**. Email alerts carry alpha IDs and counts only, never expressions or credentials, and are off until you switch them on.

---

## 6. Things the app discovers at runtime

These details are not publicly documented, so the app handles them as it goes rather than hard-coding them:

| Detail | How it is handled |
|---|---|
| **SUPER payload shape** | Selection and combo at the top level, or nested. The app tries one and falls back to the other on rejection, then remembers which works. |
| **Allowed settings** | Regions, universes, delays and neutralizations come from `OPTIONS /simulations` for your account. A built-in table is used until you load them. |
| **Quota and reset time** | Read from the `X-Ratelimit-*` headers. |
| **SPC form rules** | Not public. Enter the rules under **Form rules** (required fields, row limits, weight and long/short fields). |
| **SPC submission endpoint** | Not public. Submit one prompt on BRAIN's SPC page with the browser's developer tools open (F12, then Network). Right-click the request, choose **Copy as cURL** (or **Copy as fetch**), paste it into **SPC Lab → BRAIN submission endpoint and gates → Captured request** and click **Import**. The path, method, competition id and body template are filled in. Until then, Submit stays off. **Find my competitions** lists your competition ids. |
| **BRAIN API rate limits** | Not published. The client learns them: catalogue calls start at `catalog_rps`, halve on every 429 and recover slowly. |
| **Osmosis lock time** | Default Sunday 23:59 New York, inferred from the reference app; editable on Osmosis. |
| **Genius thresholds** | Community-reported values, editable on **Genius AI**. |

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "BRAIN signed out" chip stays red | Sign in on **Sync with BRAIN**. If Persona is pending, finish verification in the browser. |
| No universes or neutralizations to choose from | Sign in, or click **Sync with BRAIN → Refresh from BRAIN** in the allowed-settings card. |
| "Field not found" during validation | **Sync everything for scope** for that region, delay and universe. |
| `GET /data-fields → 429: API rate limit exceeded` | Fixed: the sync now waits it out. If it still happens a lot, set `catalog_rps` to 1 and `catalog_concurrency` to 1 in `settings.json`, then run the sync again. It resumes where it stopped. |
| OpenRouter "daily free-model limit reached" | The free tier's daily cap. It resets at midnight UTC. Add a fallback provider, or add $10 of credit for 1,000 free requests a day. |
| LLM 429 or budget reached | Wait. Pacing recovers automatically. Raise the daily budget on **LLM Integration**, use a paid tier, or switch to a local Ollama model (not budgeted). |
| "Ollama is not reachable" | Start Ollama (it runs at `http://localhost:11434` by default), or fix the server URL on **LLM Integration**. |
| Ollama warns the prompt was cut | Raise **Context window** on **LLM Integration**, or use a model with a longer context. |
| Native window does not open | Install the WebView2 runtime, or run with `--web`. |
| Anything else | See `%APPDATA%\AlphaGo\logs\alphago.log`. |

---

## 8. For developers

```powershell
.venv\Scripts\python.exe -m pytest -q          # 85 tests: BRAIN client (mocked), FASTEXPR, pipeline, PnL/Osmosis/Power Pool/SPC/research tasks, migration
.venv\Scripts\ruff.exe check alphago tests
```

**Code layout**

| Path | Contents |
|---|---|
| `alphago/brain/` | Async BRAIN API client: auth and Persona, quota, adaptive slots, adaptive request pacing (`ratelimit.py`), simulations, alphas, data, users. |
| `alphago/fastexpr/` | Lark grammar, parser, validator, canonical and skeleton hashes, AST transforms. |
| `alphago/knowledge/` | Operator signatures, templates, and the LLM knowledge pack. |
| `alphago/llm/` | LLM router (`llm`) and providers — Gemini, Claude, OpenAI / compatible, OpenRouter (live free-model ranking and rotation), Ollama — with structured output, adaptive pacing and the daily budget; schemas, prompts. |
| `alphago/engine/` | Simulator service (per-task cores), evaluator, generators, optimiser, Autopilot, research tasks (with field screening), template-family learning (`families.py`), daily PnL, portfolio logic (Osmosis plan/review, Power Pool pick), OOS outlook, SPC (generate, gates, submit), Alpha Quality Project (`aqp.py`), programs, Super Alphas, stats. |
| `alphago/jobs/` | Persistent background job queue. |
| `alphago/ui/` | NiceGUI pages (one module per tab), layout, theme, shared components. |
