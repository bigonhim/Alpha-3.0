# AlphaGo: a WorldQuant BRAIN alpha research desktop app

AlphaGo is a Windows desktop app for BRAIN consultants. It finds data, generates FASTEXPR alphas with Gemini and templates, and simulates them through the official BRAIN API. It then optimises near-misses, runs BRAIN's own submission checks, and submits **only when you confirm**.

It is a local, single-user app. Your BRAIN password and Gemini key are kept in Windows Credential Manager, and all research data stays in a local SQLite database.

**New to AlphaGo?** Start with the step-by-step [User guide](USER_GUIDE.md), with screenshots of every page. This README is the reference.

---

## 1. Install and run

**Requirements**

- Windows 10/11 with the Microsoft Edge WebView2 runtime (preinstalled on current Windows).
- Python 3.12 or newer. It was built and tested on 3.13.

```powershell
cd AlphaGo
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .[dev]
.venv\Scripts\python.exe -m alphago            # native window
.venv\Scripts\python.exe -m alphago --web      # same app in your browser at http://127.0.0.1:8090
```

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
| BRAIN email, password and Gemini key | Windows Credential Manager, service `AlphaGo` |

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
   - Field sync is rate-limited and can take a few minutes. Watch it on **Tasks**.
3. **LLM Integration**: paste a Google AI Studio key (https://aistudio.google.com/apikey) and click **Save & test**.
   - The test also loads the models your key can use, and the app picks one each for ideation, refinement and descriptions. You can change the picks.
   - The app detects your tier and slows down on its own when Gemini returns 429.
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
| **Genius AI** | Your quarterly Genius progress: signals, pyramids lit, combined performance and the tie-breakers. Includes a pyramid map, pyramid candidates, and a Gemini-written plan of concrete goals you can launch as targeted Autopilot runs. |
| **Osmosis** | **Prepare data** (submitted alphas + daily PnL), then **Make best plan** for all scopes: leaves out weak alphas (last-2y Sharpe < 0.5 or turnover > 50%), picks the alphas whose daily PnL combine best (Max 15/20/30/40), 100,000 points per scope. Combo Sharpe, PnL chart, editable points, CSV, confirmed push. Lock shown in local time. |
| **Osmosis Review** | Points now on BRAIN, mistakes per scope, **Predict with AI** and **Explain my mistakes** (numbers only, English or Hinglish), plus allocation history. |
| **Power Pool** | Monthly pick of the best Power-Pool-shaped submitted alphas per scope (max per scope, rating floor, PnL correlation < 0.5), then confirmed bulk tag changes. Ratings, Calmar, 2-year Sharpe, last-year PnL, sparklines. The v1 eligibility scanner stays in a collapsible section. |
| **Out of Sample** | Estimated OS outlook from in-sample daily PnL: Holds (last-2y vs earlier Sharpe), risk 0–100, Robust / Likely fine / Fading / Overfit risk, Est. OS Sharpe, main reason. |
| **SPC Lab** | Systematic Predictions Challenge workbench: prompt library and versions, form settings, copy prompt, **Check** the pasted JSON output against editable form rules, **Test PnL** on Yahoo Finance prices, stability check across runs, saved runs to re-test. Never submits. |
| **Super Alphas** | Follows your Super Alpha guide: rename about 20 ACTIVE alphas to a code (for example `321`), select them with `name == '321'`, combine with `1`, preview, simulate, write the two descriptions (100+ characters each), then a confirmed submit. |
| **Tasks** | Research tasks (Search Lab, Double dataset · Pair, Interaction) with cores, progress, best Sharpe, pause/stop/delete, **Run All / Stop All**, the Autopilot's daily plan, plus the simulation queue and background jobs. |
| **Alphas** | Every alpha, from the app or synced. Double-click for details: metrics, check badges, PnL and yearly charts, self-correlation, optimise, properties and AI description, `/check`, and **Submit…**. |
| **Research History** | Autopilot runs and the idea journal (hypothesis → verdict). Failed shapes become Gemini's "don't repeat" memory. |
| **Analytics** | Simulations per day, success rate, Sharpe distribution, most frequent failing checks, yield by generator and dataset, and Gemini usage. |
| **LLM Integration** | Gemini key, models per role, thinking level, call budget, rate-limit status, and the prompt-template editor with reset. |
| **Sync with BRAIN** | Login (including Persona), allowed settings, and catalogue, alphas and programs syncs, with last-synced times. |

**Verdicts used throughout**

| Verdict | Meaning |
|---|---|
| `rejected` | Error, or far below the Sharpe limit. |
| `weak` | Below the bar for further work. |
| `promising` | At least 70% of the Sharpe limit, with failures that can be fixed. |
| `strong` | All in-sample checks pass, correlation not yet checked. |
| `ready` | BRAIN's `/check` passes, correlation included. |
| `submitted` | Already submitted. |

Limits always come from BRAIN's own check results. The app's built-in thresholds are only a fallback.

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
- **Nothing is submitted without your click.** Super Alpha renames, Osmosis pushes and Power Pool bulk tag changes each ask for confirmation too. SPC prompts are never submitted by the app. Tag buttons and the AI description buttons write to BRAIN as soon as you click them (see section 6 of the [User guide](USER_GUIDE.md#6-safety-settings-and-troubleshooting)).
- **Assisted mode is the default.** Full Autopilot is capped at a daily number of simulations (default 500), and the top-bar **Stop all** button pauses everything.
- **Background:** BRAIN's *Consultant Do's and Don'ts* lists "using automation to generate Alpha ideas" as a research don't. There is also a community report of a fully automated account being locked.
  - Full Autopilot is behind a warning you must accept.
  - Please read the current Do's and Don'ts on the platform. Which mode you run is your decision.
- **Credentials.** They are stored only in Windows Credential Manager and are never written to logs or the database.

---

## 6. Things the app discovers at runtime

These details are not publicly documented, so the app handles them as it goes rather than hard-coding them:

| Detail | How it is handled |
|---|---|
| **SUPER payload shape** | Selection and combo at the top level, or nested. The app tries one and falls back to the other on rejection, then remembers which works. |
| **Allowed settings** | Regions, universes, delays and neutralizations come from `OPTIONS /simulations` for your account. A built-in table is used until you load them. |
| **Quota and reset time** | Read from the `X-Ratelimit-*` headers. |
| **SPC form rules** | Not public. SPC Lab never submits; enter the rules under **Form rules** (required fields, row limits, weight and long/short fields). |
| **Osmosis lock time** | Default Sunday 23:59 New York, inferred from the reference app; editable on Osmosis. |
| **Genius thresholds** | Community-reported values, editable on **Genius AI**. |

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "BRAIN signed out" chip stays red | Sign in on **Sync with BRAIN**. If Persona is pending, finish verification in the browser. |
| No universes or neutralizations to choose from | Sign in, or click **Sync with BRAIN → Refresh from BRAIN** in the allowed-settings card. |
| "Field not found" during validation | **Sync everything for scope** for that region, delay and universe. |
| Gemini 429 or budget reached | Wait. Pacing recovers automatically. Raise the daily budget on **LLM Integration**, or use a paid tier. |
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
| `alphago/brain/` | Async BRAIN API client: auth and Persona, quota, adaptive slots, simulations, alphas, data, users. |
| `alphago/fastexpr/` | Lark grammar, parser, validator, canonical and skeleton hashes, AST transforms. |
| `alphago/knowledge/` | Operator signatures, templates, and the Gemini knowledge pack. |
| `alphago/llm/` | Gemini client (model discovery, structured output, adaptive pacing), schemas, prompts. |
| `alphago/engine/` | Simulator service (per-task cores), evaluator, generators, optimiser, Autopilot, research tasks, daily PnL, portfolio logic (Osmosis plan/review, Power Pool pick), OOS outlook, SPC, programs, Super Alphas, stats. |
| `alphago/jobs/` | Persistent background job queue. |
| `alphago/ui/` | NiceGUI pages (one module per tab), layout, theme, shared components. |
