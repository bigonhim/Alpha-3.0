# Submitting an SPC prompt to BRAIN by hand, and setting up AlphaGo from it

_Last updated 11 November 2026._

This guide walks you through two things, one small step at a time:

- **Part A–C:** submit one Systematic Predictions Challenge (SPC) prompt yourself on the BRAIN website, while your
  browser records the exact request it sends.
- **Part D–G:** copy what you learned into AlphaGo's **SPC Lab**, so AlphaGo can check prompts against the same rules
  and, after your confirmation, submit them the same way.

> **Why do it by hand first?** WorldQuant BRAIN publishes no SPC rules and no SPC API. AlphaGo cannot know the form's
> rules, the address it posts to, or the names of its fields until you show it one real submission. Your first
> submission is therefore always manual. It counts as a real submission.

**What is confirmed and what is not.** The steps that happen inside AlphaGo match the app exactly. The BRAIN website
side is written from community reports, because BRAIN's SPC pages need a login and are not public:

- **Reported:** the form has a name, the prompt, a model and model version, an update frequency and a weight between
  0 and 1. Prompts can be up to 10,000 characters.
- **Reported, not confirmed:** at most 50 active prompts and 50 submissions a day.
- **Not confirmed:** the exact menu names and the output fields BRAIN expects.

Wherever the website differs from this guide, trust the website, and copy what it shows into AlphaGo.

---

## Before you start (5 minutes)

| You need | Why |
| --- | --- |
| A BRAIN consultant login with access to the SPC | The challenge is only open to enrolled consultants |
| **Google Chrome** or **Microsoft Edge** on Windows | Their developer tools can copy the request (Part C) |
| AlphaGo running and signed in (**Sync with BRAIN**, green "Signed in") | AlphaGo submits with your own BRAIN session |
| An AI provider set up on **LLM Integration** (optional) | Only for "Generate & check" (Part F) |
| About 30 minutes | Parts A–E are once only |

Keep two windows side by side: the browser on the left, AlphaGo on the right.

---

## Part A — Read the rules on BRAIN (once)

1. Open https://platform.worldquantbrain.com and sign in.
2. Open the SPC. It is usually reached from the **Competitions** (or **Events**) menu at the top, or from the SPC
   announcement in the notifications bell. Its title reads "Systematic Predictions Challenge".
3. If BRAIN asks you to **join** or **accept the rules**, read them and accept. AlphaGo cannot do this step for you.
4. Open the SPC's **rules / FAQ** page, for example the support article "What do I need to submit". Write these
   down on paper or in Notepad:

   | Write down | Example (check yours) | Goes into AlphaGo at… |
   | --- | --- | --- |
   | Maximum prompt length | 10,000 characters | Form rules → **Prompt character limit** |
   | Fields each output row must have | `company_name`, `ticker` (some rule sets add `isin`, `mic`, `confidence_score`, …) | Form rules → **Required fields** |
   | Number of stocks (min / max) | e.g. at least 1, about 50 suggested | Form rules → **Min rows** / **Max rows** |
   | Is there a long/short field? Its name and values | e.g. `side` = long/short, or a signed `confidence_score` | Form rules → **Long/short field** |
   | Is there a weight per row? Its name | e.g. `weight` | Form rules → **Weight field** |
   | Allowed models and versions | e.g. gpt / 5.6 | 1. Prompt → **Model**, **Version** |
   | Allowed update frequencies | Daily, Weekly, Monthly, Quarterly | 1. Prompt → **Frequency** |
   | Weight range | 0.00 to 1.00 (0 = inactive) | 1. Prompt → **Weight** |
   | Daily submission limit | e.g. 50 | Submit settings → **Max submissions per day** (set lower; see Part E) |

5. Keep that note open; you will need it in Part D.

---

## Part B — Prepare one prompt to submit

You need a prompt you are happy to submit for real.

1. In AlphaGo, open **SPC Lab** (sidebar).
2. In **1. Prompt**, open **Prompt library** and choose a ★ starter, for example **★ Quality momentum (long only)
   v1**. Click **Load**.
3. Read the prompt. Change anything you like. Make sure its last lines describe exactly the output fields from your
   Part A note.
4. Watch the counter under the prompt, for example "1,234 / 10,000 characters". It must stay below the limit.
5. Click **Copy prompt**. The prompt is now on your clipboard.

_Optional but recommended:_ ask any chat AI (or the model named in the form) to run the prompt once, paste its reply
into **2. Output**, and click **Check**. A green "✓ Passes the form rules" means the output shape is right.

---

## Part C — Submit it by hand while the browser records the request

This is the step that teaches AlphaGo the address and the field names.

1. In the browser, go to the SPC **submission form** (the page with the Prompt box).
2. Press **F12** (or right-click the page → **Inspect**). The developer tools open.
3. Click the **Network** tab at the top of the developer tools.
4. Tick **Preserve log**, and click **Fetch/XHR** in the filter row, so only API calls are listed.
5. Click the round **clear** button (🚫) to empty the list.
6. Fill in the form, field by field:
   - **Name:** for example `Quality momentum v1`.
   - **Prompt:** paste it (Ctrl+V).
   - **Model**, **Version**, **Frequency**, **Weight:** as in your Part A note.
7. Click the form's **Submit** button. Confirm if BRAIN asks.
8. Wait until BRAIN shows that the prompt was submitted (and appears in your list of prompts).
9. In the Network list, look for the new request that **sent** the form:
   - its **Method** is `POST` (sometimes `PUT`);
   - its **Name** or URL is on `api.worldquantbrain.com`;
   - clicking it and opening **Payload** shows your prompt text.

   If you see several, pick the one whose Payload contains your prompt.
10. **Right-click** that request → **Copy** → **Copy as cURL (bash)**. **Copy as cURL (cmd)** and **Copy as fetch**
    work too.
11. Paste it into Notepad for a moment. Check that it contains your prompt text.

> **Keep this text private.** A copied request contains your session cookie. Do not paste it into chats, emails or
> tickets. AlphaGo's **Import** reads only the address, the method and the body; it throws the cookies and headers
> away and never stores them. Delete the Notepad copy when you are done.

12. On the BRAIN page, open your list of SPC prompts and check that the one you just sent is there. If it is not,
    stop here and fix the problem on BRAIN before going on.

---

## Part D — Set AlphaGo's form rules and defaults (once)

1. In **SPC Lab**, scroll down and open **Form rules (editable — BRAIN does not publish the SPC rules)**.
2. Fill in each field from your Part A note:
   - **Required fields (comma separated):** for example `company_name, ticker`.
   - **Min rows** and **Max rows (0 = no limit)**.
   - **Weight field (empty = equal weight):** leave empty if rows carry no weight.
   - **Long/short field (empty = all long):** for example `side` for a long/short form.
   - **Benchmark ticker:** `SPY` is fine for US prompts (used by Test PnL only).
   - **Prompt character limit:** for example `10000`.

   Each field saves as you type.
3. Scroll up to **1. Prompt → Form settings**. Set **Model**, **Version**, **Frequency** and **Weight** to what you
   submitted in Part C.
4. Click the **📌 pin** button at the end of that row. A green message says "Defaults saved". From now on, new and
   AI-generated prompts use these values.

---

## Part E — Teach AlphaGo the submission request (once)

1. In **SPC Lab**, find the card **Generate with AI, check, and submit**. Open **Submit settings (BRAIN publishes no
   SPC endpoint)**.
2. Click into **Import a captured request (Copy as cURL / Copy as fetch)** and paste the request from Part C (Ctrl+V).
3. Click **Import**. A green message lists what it understood, for example:
   `Imported POST /competitions/{competition}/predictions. Filled: name→{name}, prompt→{prompt}, model→{model},
   version→{version}, frequency→{frequency}, weight→{weight}.`
4. Check the fields that **Import** filled in:
   - **Path (relative, starts with /):** the address, for example `/competitions/{competition}/predictions`. It
     must start with one `/`. AlphaGo refuses full `https://` addresses, so your session is never sent elsewhere.
   - **Method:** usually `POST`.
   - **Competition id (fills {competition}):** filled when the address contained one. If it is empty but the path
     shows `{competition}`, click **Find my competitions** and press **Use** next to the SPC.
   - **JSON body:** your captured body, with your values replaced by placeholders: `{prompt}`, `{name}`, `{model}`,
     `{version}`, `{frequency}` and `{weight}` (or `{weight:num}` where BRAIN expects a number). Values that are
     not form fields, such as `"draft": false`, stay exactly as captured.
5. Read the status line under the switch. **✓ Endpoint settings are valid.** means AlphaGo can build the request. A
   **✗** line tells you exactly what is missing.
6. If **Import** said "no prompt field", you copied the wrong request. Go back to Part C step 9 and pick the request
   whose Payload contains your prompt.
7. Set **Max submissions per day** to a safe number. Start with `1` or `2`, below BRAIN's own limit. AlphaGo counts
   every submission it sent today, including any whose outcome was unknown, and refuses past this number.
8. Leave **Submit automatically once every gate passes (no confirmation click)** **off** for now.

### Where to type placeholders by hand (if you prefer not to use Import)

Copy the address and the Payload from the request (Part C step 9) and type them into **Path** and **JSON body**. Then
replace your own values with `{prompt}`, `{name}`, `{model}`, `{version}`, `{frequency}`, `{weight}` and
`{competition}`. Example:

```json
{"name": "{name}", "prompt": "{prompt}", "model": "{model}", "modelVersion": "{version}",
 "updateFrequency": "{frequency}", "weight": "{weight:num}"}
```

Quotes, line breaks and braces inside your prompt are safe: AlphaGo fills each placeholder once and never treats
the prompt text as JSON.

---

## Part F — Check a prompt in AlphaGo, then send it with your confirmation

1. **Get a prompt checked.** In **Generate with AI, check, and submit**, do one of these:
   - type an idea in **Theme / idea**, set **Stocks** and **Portfolio**, and click **Generate & check**; or
   - load or write a prompt in **1. Prompt**, then click **Check the prompt in the editor**. Tick "let the AI rewrite
     the editor prompt if it fails" if you want the AI to fix it.
2. Wait for the progress bar. You can follow the job on **Tasks** too.
3. Read the report. Each gate shows ✓ pass, ⚠ warning, – off, ✗ fail or ? could not run:

   | Gate | What it checks |
   | --- | --- |
   | Prompt length | under the character limit |
   | Prompt format | the prompt asks for JSON and names every output field |
   | No placeholders | no leftover `{{…}}`, `<insert…>`, `TODO`, `[your…]` |
   | Form rules | every AI run returned rows that pass your Form rules |
   | Clean output | the replies were the bare JSON array |
   | Stability | the runs picked mostly the same stocks |
   | Tickers priced | the tickers exist on Yahoo Finance |
   | Backtest | optional; off by default |

   Only a report with no ✗ and no ? is **ready to submit**. Change how strict the gates are under **Gate settings**.
   The runs use your AI provider, not BRAIN's model, so they test the form, the tickers and the stability, not the
   returns.
4. Click **Preview request**. It shows the method, the path and the exact body AlphaGo would send. **Nothing is
   sent.**
5. **Compare the preview with your captured request from Part C.** Same path, same field names, same kind of values
   (a number stays a number). If anything differs, fix **Path** or **JSON body** in Submit settings first.
6. Click **Submit to BRAIN…**. The confirmation box shows the prompt, every gate, the request line and "Today's SPC
   submissions: 0/1".
7. Click **Submit to BRAIN**. Follow the job in the progress bar or on **Tasks**.
8. Read the result line under the report:

   | Result | What it means | What to do |
   | --- | --- | --- |
   | `accepted` | BRAIN answered 2xx | Check the prompt on BRAIN's SPC page |
   | `rejected` | BRAIN refused it (4xx), with BRAIN's reason | Read the reason; fix the prompt or the body |
   | `throttled` | BRAIN answered 429 | Wait a few minutes |
   | `error` | BRAIN answered 5xx | **Check BRAIN before retrying**; it may have arrived |
   | `unknown` | The connection dropped, or a redirect | **Check BRAIN before retrying**; it counts toward your daily limit |

9. Open BRAIN's SPC page and confirm the prompt is there with the right name, model, frequency and weight. Only
   after this first app-sent prompt matches what you sent by hand should you consider Part G.

AlphaGo never re-sends a submitted prompt version, never retries a failed send by itself, and refuses a prompt whose
text changed after it passed the checks. To change one, edit it, run **Check the prompt in the editor** again, and
submit the new version.

---

## Part G — Optional automation

When at least one app-sent submission has been confirmed on BRAIN:

1. Open **Submit settings** and switch on **Submit automatically once every gate passes (no confirmation click)**.
2. From now on, **Generate & check** submits a prompt as soon as every gate passes, within **Max submissions per
   day**. If anything refuses (endpoint, daily limit), the prompt stays saved as ready and you can submit it by hand.
3. Switch it off again at any time. It is off by default and applies to SPC prompts only. Alphas are never submitted
   without your click.

---

## Recommended first settings

| Where | Setting | Start with |
| --- | --- | --- |
| 1. Prompt → Form settings | Model, Version, Frequency, Weight | Exactly what you used by hand in Part C, then 📌 |
| Form rules | Required fields | The output fields from BRAIN's rules |
| Form rules | Prompt character limit | BRAIN's limit (10,000 reported) |
| Gate settings | Runs per round | 3 |
| Gate settings | AI rewrites | 2 |
| Gate settings | Min overlap between runs | 0.50 |
| Gate settings | Check tickers on Yahoo / Min share priced | on / 0.90 |
| Gate settings | Backtest days | 0 (off) |
| Submit settings | Max submissions per day | 1–2 at first |
| Submit settings | Submit automatically | off |

---

## When something goes wrong

| What you see | What to do |
| --- | --- |
| Import: "No URL found" | You copied something other than a request. Repeat Part C step 10 |
| Import: "no prompt field" | Wrong request. Pick the POST whose Payload contains your prompt |
| Import warns the request went to another host | AlphaGo only sends to BRAIN's API. Copy the request to `api.worldquantbrain.com` |
| "✗ The submit path uses {competition}, but no competition id is set." | **Find my competitions** → **Use**, or type the id |
| "✗ The submit body has no {prompt} placeholder" | Put `"{prompt}"` where your prompt text was in the body |
| Preview differs from the captured request | Edit **Path** / **JSON body** until they match |
| `rejected: …` | Read BRAIN's reason; the form rules or the body differ from BRAIN's |
| `unknown` | Look on BRAIN's SPC page before trying again |
| "Daily SPC submission guard reached" | You hit **Max submissions per day**. Raise it only if BRAIN allows more |
| "The prompt text changed after it passed the checks" | Run **Check the prompt in the editor** again |
| Session expired | Sign in again on **Sync with BRAIN**; AlphaGo renews the session with saved credentials |
| BRAIN changed the form | Repeat Parts C and E with a fresh capture |
