# What's in this bundle

New/changed files to drop into your `Auto-FMVA` repo (same relative paths):

- `orchestrate.py`                 — Full Step 1-7 pipeline runner. `--provider` flag
                                      picks how Step 4 (the narrative writing) gets done.
- `financial_engine/qa.py`         — Step 7 QA gate.
- `financial_engine/report.py`     — Step 5 Excel workbook assembler.
- `financial_engine/llm_narrative.py` — NEW. Free/local-model adapter for Step 4 — same
                                      prompts + validation as narrative.py, different backend.
- `data/raw/RELIANCE_peers.csv`    — Sample peer-comps input (replace with real data).
- `data/processed/RELIANCE/...`    — sample output of a full run (workbook, results.json,
                                      the 4 validated narrative modules + raw replies).

## Two ways to run Step 4 — no paid Anthropic API either way

### 1. Manual (default) — paste into a Claude web chat yourself
```bash
python3 orchestrate.py
```
Writes `narrative_<module>_PROMPT.txt` per module. Paste each into a Claude chat, save the
reply into the matching `narrative_<module>_REPLY.txt`, re-run. Free if you're on a Claude
Pro/free plan — this is literally you using claude.ai normally, just organized by file.

### 2. Automatic — free/cheap third-party providers
```bash
python3 orchestrate.py --provider groq       # needs GROQ_API_KEY   (groq.com — free tier)
python3 orchestrate.py --provider gemini     # needs GEMINI_API_KEY (Google AI Studio — free tier)
python3 orchestrate.py --provider openrouter # needs OPENROUTER_API_KEY (has free models)
python3 orchestrate.py --provider ollama     # no key — needs `ollama serve` running locally
```
Same prompts, same schema + numeric-fidelity validation as the manual path (never-invent-
numbers, anomaly flags, few-shot tone-matching) — `llm_narrative.py` just calls a different
HTTP endpoint. No cost, no Anthropic API key, in any of these four. Get free keys at:
- Groq:       console.groq.com/keys
- Gemini:     aistudio.google.com/apikey
- OpenRouter: openrouter.ai/keys (filter for ":free" models)
- Ollama:     just install it and `ollama pull llama3.1`, no signup at all

Quality note: these are not Claude, so treat their output the same way the whole pipeline
already treats any narrative — as a draft. The QA gate (Step 7) and its numeric-fidelity
check still validate everything regardless of which provider wrote it.

## One real bug this surfaced

`shares_outstanding` in `financials.csv` is a raw share count, but cash/debt/EBIT are in
Rs crore. `orchestrate.py`'s DCF/comps wiring converts shares to crore-of-shares before
dividing — without that, equity_value_per_share came out as ~Rs 0.0001. `qa.py` now flags
`equity_value_per_share < 1` as high-severity so this can't silently ship again.
