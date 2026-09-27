"""Step 4, alternate provider path. Same contract as financial_engine/narrative.py
(build_system_prompt, NARRATIVE_SCHEMAS, _validate_schema, numeric-fidelity check) — the
only thing that changes is which HTTP endpoint answers the prompt. Anthropic's API is paid
per token; this module talks to any OpenAI-compatible chat-completions endpoint instead, so
you can point it at a provider with a free tier, or a fully local model.

Tested-compatible providers (all speak the same request/response shape):
  groq       https://api.groq.com/openai/v1/chat/completions   -- free tier, fast, needs a key
  gemini     https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
                                                                  -- free tier, needs a key
  openrouter https://openrouter.ai/api/v1/chat/completions      -- has free models, needs a key
  ollama     http://localhost:11434/v1/chat/completions         -- fully local, no key, no cost
                                                                     (run `ollama serve` first)

None of this touches api.anthropic.com or costs anything from Anthropic's side. Bucket A
(the math) is completely unaffected either way -- this only changes who writes the prose.
"""
from __future__ import annotations
import json
import os

import requests

from .narrative import (
    NARRATIVE_SCHEMAS, build_system_prompt, _safe_json_parse, _validate_schema,
    _check_numeric_fidelity, NarrativeError,
)

# name -> (default base_url, default model, needs_api_key)
# NOTE: provider model IDs change over time. Keep these pinned to currently available values,
# and allow --model to override them when a provider has newer or region-specific options.
PROVIDER_PRESETS = {
    "groq":       ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b", True),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash", True),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free", True),
    "ollama":     ("http://localhost:11434/v1", "llama3.1", False),
}


def _compact_results_for_prompt(results: dict, max_rows_per_table: int = 2) -> dict:
    """Keep the narrative prompt compact enough for free-tier providers while retaining the
    salient financial context needed to write a valid section.
    """
    if not isinstance(results, dict):
        return results

    compact: dict = {}

    profile = results.get("profile") or {}
    compact["profile"] = {
        "company_name": profile.get("company_name"),
        "ticker_nse": profile.get("ticker_nse"),
        "current_price": profile.get("current_price"),
        "market_cap": profile.get("market_cap"),
    }

    financials = results.get("financials") or []
    if financials:
        latest = financials[-1]
        compact["latest_financials"] = {
            "year": latest.get("year"),
            "sales": latest.get("sales"),
            "sales_growth": latest.get("sales_growth"),
            "ebitda_margin": latest.get("ebitda_margin"),
            "net_margin": latest.get("net_margin"),
            "eps": latest.get("eps"),
            "borrowings": latest.get("borrowings"),
            "cash_bank": latest.get("cash_bank"),
            "equity": (latest.get("equity_share_capital") or 0) + (latest.get("reserves") or 0),
        }
        if len(financials) >= 2:
            start = financials[0]
            end = financials[-1]
            compact["trailing_growth"] = {
                "sales_3y_cagr": round((end.get("sales") / start.get("sales")) ** (1 / max(1, len(financials) - 1)) - 1, 4) if start.get("sales") else None,
                "ebitda_margin_range": [min(float(r.get("ebitda_margin") or 0) for r in financials[-max_rows_per_table:]), max(float(r.get("ebitda_margin") or 0) for r in financials[-max_rows_per_table:])],
            }

    ratios = results.get("ratios") or []
    if ratios:
        latest_ratio = ratios[-1]
        compact["ratios"] = {k: latest_ratio.get(k) for k in ("roe", "roic", "debt_to_equity", "current_ratio", "operating_margin") if k in latest_ratio}

    forecast = results.get("forecast")
    if isinstance(forecast, dict):
        compact["forecast_summary"] = forecast.get("summary")

    dcf_summary = results.get("dcf_summary") or {}
    compact["dcf_summary"] = {
        k: dcf_summary.get(k) for k in ("equity_value_per_share", "wacc", "terminal_growth") if k in dcf_summary
    }

    wacc = results.get("wacc") or {}
    compact["wacc"] = {
        k: wacc.get(k) for k in ("wacc", "cost_of_equity", "post_tax_cost_of_debt") if k in wacc
    }

    beta = results.get("beta_drifting") or {}
    if beta:
        compact["beta_drifting"] = {k: beta.get(k) for k in ("adjusted_beta", "market_beta") if k in beta}

    mc_summary = results.get("monte_carlo_var_summary") or []
    if mc_summary:
        compact["mc_var"] = mc_summary[-1] if isinstance(mc_summary, list) else mc_summary

    return compact


def _call_openai_compatible(system_prompt: str, user_content: str, base_url: str, model: str,
                             api_key: str | None, max_tokens: int = 2000, timeout: int = 120) -> str:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise NarrativeError(f"Unexpected response shape from provider: {data}")
    if not text:
        raise NarrativeError(f"Provider returned no text content: {data}")
    return text


def generate_narrative_free(module: str, company: str, results: dict, few_shot_example: dict,
                             provider: str = "groq", base_url: str | None = None, model: str | None = None,
                             api_key: str | None = None, max_retries: int = 1) -> dict:
    """Same contract as narrative.generate_narrative() (never invent numbers, flag anomalies,
    match few-shot tone, JSON-only, schema-validated, numeric-fidelity-checked) but against a
    free/cheap OpenAI-compatible endpoint instead of the paid Anthropic API.

    `provider` picks a preset (see PROVIDER_PRESETS) for base_url/model/whether a key is
    needed; pass base_url/model explicitly to override, or use provider="custom" and supply
    both yourself. `api_key` falls back to the {PROVIDER}_API_KEY env var (e.g. GROQ_API_KEY).
    """
    if module not in NARRATIVE_SCHEMAS:
        raise ValueError(f"Unknown narrative module '{module}'. Choose from {list(NARRATIVE_SCHEMAS)}")

    if provider != "custom":
        if provider not in PROVIDER_PRESETS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDER_PRESETS)} or 'custom'.")
        preset_base_url, preset_model, needs_key = PROVIDER_PRESETS[provider]
        base_url = base_url or preset_base_url
        model = model or preset_model
        if needs_key:
            api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
            if not api_key:
                raise NarrativeError(
                    f"provider='{provider}' needs an API key. Set {provider.upper()}_API_KEY "
                    f"or pass api_key= explicitly. All four providers above have a free tier."
                )
    elif not (base_url and model):
        raise ValueError("provider='custom' requires both base_url and model.")

    schema = NARRATIVE_SCHEMAS[module]
    system_prompt = build_system_prompt(company, schema)
    compact_results = _compact_results_for_prompt(results)
    user_content = json.dumps({
        "few_shot_example": few_shot_example,
        "computed_data": compact_results,
    }, separators=(",", ":"), default=str)

    last_problems: list[str] = []
    fallback_api: str | None = None
    fallback_provider: str | None = None
    if provider == "groq":
        fallback_api = os.environ.get("GEMINI_API_KEY")
        fallback_provider = "gemini"

    for attempt in range(max_retries + 1):
        if attempt > 0:
            user_content += (
                "\n\nYour previous response was rejected for these reasons: "
                f"{last_problems}. Return ONLY the corrected JSON object, nothing else."
            )
        try:
            raw = _call_openai_compatible(system_prompt, user_content, base_url, model, api_key)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if provider == "groq" and fallback_api and status in {401, 403, 429}:
                print(f"[fallback] {module}: Groq rejected request (HTTP {status}); retrying via Gemini.")
                raw = _call_openai_compatible(
                    system_prompt,
                    user_content,
                    PROVIDER_PRESETS["gemini"][0],
                    PROVIDER_PRESETS["gemini"][1],
                    fallback_api,
                )
            else:
                raise
        parsed = _safe_json_parse(raw)
        if parsed is None:
            last_problems = ["response was not valid JSON"]
            continue
        ok, problems = _validate_schema(parsed, schema)
        if not ok:
            last_problems = problems
            continue
        fidelity_issues = _check_numeric_fidelity(parsed, results)
        if fidelity_issues:
            parsed.setdefault("data_quality_flags", [])
            parsed["data_quality_flags"].extend(fidelity_issues)
        return parsed

    raise NarrativeError(
        f"'{provider}'/{model} did not return valid JSON matching the '{module}' schema after "
        f"{max_retries + 1} attempt(s). Last problems: {last_problems}"
    )
