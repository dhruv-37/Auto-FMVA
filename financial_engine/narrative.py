"""Step 3 (where Claude enters) + Step 4 (prompt pattern for narrative modules). Bucket B only —
this module never computes financial numbers itself. It takes an already-computed `results` dict
(built from Bucket A tables: financials/ratios/market_data/dcf/monte_carlo/wacc) and asks Claude to
turn it into judgment/prose, under a strict contract: never invent or alter numbers, flag anomalies
instead of narrating around them, match a provided few-shot example's tone/structure, and return
only JSON matching the module's schema.
"""
import json
import re
import numbers
from typing import Any

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Step 3 — the four narrative modules Claude is allowed to own, each with its
# own output contract. `schema` is a lightweight type spec (not full JSON
# Schema) checked by `_validate_schema` below: "str", "number", "bool",
# "list[str]", "list[dict]", or a nested dict for object fields.
# ---------------------------------------------------------------------------
NARRATIVE_SCHEMAS: dict[str, dict] = {
    "company_profile": {
        "title": "str",
        "body": "str",
        "recent_updates": "list[str]",
        "data_quality_flags": "list[str]",
    },
    "interpretive_commentary": {
        "title": "str",
        "body": "str",
        "data_quality_flags": "list[str]",
    },
    "anomaly_flags": {
        "flags": "list[dict]",          # each: {table, column, year, value, issue, severity}
        "data_quality_flags": "list[str]",
    },
    "football_field": {
        "title": "str",
        "body": "str",
        "data_quality_flags": "list[str]",
    },
}

_ALLOWED_SEVERITIES = {"low", "medium", "high"}


class NarrativeError(RuntimeError):
    """Raised when Claude's output can't be trusted as-is (bad JSON, wrong schema, or exhausted retries)."""


# ---------------------------------------------------------------------------
# Step 4 — prompt pattern
# ---------------------------------------------------------------------------
def build_system_prompt(company: str, schema: dict) -> str:
    """Exact rules from Step 4: never invent/alter numbers, flag anomalies instead of narrating
    around them, match the few-shot example's tone/structure, output only JSON matching schema."""
    return (
        "You are a financial analyst writing sections of an equity research report.\n"
        f"You will be given a JSON object of already-computed financial data for {company}.\n"
        "Rules:\n"
        "- Never invent or alter any number in the JSON. Use only what's given.\n"
        "- If a number looks anomalous (impossible %, sign error), flag it in a "
        '"data_quality_flags" field instead of narrating around it.\n'
        "- Match the tone and structure of the example section provided.\n"
        f"- Output only valid JSON matching the schema: {json.dumps(schema)}"
    )


def _call_claude(system_prompt: str, user_content: str, api_key: str, model: str,
                  max_tokens: int = 2000) -> str:
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    if not text:
        raise NarrativeError(f"Claude returned no text content: {data}")
    return text


def _safe_json_parse(raw: str):
    """Tries, in order: the whole string as-is; a fenced ```json ... ``` block found anywhere in
    the text; the substring from the first '{' to the last '}'. The API path (JSON-only system
    prompt) almost always hits the first case; the web-paste path — where Claude wraps the JSON
    in chatter like "Sure, here's the JSON:\n\n```json\n{...}\n```\n\nLet me know if..." — needs
    the other two."""
    candidates = [raw.strip()]

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    candidates.extend(fenced)

    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(raw[first:last + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _type_ok(value: Any, spec: str) -> bool:
    if spec == "str":
        return isinstance(value, str)
    if spec == "number":
        return isinstance(value, numbers.Number) and not isinstance(value, bool)
    if spec == "bool":
        return isinstance(value, bool)
    if spec == "list[str]":
        return isinstance(value, list) and all(isinstance(v, str) for v in value)
    if spec == "list[dict]":
        return isinstance(value, list) and all(isinstance(v, dict) for v in value)
    raise ValueError(f"Unknown schema spec: {spec}")


def _validate_schema(parsed: Any, schema: dict) -> tuple[bool, list[str]]:
    problems = []
    if not isinstance(parsed, dict):
        return False, ["top-level output is not a JSON object"]
    for key, spec in schema.items():
        if key not in parsed:
            problems.append(f"missing required key '{key}'")
            continue
        if not _type_ok(parsed[key], spec):
            problems.append(f"key '{key}' should be {spec}, got {type(parsed[key]).__name__}")
    extra = set(parsed.keys()) - set(schema.keys())
    if extra:
        problems.append(f"unexpected extra keys: {sorted(extra)}")
    if "flags" in parsed and isinstance(parsed["flags"], list):
        for i, f in enumerate(parsed["flags"]):
            if not isinstance(f, dict):
                continue
            sev = f.get("severity")
            if sev is not None and sev not in _ALLOWED_SEVERITIES:
                problems.append(f"flags[{i}].severity '{sev}' not in {_ALLOWED_SEVERITIES}")
    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------
# Numeric-fidelity check: the doc's hardest-to-enforce rule is "never invent or
# alter any number." We can't prove a negative, but we can catch the common
# failure mode — Claude quoting a number in prose that doesn't trace back to
# anything in the computed payload. Best-effort, not a guarantee: it flags
# suspicious numbers into data_quality_flags rather than hard-failing, since
# some numbers in fluent prose (e.g. "grew for 3 straight years") are counts,
# not data values.
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.\d+|-?\d[\d,]*%")


def _flatten_numbers(obj: Any, acc: set) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_numbers(v, acc)
    elif isinstance(obj, numbers.Number) and not isinstance(obj, bool):
        acc.add(round(float(obj), 1))
        acc.add(round(float(obj) * 100, 1))   # covers ratio (0.35) quoted as percent (35%)


def _check_numeric_fidelity(parsed: dict, results: dict) -> list[str]:
    known = set()
    _flatten_numbers(results, known)
    issues = []
    for key in ("title", "body"):
        text = parsed.get(key)
        if not isinstance(text, str):
            continue
        for match in _NUMBER_RE.findall(text):
            cleaned = match.replace(",", "").replace("%", "")
            try:
                val = round(float(cleaned), 1)
            except ValueError:
                continue
            if not any(abs(val - k) <= max(0.1, abs(k) * 0.01) for k in known):
                issues.append(f"'{key}' mentions {match!r}, which doesn't match any computed value")
    return issues


# ---------------------------------------------------------------------------
# Public entry point — API path
# ---------------------------------------------------------------------------
def generate_narrative(module: str, company: str, results: dict, few_shot_example: dict,
                        api_key: str, model: str = DEFAULT_MODEL, max_retries: int = 1) -> dict:
    """Runs one Step-3 narrative module through the Step-4 prompt pattern.

    `results` — already-computed Bucket A output (numbers only, from financial_engine).
    `few_shot_example` — an existing report's section in this module's schema shape, so Claude
    matches house style (per Step 4: "pass a few-shot example ... plus the new company's computed
    values ... ask for structured JSON out").
    Raises NarrativeError if Claude never returns valid, schema-conforming JSON.
    """
    if module not in NARRATIVE_SCHEMAS:
        raise ValueError(f"Unknown narrative module '{module}'. Choose from {list(NARRATIVE_SCHEMAS)}")
    schema = NARRATIVE_SCHEMAS[module]
    system_prompt = build_system_prompt(company, schema)

    user_content = json.dumps({
        "few_shot_example": few_shot_example,
        "computed_data": results,
    }, indent=2, default=str)

    last_problems: list[str] = []
    for attempt in range(max_retries + 1):
        if attempt > 0:
            user_content += (
                "\n\nYour previous response was rejected for these reasons: "
                f"{last_problems}. Return ONLY the corrected JSON object, nothing else."
            )
        raw = _call_claude(system_prompt, user_content, api_key=api_key, model=model)
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
        f"Claude did not return valid JSON matching the '{module}' schema after "
        f"{max_retries + 1} attempt(s). Last problems: {last_problems}"
    )


# ---------------------------------------------------------------------------
# Public entry point — Claude-web (manual copy/paste) path, no API cost.
# Same contract, same validation, no _call_claude involved.
# ---------------------------------------------------------------------------
def build_web_prompt(module: str, company: str, results: dict, few_shot_example: dict) -> str:
    """Returns one block of text to paste into a Claude web chat: the system prompt (Step 4)
    followed by the few-shot example + computed data, formatted the same way generate_narrative()
    would've sent it over the API. Copy Claude's reply and pass it to paste_narrative()."""
    if module not in NARRATIVE_SCHEMAS:
        raise ValueError(f"Unknown narrative module '{module}'. Choose from {list(NARRATIVE_SCHEMAS)}")
    schema = NARRATIVE_SCHEMAS[module]
    system_prompt = build_system_prompt(company, schema)
    user_content = json.dumps({
        "few_shot_example": few_shot_example,
        "computed_data": results,
    }, indent=2, default=str)
    return f"{system_prompt}\n\n{user_content}"


def paste_narrative(module: str, results: dict, raw_text: str, out_path: str = None) -> dict:
    """Validates text pasted back from a Claude web reply exactly as generate_narrative() would
    validate an API response — same JSON parse, same schema check, same numeric-fidelity check
    against `results` — just with no HTTP call and no retry (there's no automated retry loop when
    a human is doing the copy/paste; if this raises, fix it in the web chat and paste again).

    Raises NarrativeError on bad JSON or a schema violation — nothing gets silently accepted.
    On success, returns the parsed dict and, if `out_path` is given, writes it there as JSON so
    the report-assembly step (Step 5) can read it exactly like an API-generated narrative file."""
    if module not in NARRATIVE_SCHEMAS:
        raise ValueError(f"Unknown narrative module '{module}'. Choose from {list(NARRATIVE_SCHEMAS)}")
    schema = NARRATIVE_SCHEMAS[module]

    parsed = _safe_json_parse(raw_text)
    if parsed is None:
        raise NarrativeError(
            "Pasted text is not valid JSON. Make sure you copied only the JSON object Claude "
            "returned (no surrounding commentary), then try again."
        )
    ok, problems = _validate_schema(parsed, schema)
    if not ok:
        raise NarrativeError(f"Pasted JSON doesn't match the '{module}' schema: {problems}")

    fidelity_issues = _check_numeric_fidelity(parsed, results)
    if fidelity_issues:
        parsed.setdefault("data_quality_flags", [])
        parsed["data_quality_flags"].extend(fidelity_issues)

    if out_path:
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=2)

    return parsed


# ---------------------------------------------------------------------------
# Auxiliary Bucket-A-side pre-check. Not a replacement for Claude's own
# "anomaly_flags" module (Step 3) — this is a cheap, deterministic sweep over
# the ratios table so Claude is fed likely candidates rather than having to
# spot every impossible number unaided (e.g. the -100% gross-margin case the
# doc calls out by name).
# ---------------------------------------------------------------------------
def flag_engine_anomalies(ratios_records: list[dict]) -> list[dict]:
    """`ratios_records` = ratios.csv/DataFrame rows as dicts. Returns candidate anomaly flags in
    the same shape as the `anomaly_flags` module's `flags` field, for Claude to confirm/extend."""
    candidates = []
    margin_cols = ("gross_margin", "ebitda_margin", "ebit_margin", "ebt_margin", "net_profit_margin")
    for row in ratios_records:
        year = row.get("year")
        for col in margin_cols:
            val = row.get(col)
            if isinstance(val, numbers.Number) and (val < -1 or val > 1):
                candidates.append({
                    "table": "ratios", "column": col, "year": year, "value": val,
                    "issue": f"{col} of {val:.1%} is outside a plausible [-100%, 100%] range",
                    "severity": "high",
                })
        roe = row.get("roe")
        if isinstance(roe, numbers.Number) and abs(roe) > 2:
            candidates.append({
                "table": "ratios", "column": "roe", "year": year, "value": roe,
                "issue": f"ROE of {roe:.1%} is extreme — check for a near-zero or negative equity denominator",
                "severity": "medium",
            })
    return candidates