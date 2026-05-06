"""
POC: English → Marathi (and styles) via Azure OpenAI, aligned with the prior FastAPI pattern.
- Custom .env load (only fills missing os.environ keys)
- OUTPUT_STYLE ``minglish`` · TASK_MODE ``auto`` (fixed constants in ``app.py``; no UI)
- Flask: SPA ``templates/index.html`` calls REST ``/api/v1/translate`` (typing); legacy ``POST /translate`` aliases.
- Optional: python app.py --file  # reads input.txt → output.txt
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import OrderedDict
from typing import Any

from azure.identity import ClientSecretCredential, get_bearer_token_provider
from flask import Blueprint, Flask, jsonify, render_template, request
from flask_cors import CORS
from openai import AzureOpenAI, OpenAIError

_log = logging.getLogger(__name__)
_root = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(_root, ".env")
INPUT_FILE = os.path.join(_root, "input.txt")
OUTPUT_FILE = os.path.join(_root, "output.txt")


def load_env_file(file_path: str, override: bool = False) -> None:
    """Load .env into os.environ. If override is False, skip keys that are already set."""
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value


# Parent .env (shared), then this folder’s .env — local always wins (fixes empty parent keys).
load_env_file(os.path.join(_root, "..", ".env"), override=False)
load_env_file(ENV_FILE, override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
).strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
OUTPUT_STYLE = "minglish"
TASK_MODE = "auto"


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _expose_upstream_detail() -> bool:
    """Include Azure/OpenAI failure text in API JSON (avoid in production-facing hosts)."""
    return _env_truthy("AZURE_OPENAI_VERBOSE_ERRORS") or (
        os.environ.get("FLASK_DEBUG") == "1"
    )


def _upstream_api_message(exc: BaseException | None, generic: str) -> str:
    if _expose_upstream_detail() and exc:
        return str(exc)
    return generic


# Optional: public base URL of this API (no trailing slash). Set when the HTML is opened
# from another domain (e.g. PHP on cPanel) so fetches target the Python host.
API_PUBLIC_BASE = (os.getenv("API_PUBLIC_BASE") or "").strip().rstrip("/")
# Lower = faster, more deterministic completions (ignored if temperature is omitted for the model).
_COMP_TEMP = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.2") or "0.2")


def _include_temperature_in_request(dep_lower: str) -> bool:
    """Several Azure GPT-5 / reasoning SKUs reject custom ``temperature`` (only default applies)."""
    if os.getenv("AZURE_OPENAI_OMIT_TEMPERATURE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.getenv("AZURE_OPENAI_FORCE_TEMPERATURE", "").strip().lower() in ("1", "true", "yes"):
        return True
    return not _deployment_is_reasoning(dep_lower)


def _attach_temperature_maybe(kwargs: dict[str, Any], dep_lower: str, value: float) -> None:
    if _include_temperature_in_request(dep_lower):
        kwargs["temperature"] = value


def _completion_budget_translate(text: str, max_cap: int = 1536, floor: int = 80) -> int:
    """Tight bound: transliterated output length usually tracks input; avoids huge-generation latency."""
    n = len(text)
    # ~1–1.5× input length worth of tokens is enough for sentence-level Marathi script.
    estimate = floor + max(n, n * 6 // 5)
    return min(max_cap, max(floor, estimate))


def _completion_budget_suggest() -> int:
    return int(os.getenv("SUGGEST_MAX_COMPLETION_TOKENS", "200") or "200")


def _deployment_is_reasoning(deployment_name: str) -> bool:
    """Reasoning models count hidden reasoning toward the same completion budget as assistant text."""
    n = (deployment_name or "").lower()
    return "gpt-5" in n or "o1" in n or "o3" in n


def _reasoning_effort_param(deployment_name: str) -> str | None:
    if not _deployment_is_reasoning(deployment_name):
        return None
    raw = (os.getenv("AZURE_OPENAI_REASONING_EFFORT", "off") or "").strip()
    if raw.lower() in ("", "off", "false", "none"):
        return None
    if raw in ("minimal", "low", "medium", "high"):
        return raw
    return None


def _translate_completion_budget(text: str, deployment_name: str) -> int:
    base = _completion_budget_translate(text)
    if not _deployment_is_reasoning(deployment_name):
        return base
    floor = int(os.getenv("TRANSLATE_MIN_COMPLETION_TOKENS", "512") or "512")
    # For short word/phrase live-preview inputs, keep budget tight for lower latency.
    if len((text or "").strip()) <= 24:
        floor = min(floor, 256)
    return max(base, floor)


def _suggest_completion_cap(dep: str) -> int:
    base = max(256, _completion_budget_suggest())
    if not _deployment_is_reasoning(dep):
        return base
    floor = int(os.getenv("SUGGEST_MIN_COMPLETION_TOKENS", "256") or "256")
    return max(base, floor)


def _extract_assistant_text(response: Any) -> str:
    if not getattr(response, "choices", None):
        return ""
    msg = response.choices[0].message
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c.strip()
    return ""


def _chat_completions_create_safe(client: AzureOpenAI, kwargs: dict[str, Any]) -> Any:
    """Some Azure deployments reject extra params — retry slimmer payloads."""
    try:
        return client.chat.completions.create(**kwargs)
    except OpenAIError as e:
        err = str(e).lower()
        if any(p in err for p in ("reasoning_effort", "verbosity")) and any(
            p in err
            for p in ("unrecognized", "unknown parameter", "invalid", "unsupported parameter")
        ):
            slim = {
                k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "verbosity")
            }
            return client.chat.completions.create(**slim)
        if "temperature" in err and any(p in err for p in ("unsupported", "unsupported_value", "does not support")):
            slim = {k: v for k, v in kwargs.items() if k != "temperature"}
            return client.chat.completions.create(**slim)
        raise


app = Flask(__name__)
# Allow browser calls from your main site (e.g. technobiz.online) to this API on another host.
CORS(app, origins="*", allow_methods=["*"], allow_headers=["*"])

_azure_client: Any | None = None

STYLE_CHOICES = ("marathi", "marlish", "minglish")
MODE_CHOICES = ("translate", "transliterate", "auto")

_TRANSLATE_CACHE_MAX = int(os.getenv("TRANSLATE_CACHE_MAX", "400") or "400")
_translate_response_cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()


def _translate_cache_get(
    deployment: str, text: str, output_style: str, task_mode: str
) -> str | None:
    key = (
        (deployment or ""),
        text,
        output_style,
        task_mode,
    )
    if key not in _translate_response_cache:
        return None
    _translate_response_cache.move_to_end(key)
    return _translate_response_cache[key]


def _translate_cache_put(
    deployment: str, text: str, output_style: str, task_mode: str, result: str
) -> None:
    if not (result and result.strip()):
        return
    key = (
        (deployment or ""),
        text,
        output_style,
        task_mode,
    )
    _translate_response_cache[key] = result.strip()
    _translate_response_cache.move_to_end(key)
    while len(_translate_response_cache) > max(16, _TRANSLATE_CACHE_MAX):
        _translate_response_cache.popitem(last=False)


def _attach_default_verbosity_maybe(kwargs: dict[str, Any], deployment_name: str) -> None:
    """Default ``verbosity: low`` on reasoning deployments (faster replies) unless overridden."""
    ve_raw = os.getenv("AZURE_OPENAI_VERBOSITY")
    if ve_raw is not None and str(ve_raw).strip() != "":
        v = ve_raw.strip().lower()
        if v in ("low", "medium", "high"):
            kwargs["verbosity"] = v
        return
    if os.getenv("AZURE_OPENAI_DISABLE_DEFAULT_VERBOSITY", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    if _deployment_is_reasoning(deployment_name):
        kwargs["verbosity"] = "low"


def get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is not None:
        return _azure_client

    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT in .env")
    if not AZURE_OPENAI_DEPLOYMENT:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT in .env")

    ep = AZURE_OPENAI_ENDPOINT.rstrip("/")
    if AZURE_OPENAI_API_KEY:
        _azure_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=ep,
        )
        return _azure_client

    tenant = (os.environ.get("AZURE_TENANT_ID") or "").strip()
    client_id = (os.environ.get("AZURE_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("AZURE_CLIENT_SECRET") or "").strip()
    if not (tenant and client_id and client_secret):
        raise ValueError(
            "Missing AZURE_OPENAI_API_KEY in .env, or set AZURE_TENANT_ID, "
            "AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET for Entra ID."
        )
    cred = ClientSecretCredential(tenant, client_id, client_secret)
    token = get_bearer_token_provider(
        cred, "https://cognitiveservices.azure.com/.default"
    )
    _azure_client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=ep,
        azure_ad_token_provider=token,
    )
    return _azure_client


def convert_text(
    text: str,
    output_style: str | None = None,
    task_mode: str | None = None,
) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    style_to_instruction = {
        "marathi": "Return output in Marathi script (Devanagari).",
        "marlish": "Return output in Roman Marathi (Marlish). Use English letters only.",
        "minglish": (
            "Return output in Marathi-English mixed style (Minglish), "
            "using natural colloquial Roman script."
        ),
    }
    style_input = (output_style or OUTPUT_STYLE or "marathi").strip().lower()
    selected_style = style_input if style_input in style_to_instruction else "marathi"
    mode_instruction = {
        "translate": "Translate meaning from English to Marathi naturally.",
        "transliterate": (
            "Transliterate phonetically from Roman English letters to Marathi script. "
            "Do not change meaning; convert sound only."
        ),
        "auto": (
            "If input is Romanized Marathi words (for example tebal, by, namaskar), "
            "transliterate to Marathi script; otherwise translate meaning."
        ),
    }
    mode_input = (task_mode or TASK_MODE or "transliterate").strip().lower()
    selected_mode = mode_input if mode_input in mode_instruction else "transliterate"

    dep_key = AZURE_OPENAI_DEPLOYMENT or ""
    cached_reply = _translate_cache_get(dep_key, t, selected_style, selected_mode)
    if cached_reply is not None:
        return cached_reply

    system_prompt = (
        "Reply with Marathi result only — no preamble, quotes, labels, or English gloss.\n"
        f"{style_to_instruction[selected_style]}\n"
        f"{mode_instruction[selected_mode]}"
    )
    user_content = t

    client = get_azure_client()
    dep_name = AZURE_OPENAI_DEPLOYMENT or ""
    dep = dep_name.lower()
    toks = _translate_completion_budget(t, dep_name)
    max_cap = int(os.getenv("TRANSLATE_ABSOLUTE_COMPLETION_CAP", "32000") or "32000")

    def _one_translate_call(completion_tokens: int) -> str:
        create_kwargs_inner: dict[str, Any] = {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        capped = max(1, min(int(completion_tokens), max_cap))
        if "gpt-5" in dep or "o3" in dep or "o1" in dep:
            create_kwargs_inner["max_completion_tokens"] = capped
        else:
            create_kwargs_inner["max_tokens"] = capped
        _attach_temperature_maybe(create_kwargs_inner, dep, _COMP_TEMP)
        re_eff = _reasoning_effort_param(dep_name)
        if re_eff is not None:
            create_kwargs_inner["reasoning_effort"] = re_eff
        _attach_default_verbosity_maybe(create_kwargs_inner, dep_name)
        response_inner = _chat_completions_create_safe(client, create_kwargs_inner)
        return _extract_assistant_text(response_inner)

    try:
        txt = _one_translate_call(toks)
        if txt:
            _translate_cache_put(dep_key, t, selected_style, selected_mode, txt)
            return txt
        if _deployment_is_reasoning(dep_name):
            txt = _one_translate_call(min(max_cap, max(toks * 2, 16384)))
        if txt:
            _translate_cache_put(dep_key, t, selected_style, selected_mode, txt)
        return txt
    except OpenAIError as e:
        raise RuntimeError(
            f"Azure OpenAI error: {e}. Check endpoint, key, deployment, and quotas."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI error: {e}") from e


def translate_to_marathi(text: str) -> str:
    return convert_text(text, output_style=OUTPUT_STYLE, task_mode=TASK_MODE)


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse model output into a list of suggestion strings; tolerate markdown and loose JSON."""
    s = (raw or "").strip()
    if not s:
        return []

    # Strip ```json ... ``` or ``` ... ``` wrappers
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    def try_load_array(text: str) -> list[str] | None:
        t = text.strip()
        if not t:
            return None
        try:
            data = json.loads(t)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()][:12]
        except json.JSONDecodeError:
            pass
        return None

    out = try_load_array(s)
    if out:
        return out

    # Balanced bracket slice: first '[' to matching ']' (works when extra prose exists)
    start = s.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            c = s[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    chunk = s[start : i + 1]
                    out = try_load_array(chunk)
                    if out:
                        return out
                    break

    # Regex array (non-greedy can truncate; try greedy last)
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        out = try_load_array(m.group(0))
        if out:
            return out

    lines = [
        ln.rstrip()
        for ln in s.splitlines()
        if ln.strip()
    ]
    collected: list[str] = []
    # Line fallback: numbered or plain lines without JSON
    for ln in lines:
        ln = re.sub(r"^\d+[\).\s\-]+", "", ln).strip().strip(',').strip()
        if not ln.startswith(("[", '"')):
            cleaned = ln.lstrip("*•- ").strip("\"' ")
        else:
            cleaned = ln.strip().strip(',').strip('[]" ')
        cleaned = cleaned.replace("```", "").strip()
        if cleaned and cleaned not in ("```", "---", "...") and len(cleaned) < 140:
            collected.append(cleaned)
    # Dedupe preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for x in collected:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:12]


def _suggest_plain_line_retry(
    client: AzureOpenAI,
    dep: str,
    frag: str,
    style_hint: str,
    mode_hint: str,
) -> list[str]:
    """Second call when JSON parsing fails — one alternative per line."""
    dep_orig = AZURE_OPENAI_DEPLOYMENT or ""
    cap = max(_suggest_completion_cap(dep), 512)
    sys_text = (
        "The user typed a Roman fragment. Output EXACTLY 5 non-empty lines.\n"
        "Each line shows ONE short alternative rendering only. No numbers, bullets, "
        "quotes, JSON, keywords, or markdown — only the text of each alternative.\n"
        f"{style_hint}\n"
        f"{mode_hint}"
    )
    create_kwargs: dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": frag},
        ],
    }
    if "gpt-5" in dep or "o3" in dep or "o1" in dep:
        create_kwargs["max_completion_tokens"] = cap
    else:
        create_kwargs["max_tokens"] = cap
    _attach_temperature_maybe(create_kwargs, dep, min(0.55, _COMP_TEMP + 0.15))
    re_eff = _reasoning_effort_param(dep_orig)
    if re_eff is not None:
        create_kwargs["reasoning_effort"] = re_eff
    _attach_default_verbosity_maybe(create_kwargs, dep_orig)
    try:
        response = _chat_completions_create_safe(client, create_kwargs)
    except OpenAIError as e:
        raise RuntimeError(f"Azure OpenAI error: {e}") from e
    raw = _extract_assistant_text(response)
    out: list[str] = []
    for ln in raw.splitlines():
        t = re.sub(r"^\s*[\d]+\s*[\).\-\s]+\s*", "", ln).strip()
        t = t.strip("•-*'\"`\t []")
        if not t or len(t) > 180:
            continue
        if len(out) >= 8:
            break
        out.append(t)
    return out[:8]


def last_roman_fragment(full: str) -> str:
    """Last whitespace-delimited fragment the user might still be typing."""
    parts = full.rstrip().split()
    if not parts:
        return ""
    return parts[-1].strip()


def _suggest_transliteration_alternatives_inner(
    fragment: str,
    output_style: str,
    task_mode: str,
) -> tuple[str, ...]:
    """One or two Azure calls; never cache here (see suggest_transliteration_alternatives)."""
    frag = (fragment or "").strip()
    if len(frag) < 2:
        return ()

    style_to_instruction = {
        "marathi": "Suggestions must be Marathi/Devanagari only.",
        "marlish": "Suggestions must use Roman letters only.",
        "minglish": (
            "Suggestions must be Minglish Roman (natural mixed colloquial), short phrases."
        ),
    }
    selected_style = output_style if output_style in style_to_instruction else "marathi"
    mode_instruction = {
        "translate": "Interpret as English and give plausible Marathi word renderings.",
        "transliterate": "Interpret as Roman phonetic input; give plausible spellings/spaces.",
        "auto": (
            "If it looks Romanized Marathi give script variants; if English proper noun, "
            "give culturally common variants."
        ),
    }
    selected_mode = task_mode if task_mode in mode_instruction else "transliterate"

    sys_prompt = (
        "Respond with VALID JSON ONLY: one JSON array containing exactly 5 SHORT strings "
        '(UTF-8; example ["राम","रॅम्"]). Nothing before or after the array. '
        "No markdown, no triple backticks, no explanation.\n"
        f"{style_to_instruction[selected_style]}\n"
        f"{mode_instruction[selected_mode]}\n"
        "Duplicates not allowed."
    )
    client = get_azure_client()
    dep = (AZURE_OPENAI_DEPLOYMENT or "").lower()
    dep_original = AZURE_OPENAI_DEPLOYMENT or ""
    cap = _suggest_completion_cap(dep)
    create_kwargs: dict[str, Any] = {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": frag},
        ],
    }
    if "gpt-5" in dep or "o3" in dep or "o1" in dep:
        create_kwargs["max_completion_tokens"] = cap
    else:
        create_kwargs["max_tokens"] = cap
    _attach_temperature_maybe(create_kwargs, dep, min(0.5, _COMP_TEMP + 0.1))
    re_eff = _reasoning_effort_param(dep_original)
    if re_eff is not None:
        create_kwargs["reasoning_effort"] = re_eff
    _attach_default_verbosity_maybe(create_kwargs, dep_original)
    try:
        response = _chat_completions_create_safe(client, create_kwargs)
    except OpenAIError as e:
        raise RuntimeError(f"Azure OpenAI error: {e}") from e
    raw = _extract_assistant_text(response)
    items = _parse_json_string_list(raw)
    if not items:
        items = _suggest_plain_line_retry(
            client,
            dep,
            frag,
            style_to_instruction[selected_style],
            mode_instruction[selected_mode],
        )
    return tuple(items[:8])


_SUGGEST_OK_CACHE: dict[tuple[str, str, str], tuple[str, ...]] = {}

def suggest_transliteration_alternatives(
    fragment: str,
    output_style: str | None,
    task_mode: str | None,
) -> list[str]:
    o = (output_style or OUTPUT_STYLE or "marathi").strip().lower()
    m = (task_mode or TASK_MODE or "transliterate").strip().lower()
    if o not in STYLE_CHOICES:
        o = "marathi"
    if m not in MODE_CHOICES:
        m = "transliterate"
    key = (fragment.strip().lower(), o, m)
    cached = _SUGGEST_OK_CACHE.get(key)
    if cached is not None:
        return list(cached)
    tup = _suggest_transliteration_alternatives_inner(fragment.strip(), o, m)
    if tup:
        if len(_SUGGEST_OK_CACHE) > 400:
            _SUGGEST_OK_CACHE.clear()
        _SUGGEST_OK_CACHE[key] = tup
    return list(tup)


# --- file pipeline (your prior main()) ---


def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        print("Error reading file:", e)
        return ""


def write_file(file_path: str, content: str) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Output saved to {file_path}")
    except OSError as e:
        print("Error writing file:", e)


def run_file_pipeline() -> None:
    print("Translation agent (file mode)")
    print("Output style:", OUTPUT_STYLE, "| Task mode:", TASK_MODE)
    input_text = read_file(INPUT_FILE)
    if not input_text:
        print("No input text found in", INPUT_FILE)
        return
    print("Translating…")
    try:
        out = translate_to_marathi(input_text)
    except Exception as e:
        print("Error:", e)
        return
    write_file(OUTPUT_FILE, out)
    print("Done.")


# --- Flask ---


def _template_ctx(**kwargs: Any) -> dict[str, Any]:
    return {
        "api_base": API_PUBLIC_BASE,
        "api_prefix": "/api/v1",
        **kwargs,
    }


# --- JSON API (index.html and other clients) ---


def _api_translate_json() -> tuple[Any, int] | Any:
    """POST JSON: ``{"text": "..."}`` → ``text``, ``task_mode``, ``output_style`` (both fixed server-side)."""
    data = request.get_json(silent=True) or {}
    raw = data.get("text")
    if raw is None:
        return jsonify({"error": "bad_request", "message": 'JSON body must include "text".'}), 400
    if not isinstance(raw, str):
        return jsonify({"error": "bad_request", "message": '"text" must be a string.'}), 400
    text = raw.strip()
    if not text:
        return jsonify(
            {"error": "bad_request", "message": '"text" must be non-empty after trimming.'}
        ), 400
    o_style = OUTPUT_STYLE
    t_mode = TASK_MODE
    try:
        result = convert_text(text, output_style=o_style, task_mode=t_mode)
    except ValueError as e:
        return jsonify({"error": "configuration_error", "message": str(e)}), 503
    except RuntimeError as e:
        _log.warning("Upstream translate failure: %s", e, exc_info=True)
        return jsonify(
            {
                "error": "upstream_error",
                "message": _upstream_api_message(
                    e, "Azure OpenAI request failed."
                ),
            }
        ), 502
    if not result:
        return jsonify(
            {"error": "empty_content", "message": "Model returned empty content."}
        ), 502
    return jsonify(text=result, task_mode=t_mode, output_style=o_style)


def _api_suggest_json() -> tuple[Any, int] | Any:
    """POST JSON: ``{"text"}`` and/or ``{"fragment"}`` (output style / task mode fixed server-side)."""
    data = request.get_json(silent=True) or {}
    full_raw = data.get("text")
    fragment_in = data.get("fragment")

    if fragment_in is not None:
        if not isinstance(fragment_in, str):
            return jsonify({"error": "bad_request", "message": '"fragment" must be a string.'}), 400
        fragment = fragment_in.strip()
    elif full_raw is not None:
        if not isinstance(full_raw, str):
            return jsonify(
                {"error": "bad_request", "message": '"text" must be a string.'}
            ), 400
        fragment = last_roman_fragment(full_raw)
    else:
        return jsonify(
            {
                "error": "bad_request",
                "message": 'JSON body must include "text" and/or "fragment".',
            }
        ), 400

    if len(fragment) < 2:
        return jsonify(
            fragment=fragment,
            suggestions=[],
            task_mode=TASK_MODE,
            output_style=OUTPUT_STYLE,
        )

    o_style = OUTPUT_STYLE
    t_mode = TASK_MODE
    try:
        items = suggest_transliteration_alternatives(
            fragment, output_style=o_style, task_mode=t_mode
        )
    except ValueError as e:
        return jsonify({"error": "configuration_error", "message": str(e)}), 503
    except RuntimeError as e:
        _log.warning("Upstream suggest failure: %s", e, exc_info=True)
        return jsonify(
            {
                "error": "upstream_error",
                "message": _upstream_api_message(
                    e, "Suggestion request failed."
                ),
            }
        ), 502
    return jsonify(
        fragment=fragment,
        suggestions=items,
        task_mode=t_mode,
        output_style=o_style,
    )


api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")


@api_v1.post("/translate")
def api_v1_translate():
    return _api_translate_json()


@api_v1.post("/suggest")
def api_v1_suggest():
    return _api_suggest_json()


@api_v1.get("/health")
def api_v1_health():
    return jsonify({"status": "ok"})


@api_v1.get("/status")
def api_v1_status():
    return jsonify(
        has_endpoint=bool(AZURE_OPENAI_ENDPOINT),
        has_api_key=bool(AZURE_OPENAI_API_KEY),
        has_deployment=bool(AZURE_OPENAI_DEPLOYMENT),
        api_version=AZURE_OPENAI_API_VERSION,
        output_style=OUTPUT_STYLE,
        task_mode=TASK_MODE,
    )


# Backward compatibility: same-origin ``/translate`` and ``/suggest`` aliases
@app.post("/translate")
def translate_api_legacy():
    return _api_translate_json()


@app.post("/suggest")
def suggest_api_legacy():
    return _api_suggest_json()


@app.get("/")
def index():
    return render_template(
        "index.html",
        **_template_ctx(
            english="",
            output="",
            error=None,
        ),
    )


@app.post("/")
def index_post():
    act = (request.form.get("action") or "translate").strip()
    if act == "clear":
        return render_template(
            "index.html",
            **_template_ctx(
                english="",
                output="",
                error=None,
            ),
        )

    english = (request.form.get("english") or "").strip()
    err: str | None = None
    out_text = ""
    if english:
        try:
            out_text = convert_text(english, output_style=OUTPUT_STYLE, task_mode=TASK_MODE)
        except (RuntimeError, ValueError) as e:
            err = str(e)
    return render_template(
        "index.html",
        **_template_ctx(
            english=english,
            output=out_text,
            error=err,
        ),
    )


app.register_blueprint(api_v1)


@app.get("/health")
def health_check():
    """Same response as ``GET /api/v1/health`` (root path kept for probes)."""
    return api_v1_health()


@app.get("/api/status")
def api_status_compat():
    """Same as ``GET /api/v1/status`` (legacy path)."""
    return api_v1_status()


def main() -> None:
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    if "--file" in sys.argv:
        run_file_pipeline()
    else:
        main()
