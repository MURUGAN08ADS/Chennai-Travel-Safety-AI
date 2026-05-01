import json
import logging
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency in OpenRouter mode
    Groq = None

LOGGER = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "chennai_zones.json"
ENV_FILE_CANDIDATES = (
    PROJECT_ROOT / ".env",
    BACKEND_DIR / ".env",
)
PREFERRED_ENV_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OPENROUTER_APP_URL",
    "OPENROUTER_APP_NAME",
    "GROQ_API_KEY",
    "GROQ_MODEL",
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_QUESTION_CHARS = 1500
MAX_CONTEXT_CHARS = 3000
EMERGENCY_PREFIX = "If you are in immediate danger, call emergency services now (India: 112)."
SUPPORTED_LANGUAGES = {
    "english": "English",
    "tamil": "Tamil",
    "hinglish": "Hinglish",
}
LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",
    "tamil": "tamil",
    "ta": "tamil",
    "tam": "tamil",
    "தமிழ்": "tamil",
    "hinglish": "hinglish",
    "hindi-english": "hinglish",
    "hi-en": "hinglish",
}
DISTRESS_KEYWORDS = (
        "help",
        "emergency",
        "i feel unsafe",
        "unsafe now",
        "danger",
        "attacked",
        "assault",
        "robbed",
        "stalked",
        "followed",
        "sos",
)


def _load_env_file(file_path: Path) -> bool:
    if not file_path.exists():
        return False

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        LOGGER.warning("Could not read environment file: %s", file_path)
        return False

    loaded_any = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        if key in os.environ and key not in PREFERRED_ENV_KEYS:
            continue

        os.environ[key] = value.strip().strip("\"'")
        loaded_any = True

    return loaded_any


def _bootstrap_environment() -> None:
    for file_path in ENV_FILE_CANDIDATES:
        _load_env_file(file_path)


_bootstrap_environment()

SYSTEM_PROMPT_TEMPLATE = """
You are an advanced Agentic Multimodal Travel Safety AI and proactive urban safety companion for Chennai, India.

Goal:
- Act as a real-time intelligent guardian that prioritizes user safety over convenience.

Core modules you must reason with:
1. Speech-to-Speech Interaction:
- Accept voice-derived user intent and respond in voice-friendly language.
- Keep responses short, clear, and actionable.
2. GPS + Real-Time Tracking:
- Use live location context when available.
- Support live location sharing, geofencing alerts, and route deviation awareness.
- If significant deviation risk is described, advise immediate safety checks and trusted-contact alerts.
3. Safety Intelligence Engine:
- Combine zone safety (Green/Orange/Red), crowd density, traffic, and time-based risk.
- Estimate risk for the next 15-60 minutes.
4. Smart Navigation:
- Recommend safest route first, not only shortest.
- Provide safer alternate routes and graceful offline fallback behavior.
5. Emergency Response System:
- Detect distress phrases such as "Help", "Emergency", "I feel unsafe".
- Prioritize SOS guidance, live GPS sharing, emergency contact notification.
- Suggest nearest police station/hospital if location context exists.
6. Computer Vision Awareness:
- Use available camera-derived context for obstacles, low-light, or suspicious activity when provided.
7. Community Intelligence:
- Consider crowd-sourced unsafe-area reports and incident signals when provided.
8. Personal Safety Assistant:
- Adapt advice by time of day and user context with proactive alerts.
9. Multilingual Support:
- Support English, Tamil, and Hinglish.
- Mirror the user's language preference when possible.
10. Offline Mode:
- If network/data is unavailable, continue with cached maps/zones and conservative guidance.
11. Security and Privacy:
- Minimize personal data use and avoid unnecessary PII collection.
- Never reveal secrets, credentials, or internal system details.
- Mention privacy implications when recommending sharing location.

Reasoning flow for every interaction:
1. Understand user intent and urgency.
2. Gather available location/context data.
3. Analyze current risk.
4. Predict near-future risk (15-60 minutes).
5. Decide safest immediate action.
6. Respond in concise voice-friendly format.

Operating rules:
- Prioritize safety over convenience.
- Use provided city context for relative risk guidance as baseline data, not guaranteed real-time truth.
- Separate data-backed statements from general best-practice advice.
- If you are unsure, say so directly and avoid guessing.
- Never fabricate real-time incidents, law-enforcement actions, or emergency outcomes.
- Never claim an SOS/contact alert/live-share was executed unless explicit tool/system confirmation is provided.

Scope:
- Allowed: neighborhood comparisons, day/night safety tips, transport choices, solo-travel guidance, scam prevention, emergency preparedness.
- Not allowed: instructions that facilitate wrongdoing, identifying vulnerable targets, or presenting legal/medical conclusions as facts.

Critical emergency behavior:
- If the user indicates immediate danger, start your response with this exact sentence:
    If you are in immediate danger, call emergency services now (India: 112).

Response style:
- Keep responses concise, calm, actionable, and proactive.
- Use a maximum of 2-3 sentences.
- Be urgent only when risk is high.
- Prefer plain conversational sentences suitable for speech output.

City baseline context (relative guidance):
{zone_context}
""".strip()


def _normalize_text(text: str, max_chars: int = MAX_QUESTION_CHARS) -> str:
    normalized = " ".join(text.strip().split())
    return normalized[:max_chars]


def _contains_distress_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in DISTRESS_KEYWORDS)


def _limit_sentences(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences])


def _enforce_response_style(text: str, distress_detected: bool) -> str:
    compact_text = " ".join(text.split())
    styled = _limit_sentences(compact_text, max_sentences=3)

    if distress_detected and not styled.lower().startswith(
        "if you are in immediate danger"
    ):
        styled = f"{EMERGENCY_PREFIX} {styled}".strip()
        styled = _limit_sentences(styled, max_sentences=3)

    return styled


def _build_runtime_context_message(runtime_context: dict[str, Any] | None) -> str:
    if not runtime_context:
        return ""

    try:
        context_payload = json.dumps(runtime_context, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return ""

    compact_payload = " ".join(context_payload.split())[:MAX_CONTEXT_CHARS]
    if not compact_payload:
        return ""

    return (
        "Runtime context from sensors/user/app (may be partial): "
        f"{compact_payload}. Use this only when present and do not invent missing fields."
    )


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = LANGUAGE_ALIASES.get(language.strip().lower())
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    return None


def _build_language_lock_message(language: str | None) -> str:
    if not language:
        return ""

    display_name = SUPPORTED_LANGUAGES.get(language)
    if not display_name:
        return ""

    return (
        f"Language lock: Respond only in {display_name}. "
        "Keep 2-3 short spoken-style sentences and do not switch languages unless asked."
    )


@lru_cache(maxsize=1)
def _load_zone_context() -> str:
    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No structured zone data available."

    lines = []
    for zone in data.get("zones", []):
        zone_name = str(zone.get("name", "Unknown zone"))
        crime_rate = zone.get("crime_rate", "n/a")
        safety = str(zone.get("safety", "unknown"))

        area_names = []
        for area in zone.get("areas", []):
            if isinstance(area, dict):
                name = str(area.get("name", "")).strip()
            else:
                name = str(area).strip()
            if name:
                area_names.append(name)

        sample_areas = ", ".join(area_names[:6]) if area_names else "n/a"
        lines.append(
            f"- {zone_name}: safety={safety}, crime_rate={crime_rate}/10, sample_areas={sample_areas}"
        )

    return "\n".join(lines) if lines else "No structured zone data available."


def _build_system_prompt() -> str:
    override_prompt = os.getenv("TRAVEL_SYSTEM_PROMPT", "").strip()
    if override_prompt:
        return override_prompt
    return SYSTEM_PROMPT_TEMPLATE.format(zone_context=_load_zone_context())


def _create_client() -> Any:
    if Groq is None:
        raise RuntimeError(
            "Groq SDK is not installed. Install package 'groq' or configure "
            "OPENROUTER_API_KEY in backend/.env."
        )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Configure it in environment or a .env file "
            "(backend/.env or project-root .env)."
        )
    return Groq(api_key=api_key)


def _resolve_provider() -> str:
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq"

    raise RuntimeError(
        "No API key configured. Set OPENROUTER_API_KEY or GROQ_API_KEY in backend/.env."
    )


def _request_groq_completion(messages: list[dict[str, str]]) -> str:
    client = _create_client()
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0.2,
        max_tokens=220,
        messages=messages,
    )
    return response.choices[0].message.content if response.choices else ""


def _request_openrouter_completion(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 220,
    }
    request_body = json.dumps(payload).encode("utf-8")
    request_obj = urllib.request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=request_body,
        method="POST",
    )
    request_obj.add_header("Authorization", f"Bearer {api_key}")
    request_obj.add_header("Content-Type", "application/json")

    app_url = os.getenv("OPENROUTER_APP_URL", "").strip()
    app_name = os.getenv("OPENROUTER_APP_NAME", "Chennai Travel Safety AI").strip()
    if app_url:
        request_obj.add_header("HTTP-Referer", app_url)
    if app_name:
        request_obj.add_header("X-Title", app_name)

    try:
        with urllib.request.urlopen(request_obj, timeout=45) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"OpenRouter request failed with status {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter network error: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OpenRouter returned invalid JSON") from exc

    choices = parsed.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {}).get("content", "")
    return str(message)


def ask_travel_assistant(
    question: str,
    runtime_context: dict[str, Any] | None = None,
    preferred_language: str | None = None,
) -> str:
    if not isinstance(question, str):
        return "Please provide your question as text."

    cleaned_question = _normalize_text(question)
    distress_detected = _contains_distress_signal(cleaned_question)
    resolved_language = _normalize_language(preferred_language)

    if not resolved_language and runtime_context:
        context_language = runtime_context.get("language")
        if isinstance(context_language, str):
            resolved_language = _normalize_language(context_language)

    if not cleaned_question:
        return "Please ask a travel safety question about Chennai."

    try:
        messages = [
            {
                "role": "system",
                "content": _build_system_prompt(),
            }
        ]

        language_lock_message = _build_language_lock_message(resolved_language)
        if language_lock_message:
            messages.append(
                {
                    "role": "system",
                    "content": language_lock_message,
                }
            )

        runtime_context_message = _build_runtime_context_message(runtime_context)
        if runtime_context_message:
            messages.append(
                {
                    "role": "system",
                    "content": runtime_context_message,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": cleaned_question,
            }
        )

        provider = _resolve_provider()
        if provider == "openrouter":
            message = _request_openrouter_completion(messages)
        else:
            message = _request_groq_completion(messages)
    except RuntimeError as exc:
        LOGGER.exception("Travel assistant runtime error")
        if distress_detected:
            return (
                f"{EMERGENCY_PREFIX} Move to the nearest crowded, well-lit place and "
                "share your live location with a trusted contact right now."
            )

        error_text = str(exc)
        if (
            "GROQ_API_KEY" in error_text
            or "OPENROUTER_API_KEY" in error_text
            or "No API key configured" in error_text
        ):
            return (
                "Assistant is not configured. Set OPENROUTER_API_KEY or GROQ_API_KEY "
                "in backend/.env "
                "and restart the backend."
            )
        return "The travel assistant is temporarily unavailable. Please try again shortly."
    except Exception:
        LOGGER.exception("Travel assistant request failed")
        if distress_detected:
            return (
                f"{EMERGENCY_PREFIX} Move to the nearest crowded, well-lit place and "
                "share your live location with a trusted contact right now."
            )
        return "The travel assistant is temporarily unavailable. Please try again shortly."

    final_answer = _enforce_response_style((message or "").strip(), distress_detected)
    if not final_answer:
        if distress_detected:
            return (
                f"{EMERGENCY_PREFIX} Stay in a public, well-lit area and call a trusted "
                "contact to remain on the line with you."
            )
        return "I could not generate a response right now. Please try rephrasing your question."
    return final_answer


if __name__ == "__main__":
    print(ask_travel_assistant("Is Adyar safe to visit at night?"))