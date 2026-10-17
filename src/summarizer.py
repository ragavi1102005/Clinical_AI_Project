"""
Module 3c (OPTIONAL): grounded LLM summary of the retrieved ICMR guideline.

This is the "G" in RAG: the retriever finds the authoritative guideline page(s); the LLM only
REWORDS them into a short, doctor-friendly summary. It is told to use nothing except the text it
is given, and the full source page is always shown next to the summary so the doctor can verify it.

Backends (picked automatically, in this order, nothing else in the project changes):
  1. Anthropic API : set ANTHROPIC_API_KEY                      (pip install anthropic; paid, no free tier)
  2. Groq (FREE tier, no credit card): set GROQ_API_KEY          (uses the 'requests' package)
  3. Any other OpenAI-compatible provider (e.g. OpenRouter): set LLM_API_KEY, LLM_BASE_URL and
     LLM_COMPAT_MODEL (copy the base URL and model name from the provider's docs)
  4. Local model   : set LOCAL_LLM=1  (needs transformers + torch; first run downloads ~3 GB, CPU is slow)
If none is configured, summarize() returns None and the app simply shows the guideline text.

Windows PowerShell examples (start Streamlit from the SAME terminal afterwards):
    $env:GROQ_API_KEY = "gsk_..."
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:LOCAL_LLM = "1"
Only ever send demo / synthetic text to a free hosted API, never real patient data.
"""

import os
import re

ANTHROPIC_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_CHARS_PER_PAGE = 12000

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for doctors in India. You rewrite official ICMR "
    "Standard Treatment Workflow (STW) text into a short, clear summary.\n"
    "STRICT RULES:\n"
    "1. Use ONLY information that appears in the GUIDELINE TEXT. Never add drugs, doses, durations, "
    "investigations, referrals or advice from your own knowledge.\n"
    "2. Copy drug names, doses and durations exactly as written in the guideline.\n"
    "3. The guideline text may have jumbled line order (it was extracted from an infographic); "
    "reconstruct the meaning carefully but do not invent anything.\n"
    "4. The FIRST line of your reply must be exactly 'VERDICT: MATCH' or 'VERDICT: NO_MATCH'. "
    "Use NO_MATCH whenever the guideline text does not specifically cover the patient's MAIN problem "
    "(for example it is for a different disease or a different age group, or only mentions a related word). "
    "If NO_MATCH, write nothing else.\n"
    "5. If MATCH: after the verdict line write at most 180 words using exactly these headings: "
    "Matched guideline / Recommended management / Investigations / Refer if."
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"

_local_pipe = None


def _compat_settings():
    """(name, api_key, base_url, model) for an OpenAI-compatible provider, or None."""
    model = os.environ.get("LLM_COMPAT_MODEL", "")
    if os.environ.get("GROQ_API_KEY"):
        return ("groq", os.environ["GROQ_API_KEY"], GROQ_BASE_URL, model or GROQ_DEFAULT_MODEL)
    if os.environ.get("LLM_API_KEY") and os.environ.get("LLM_BASE_URL") and model:
        return ("custom", os.environ["LLM_API_KEY"], os.environ["LLM_BASE_URL"].rstrip("/"), model)
    return None


def available_backend():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    compat = _compat_settings()
    if compat:
        return compat[0]                      # "groq" or "custom"
    if os.environ.get("LOCAL_LLM") == "1":
        return "local"
    return None


def build_prompt(patient_text, entities, results, max_pages=2):
    pages = []
    for r in results[:max_pages]:
        pages.append(f"[{r['title']} - {r['source_file']} page {r['page']}]\n{r['text'][:MAX_CHARS_PER_PAGE]}")
    ents = ", ".join(entities) if entities else "none extracted"
    return (
        f"PATIENT DESCRIPTION:\n{patient_text}\n\n"
        f"KEY CLINICAL TERMS: {ents}\n\n"
        "GUIDELINE TEXT:\n" + "\n\n".join(pages) + "\n\n"
        "Write the summary now, following the rules."
    )


def _call_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def _call_openai_compatible(prompt):
    """Chat-completions call that works for Groq, OpenRouter and other OpenAI-compatible APIs."""
    import requests
    _, key, base_url, model = _compat_settings()
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "max_tokens": 1500,        # reasoning models spend part of this on thinking before they answer
    }
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    if not text:
        raise RuntimeError("the model returned an empty answer (try another model name)")
    return text


def _call_local(prompt):
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline
        _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    out = _local_pipe(messages, max_new_tokens=350, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip()


def _apply_verdict(text, results):
    """NO_MATCH -> fixed safe message (no model-written advice). MATCH -> summary without the verdict line."""
    first, _, rest = text.strip().partition("\n")
    m = re.match(r"\W*VERDICT\s*:\s*(NO[_ ]MATCH|MATCH)", first, re.IGNORECASE)
    if m and m.group(1).upper().startswith("NO"):
        return ("No matching guideline was found in the knowledge base for this presentation. "
                f"The closest retrieved guideline ('{results[0]['title']}') covers a different problem, so no "
                "treatment advice is generated. Please consult a clinician or the full ICMR STW portal.")
    return rest.strip() if m else text.strip()


def summarize(patient_text, entities, results):
    """Returns {"text", "backend", "error"} or None when no backend is configured."""
    backend = available_backend()
    if backend is None or not results:
        return None
    prompt = build_prompt(patient_text, entities, results)
    try:
        if backend == "anthropic":
            text = _call_anthropic(prompt)
        elif backend == "local":
            text = _call_local(prompt)
        else:                                   # "groq" or "custom"
            text = _call_openai_compatible(prompt)
        return {"text": _apply_verdict(text, results), "backend": backend, "error": None}
    except Exception as e:                      # never crash the app because of the LLM
        return {"text": None, "backend": backend, "error": f"{type(e).__name__}: {e}"}