"""
Module 3c (OPTIONAL): grounded LLM summary of the retrieved ICMR guideline.

This is the "G" in RAG: the retriever finds the authoritative guideline page(s); the LLM only
REWORDS them into a short, doctor-friendly summary. It is told to use nothing except the text it
is given, and the full source page is always shown next to the summary so the doctor can verify it.

Backends (picked automatically, nothing else in the project changes):
  1. Anthropic API : set the environment variable ANTHROPIC_API_KEY   (pip install anthropic)
  2. Local model   : set LOCAL_LLM=1  (needs transformers + torch; first run downloads ~3 GB, CPU is slow)
If neither is configured, summarize() returns None and the app simply shows the guideline text.

Windows PowerShell examples:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:LOCAL_LLM = "1"
"""

import os

ANTHROPIC_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_CHARS_PER_PAGE = 4500

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for doctors in India. You rewrite official ICMR "
    "Standard Treatment Workflow (STW) text into a short, clear summary.\n"
    "STRICT RULES:\n"
    "1. Use ONLY information that appears in the GUIDELINE TEXT. Never add drugs, doses, durations, "
    "investigations or advice from your own knowledge.\n"
    "2. Copy drug names, doses and durations exactly as written in the guideline.\n"
    "3. If the guideline text does not match the patient's problem, say so in one sentence instead of guessing.\n"
    "4. The guideline text may have jumbled line order (it was extracted from an infographic); "
    "reconstruct the meaning carefully but do not invent anything.\n"
    "5. Maximum 180 words. Use exactly these headings: "
    "Matched guideline / Recommended management / Investigations / Refer if."
)

_local_pipe = None


def available_backend():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
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


def _call_local(prompt):
    global _local_pipe
    if _local_pipe is None:
        from transformers import pipeline
        _local_pipe = pipeline("text-generation", model=LOCAL_MODEL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    out = _local_pipe(messages, max_new_tokens=350, do_sample=False)
    return out[0]["generated_text"][-1]["content"].strip()


def summarize(patient_text, entities, results):
    """Returns {"text", "backend", "error"} or None when no backend is configured."""
    backend = available_backend()
    if backend is None or not results:
        return None
    prompt = build_prompt(patient_text, entities, results)
    try:
        text = _call_anthropic(prompt) if backend == "anthropic" else _call_local(prompt)
        return {"text": text, "backend": backend, "error": None}
    except Exception as e:                      # never crash the app because of the LLM
        return {"text": None, "backend": backend, "error": f"{type(e).__name__}: {e}"}