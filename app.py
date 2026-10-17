"""
Module 4: Streamlit web app
classifier -> NER -> hybrid retriever (whole guideline pages) -> optional grounded LLM summary
"""

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.classifier import predict_specialty, map_specialty_to_domain
from src.ner import ClinicalNER
from src.rag import HybridRetriever, DOMAIN_EXPANSION
from src.summarizer import available_backend, summarize

st.set_page_config(page_title="Clinical RAG Assistant", layout="wide", initial_sidebar_state="collapsed")

BULLET = re.compile(r"^[∙•·▪●○]\s*")
NO_SAMPLE = "- type your own -"
DEPT_LABEL = {"ent": "ENT - Otolaryngology", "cardiology": "Cardiovascular / Pulmonary",
              "hypertension": "Cardiovascular (Hypertension)", "psychiatry": "Psychiatry / Psychology",
              "orthopaedics": "Orthopedic", "nephrology": "Nephrology", "neurology": "Neurology"}


@st.cache_resource
def load_models():
    return ClinicalNER(), HybridRetriever()


@st.cache_data
def load_samples():
    p = Path("demo_test_samples.csv")
    return pd.read_csv(p) if p.exists() else None


def page_to_markdown(text: str) -> str:
    """Re-joins wrapped lines into their bullet and renders headings in bold (display only)."""
    items = []
    for ln in (l.strip() for l in text.split("\n")):
        if not ln:
            continue
        if ln.isupper() and len(ln) > 3 and items and items[-1].isupper() and len(items[-1]) > 3:
            items[-1] += " · " + ln          # consecutive table headings -> one line instead of 8 separate headings
            continue
        starts_new = (not items) or BULLET.match(ln) or (ln.isupper() and len(ln) > 3)
        if starts_new:
            items.append(ln)
        else:
            items[-1] += " " + ln
    out = []
    for it in items:
        it = it.replace("$", "\\$")
        if BULLET.match(it):
            out.append("- " + BULLET.sub("", it))
        elif it.isupper() and len(it) > 3:
            out.append(f"\n**{it}**\n")
        else:
            out.append(it + "\n")
    return "\n".join(out)


ner, retriever = load_models()

# ------------------------------------------------------------------ main page
st.title("🏥 Clinical Decision Support — ICMR Treatment Guideline Retrieval")
st.caption("Enter a patient's clinical description to get the predicted specialty, extracted clinical "
           "entities, and the matching ICMR Standard Treatment Workflow.")

# optional: pick a ready-made demo case (needs demo_test_samples.csv in the same folder as app.py)
samples = load_samples()
default_text, choice = "", NO_SAMPLE
if samples is not None:
    label_col = "expected_guideline" if "expected_guideline" in samples.columns else "expected_domain"
    labels = [f"{r.sample_id}. {str(getattr(r, label_col))[:70]}" for r in samples.itertuples()]
    choice = st.selectbox("Load a demo sample (optional)", [NO_SAMPLE] + labels)
    if choice != NO_SAMPLE:
        default_text = samples.iloc[labels.index(choice)]["clinical_text"]

clinical_text = st.text_area(
    "Patient Clinical Description", value=default_text, height=150, key=f"text_{choice}",
    placeholder="e.g. Patient presents with severe acute rhinosinusitis, facial pain, and nasal obstruction...",
)

opt_left, opt_right = st.columns(2)
with opt_left:
    top_k = st.slider("Number of guidelines to retrieve", 1, 3, 2)
with opt_right:
    backend = available_backend()
    if backend:
        use_llm = st.checkbox(f"Generate AI summary (LLM backend: {backend})", value=True)
    else:
        use_llm = False
        st.markdown("**AI summary:** off")
        st.caption("No LLM is connected, so the app runs in retrieval-only mode. "
                   "To switch it on, see the instructions at the top of src/summarizer.py.")

if st.button("Analyze", type="primary"):
    if not clinical_text.strip():
        st.warning("Please enter some clinical text first.")
        st.stop()

    with st.spinner("Classifying, extracting entities and searching guidelines..."):
        raw_specialty = predict_specialty(clinical_text)
        domain = map_specialty_to_domain(raw_specialty)
        entities = ner.extract_entities(clinical_text)
        # no domain boost: the classifier is trained on long MTSamples notes and often mislabels short
        # queries, and a wrong boost pushed the right guideline down. Retrieval decides the guideline.
        results = retriever.retrieve(clinical_text, entities=entities, domain=None, top_k=top_k)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🩺 Predicted Specialty")
        dept = DEPT_LABEL.get(results[0]["domain"], raw_specialty) if results else raw_specialty
        st.info(dept)
        if map_specialty_to_domain(raw_specialty) == (results[0]["domain"] if results else None):
            st.caption("Text classifier agrees with the retrieved guideline.")
        else:
            st.caption(f"Department taken from the best-matching ICMR guideline. "
                       f"Text classifier alone predicted: {raw_specialty}")
    with col2:
        st.subheader("🔑 Extracted Clinical Entities")
        st.write(", ".join(entities) if entities else "No entities found.")

    if not results:
        st.warning("No relevant guidelines found.")
        st.stop()

    if use_llm:
        with st.spinner("Writing summary from the retrieved guideline..."):
            summary = summarize(clinical_text, entities, results)
        st.subheader("🧾 AI Summary (generated only from the guideline text below)")
        if summary and summary["text"]:
            (st.warning if summary["text"].startswith("No matching guideline") else st.success)(summary["text"])
            st.caption("AI-generated rewording of the retrieved ICMR text. Always verify against the source guideline below.")
        elif summary:
            st.error(f"LLM summary failed ({summary['error']}). The retrieved guideline is shown below.")

    st.subheader("📄 Retrieved ICMR Guideline(s)")
    for i, r in enumerate(results, 1):
        st.markdown(f"### {i}. {r['title']}")
        st.caption(f"{r['source_file']} · page {r['page']} · domain: {r['domain']} · relevance score: {r['rrf_score']}")
        with st.expander("Best-matching passages", expanded=False):
            for j, p in enumerate(r["matched_passages"]):
                if j:
                    st.divider()
                st.markdown(page_to_markdown(p))
        with st.expander("Full guideline page", expanded=(i == 1)):
            st.markdown(page_to_markdown(r["text"]))