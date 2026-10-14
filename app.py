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

st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")

BULLET = re.compile(r"^[∙•·▪●○]\s*")


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

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Guidelines to retrieve", 1, 3, 2)
backend = available_backend()
use_llm = st.sidebar.checkbox(
    "Generate AI summary (LLM)",
    value=backend is not None,
    disabled=backend is None,
    help="Needs ANTHROPIC_API_KEY or LOCAL_LLM=1. See src/summarizer.py.",
)
if backend is None:
    st.sidebar.caption("LLM summary is off: no backend configured.")
else:
    st.sidebar.caption(f"LLM backend: {backend}")

samples = load_samples()
default_text, choice = "", "- type your own -"
if samples is not None:
    labels = [f"{r.sample_id}. {r.expected_domain}" for r in samples.itertuples()]
    choice = st.sidebar.selectbox("Load a demo sample", ["- type your own -"] + labels)
    if choice != "- type your own -":
        default_text = samples.iloc[labels.index(choice)]["clinical_text"]

# ------------------------------------------------------------------ main page
st.title("🏥 Clinical Decision Support — ICMR Treatment Guideline Retrieval")
st.caption("Enter a patient's clinical description to get the predicted specialty, extracted clinical "
           "entities, and the matching ICMR Standard Treatment Workflow.")

clinical_text = st.text_area(
    "Patient Clinical Description", value=default_text, height=150, key=f"text_{choice}",
    placeholder="e.g. Patient presents with severe acute rhinosinusitis, facial pain, and nasal obstruction...",
)

if st.button("Analyze", type="primary"):
    if not clinical_text.strip():
        st.warning("Please enter some clinical text first.")
        st.stop()

    with st.spinner("Classifying, extracting entities and searching guidelines..."):
        raw_specialty = predict_specialty(clinical_text)
        domain = map_specialty_to_domain(raw_specialty)
        entities = ner.extract_entities(clinical_text)
        results = retriever.retrieve(clinical_text, entities=entities, domain=domain, top_k=top_k)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🩺 Predicted Specialty")
        st.info(raw_specialty)
    with col2:
        st.subheader("🔑 Extracted Clinical Entities")
        st.write(", ".join(entities) if entities else "No entities found.")

    if not results:
        st.warning("No relevant guidelines found.")
        st.stop()

    expected = set(DOMAIN_EXPANSION.get(domain, [domain]))
    if results[0]["domain"] not in expected:
        st.warning(
            f"The best-matching guideline is from **{results[0]['domain']}**, but the classifier predicted "
            f"**{raw_specialty}**. The retrieval is driven by the clinical text, so please verify the specialty."
        )

    if use_llm:
        with st.spinner("Writing summary from the retrieved guideline..."):
            summary = summarize(clinical_text, entities, results)
        st.subheader("🧾 AI Summary (generated only from the guideline text below)")
        if summary and summary["text"]:
            st.success(summary["text"])
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