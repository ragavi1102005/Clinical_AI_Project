"""
Module 4: Streamlit Web App
Connects classifier -> NER -> hybrid RAG retriever into one dashboard.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent))

from src.classifier import predict_specialty, predict_domain
from src.ner import ClinicalNER
from src.rag import HybridRetriever

st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")


@st.cache_resource
def load_models():
    """Loads heavy models once and reuses them across user interactions."""
    ner = ClinicalNER()
    retriever = HybridRetriever()
    return ner, retriever


ner, retriever = load_models()

st.title("🏥 Clinical Decision Support — ICMR Treatment Guideline Retrieval")
st.caption(
    "Enter a patient's clinical description to get the predicted specialty, "
    "extracted clinical entities, and matching ICMR treatment guidance."
)

clinical_text = st.text_area(
    "Patient Clinical Description",
    height=150,
    placeholder="e.g. Patient presents with severe acute rhinosinusitis, facial pain, and nasal obstruction...",
)

if st.button("Analyze", type="primary"):
    if not clinical_text.strip():
        st.warning("Please enter some clinical text first.")
    else:
        with st.spinner("Classifying specialty..."):
            raw_specialty = predict_specialty(clinical_text)
            domain = predict_domain(clinical_text)

        with st.spinner("Extracting clinical entities..."):
            entities = ner.extract_entities(clinical_text)

        with st.spinner("Retrieving treatment guidelines..."):
            results = retriever.retrieve(
                query_text=clinical_text,
                entities=entities,
                domain=domain,
                top_k=3,
            )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🩺 Predicted Specialty")
            st.info(raw_specialty)

        with col2:
            st.subheader("🔑 Extracted Clinical Entities")
            st.write(", ".join(entities) if entities else "No entities found.")

        st.subheader("📄 Retrieved Treatment Guidance")
        if not results:
            st.warning("No relevant guidelines found.")
        else:
            for i, r in enumerate(results, 1):
                with st.expander(f"Result {i} — {r['source_file']} (score: {r['rrf_score']})"):
                    st.text(r["text"])