import spacy

class ClinicalNER:
    def __init__(self, model_name="en_core_sci_sm"):
        print(f"[*] Loading SciSpaCy model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"[!] Model '{model_name}' not found. "
                "Ensure you ran: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
            )

    def extract_entities(self, text: str):
        if not text or not isinstance(text, str):
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            cleaned_span = ent.text.strip(" ,.;:!?\t\n")
            if len(cleaned_span) <= 1:
                continue

            # Use the ORIGINAL doc's tokens for this span — preserves real
            # sentence context (dependency parse, POS) instead of re-parsing
            # the entity string in isolation.
            span_tokens = list(ent)  # tokens as they appear in `doc`, not a re-parse
            root = ent.root          # syntactic head of this span, in original context

            # Drop spans that are purely stopwords/punctuation
            if all(tok.is_stop or tok.is_punct or tok.is_space for tok in span_tokens):
                continue

            # Drop spans whose HEAD token is a pronoun/determiner, or whose
            # head plays the grammatical subject role (nsubj) in the real sentence.
            # This is what correctly catches "Patient" in "Patient presents with...".
            if root.pos_ in {"PRON", "DET"} or root.dep_ in {"nsubj", "nsubjpass"}:
                continue

            # Extra guard: single-token generic person/role nouns acting as
            # the sentence subject are the main false-positive source; the
            # dep_ check above already covers this without any lexicon.

            if cleaned_span.lower() not in [e.lower() for e in entities]:
                entities.append(cleaned_span)

        return entities


if __name__ == "__main__":
    ner = ClinicalNER()

    sample_text = (
        "Patient presents with severe acute rhinosinusitis, facial pain, and nasal obstruction. "
        "Also reports high fever, generalized fatigue, and mild shortness of breath. "
        "Recommended Amoxicillin 500mg and saline nasal sprays."
    )

    print("\n[+] Analyzing clinical text...")
    print(f"Input Text:\n'{sample_text}'\n")

    extracted = ner.extract_entities(sample_text)
    print("[+] Extracted Biomedical Entities:")
    print(extracted)
    