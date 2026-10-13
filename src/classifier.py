import os
import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


def clean_clinical_text(text):
    """Normalize and clean raw transcription text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SPECIALTY_TO_DOMAIN = {
    "Cardiovascular / Pulmonary": "cardiology",
    "ENT - Otolaryngology": "ent",
    "Nephrology": "nephrology",
    "Neurology": "neurology",
    "Orthopedic": "orthopaedics",
    "Psychiatry / Psychology": "psychiatry",
}


def map_specialty_to_domain(specialty_label: str) -> str:
    return SPECIALTY_TO_DOMAIN.get(specialty_label.strip(), "general")


def train_and_save_classifier(raw_data_path="data/mtsamples.csv",
                               output_data_path="data/mtsamples_cleaned.csv",
                               model_dir="models"):
    print("[*] Loading raw clinical data...")
    if not os.path.exists(raw_data_path):
        print(f"[!] Error: Cannot find {raw_data_path}. Please make sure mtsamples.csv is in the data folder.")
        return

    df = pd.read_csv(raw_data_path)

    df = df.dropna(subset=['transcription', 'medical_specialty'])
    df['cleaned_text'] = df['transcription'].apply(clean_clinical_text)

    target_specialties = [
        ' Cardiovascular / Pulmonary',
        ' ENT - Otolaryngology',
        ' Nephrology',
        ' Neurology',
        ' Orthopedic',
        ' Psychiatry / Psychology'
    ]

    df_filtered = df[df['medical_specialty'].isin(target_specialties)].copy()
    df_filtered['medical_specialty'] = df_filtered['medical_specialty'].str.strip()

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df_filtered.to_csv(output_data_path, index=False)
    print(f"[+] Dataset preprocessed: {len(df_filtered)} samples retained across {df_filtered['medical_specialty'].nunique()} specialties.")

    # Show class distribution so imbalance is visible every time we train
    print("\n[*] Class distribution (sample count per specialty):")
    print(df_filtered['medical_specialty'].value_counts())
    print()

    X = df_filtered['cleaned_text']
    y = df_filtered['medical_specialty']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[*] Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[*] Training Support Vector Classifier (class-balanced)...")
    # class_weight='balanced' makes the SVM pay more attention to
    # under-represented specialties (e.g. Psychiatry) instead of always
    # favoring whichever specialty has the most training examples.
    classifier = LinearSVC(random_state=42, C=1.0, class_weight='balanced')
    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"[+] Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"==========================================\n")
    print(classification_report(y_test, y_pred))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(model_dir, "svm_classifier.pkl"))
    print(f"[+] Models successfully saved in '{model_dir}/'.")


def predict_specialty(text, model_dir="models"):
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(model_dir, "svm_classifier.pkl"))
    cleaned = clean_clinical_text(text)
    vec = vectorizer.transform([cleaned])
    return classifier.predict(vec)[0]


def predict_domain(text, model_dir="models"):
    raw_label = predict_specialty(text, model_dir)
    return map_specialty_to_domain(raw_label)


if __name__ == "__main__":
    train_and_save_classifier()

def predict_specialty_debug(text, model_dir="models"):
    """
    Diagnostic helper: shows how confident the model was in EVERY
    specialty, not just its final answer. Useful for understanding
    whether a wrong prediction was a close call or a confident mistake.
    """
    vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join(model_dir, "svm_classifier.pkl"))

    cleaned = clean_clinical_text(text)
    vec = vectorizer.transform([cleaned])

    scores = classifier.decision_function(vec)[0]
    classes = classifier.classes_

    ranked = sorted(zip(classes, scores), key=lambda x: x[1], reverse=True)

    print("\n[Decision scores — higher means more confident]")
    for label, score in ranked:
        print(f"  {label:30s} {score:.3f}")

    return ranked[0][0]