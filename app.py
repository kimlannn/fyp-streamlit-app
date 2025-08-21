import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU init on Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
import easyocr
from PIL import Image
import pdfplumber
import re

# =========================================
# Cache model loading for speed
# =========================================
@st.cache_resource
def load_foundation_model():
    model = load_model("foundation_model.h5", compile=False)
    scaler = joblib.load("foundation_scaler.pkl")
    with open("foundation_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

@st.cache_resource
def load_degree_model():
    model = load_model("degree_model.h5", compile=False)
    scaler = joblib.load("degree_scaler.pkl")
    with open("degree_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

foundation_model, foundation_scaler, foundation_encoder = load_foundation_model()
degree_model, degree_scaler, degree_encoder = load_degree_model()

# =========================================
# Mappings
# =========================================
grade_mapping_foundation = {
    "A": 6, "A+": 6, "A-": 6,
    "B": 5, "B+": 5, "B-": 5,
    "C": 4, "C+": 4, "C-": 4,
    "D": 3, "D+": 3, "D-": 3,
    "E": 2, "E+": 2, "E-": 2,
    "F": 1, "F+": 1, "F-": 1,
    "0": 0
}
qualification_mapping_foundation = {"SPM": 0, "UEC": 1, "O-Level": 2}

grade_mapping_degree = {
    "A+": 4.0, "A": 4.0, "A-": 3.67,
    "B+": 3.33, "B": 3.0, "B-": 2.67,
    "C+": 2.33, "C": 2.0, "C-": 1.67,
    "D+": 1.33, "D": 1.0, "E": 0.67, "F": 0.0,
    "0": 0
}
qualification_mapping_degree = {
    "A-Level": 0, "Diploma": 1,
    "Foundation in Arts, UTAR": 2,
    "Foundation in Science, UTAR": 3,
    "Matriculation": 4,
    "STAM": 5, "STPM": 6, "UEC": 7
}

foundation_subjects = ["Mathematics", "English", "Science", "Physics", "Chemistry",
                       "Biology", "Additional Mathematics", "Accounting", "Chinese", "Art"]

degree_subjects = ["Mathematics", "Additional Mathematics", "English", "Physics",
                   "Biology", "Chemistry", "ICT", "Technology", "Pendidikan Seni",
                   "Advanced Mathematics I", "Advanced Mathematics II"]

# =========================================
# Preprocessing
# =========================================
@st.cache_data
def preprocess_foundation(user_input):
    df_new = pd.DataFrame([user_input])
    df_new["Qualification"] = df_new["Qualification"].map(qualification_mapping_foundation)
    for col in foundation_subjects:
        df_new[col] = df_new[col].map(grade_mapping_foundation).fillna(0).astype(int)
    return foundation_scaler.transform(df_new)

@st.cache_data
def preprocess_degree(user_input):
    df_new = pd.DataFrame([user_input])
    df_new["Qualification"] = df_new["Qualification"].map(qualification_mapping_degree).fillna(0).astype(int)
    df_new["CGPA"] = df_new["CGPA"].fillna(0)
    for col in degree_subjects:
        df_new[col] = df_new[col].map(grade_mapping_degree).fillna(0)
    df_new = df_new[["Qualification", "CGPA"] + degree_subjects]
    return degree_scaler.transform(df_new)

def get_top_n_programmes(model, X_new, encoder, n=10):  # default 10 now
    pred_probs = model.predict(X_new, verbose=0)[0]
    top_idx = np.argsort(pred_probs)[::-1][:n]
    return encoder.inverse_transform(top_idx)

# =========================================
# OCR and PDF text extraction
# =========================================
reader = easyocr.Reader(["en"])

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    else:  # image
        image = Image.open(uploaded_file)
        result = reader.readtext(np.array(image))
        text = " ".join([res[1] for res in result])
    return text

# =========================================
# OCR Parsing Logic
# =========================================
subject_aliases = {
    "bahasa melayu": "Bahasa Melayu",
    "bahasa inggeris": "English",
    "english": "English",
    "pendidikan moral": "Pendidikan Moral",
    "sejarah": "Sejarah",
    "mathematics": "Mathematics",
    "maths": "Mathematics",
    "additional mathematics": "Additional Mathematics",
    "add maths": "Additional Mathematics",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "biology": "Biology",
    "bahasa cina": "Chinese",
    "chinese": "Chinese",
    "pendidikan seni": "Art",
    "art": "Art",
    "ict": "ICT",
    "technology": "Technology",
    "advanced mathematics i": "Advanced Mathematics I",
    "advanced mathematics ii": "Advanced Mathematics II"
}
grade_pattern = re.compile(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D|E|F)\b", re.IGNORECASE)

def normalize(text):
    return re.sub(r"[^a-z0-9+]", " ", text.lower()).strip()

def parse_grades(text, mode="foundation"):
    lines = text.splitlines()
    results = {}

    subjects = foundation_subjects if mode == "foundation" else degree_subjects

    for line in lines:
        norm_line = normalize(line)
        for alias, subject in subject_aliases.items():
            if alias in norm_line:
                match = grade_pattern.search(line)
                if match:
                    grade = match.group(0).upper().replace(" ", "")
                    results[subject] = grade
                break

    final_results = {subj: results.get(subj, "0") for subj in subjects}
    return final_results

# =========================================
# Questionnaire (same as before)
# =========================================
general_questions = [
    ("Which activity do you enjoy the most?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which type of task do you usually enjoy?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("What kind of schoolwork feels most satisfying?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which hobby sounds most fun?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which kind of job you’d enjoy?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which topics interest you the most?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which project would you most enjoy?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which type of problem do you prefer?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
    ("Which would you most like to learn more about?", ["Maths", "Engineering", "Software Engineering", "Architecture"]),
]

maths_questions = {
    "Which of the following sounds most interesting?": ["Applied Mathematics", "Financial Mathematics", "Actuarial Science", "Quantity Surveying"],
    "What kind of problems do you enjoy solving most?": ["Applied Mathematics", "Financial Mathematics", "Actuarial Science", "Quantity Surveying"],
}

engineering_questions = {
    "Which of these jobs sounds the most exciting?": [
        "Biomedical Engineering", "Chemical Engineering", "Civil Engineering", "Electrical & Electronic Engineering",
        "Materials Engineering", "Mechanical Engineering", "Mechatronics Engineering", "Telecommunications Engineering"
    ]
}

# =========================================
# Streamlit App
# =========================================
st.title("🎓 UTAR Programme Recommendation System")
option = st.radio("Choose Recommendation Type:", ["Foundation", "Degree Programme"])

# ===== Foundation Path =====
if option == "Foundation":
    st.header("Foundation Programme Recommendation")
    uploaded_file = st.file_uploader("Upload your academic result (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    qualification = st.selectbox("Qualification:", ["SPM", "UEC", "O-Level"])

    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        extracted_grades = parse_grades(text, mode="foundation")

        st.subheader("Validate Extracted Results")
        grade_options = list(grade_mapping_foundation.keys())
        corrected_grades = {}
        for subj in foundation_subjects:
            grade_val = extracted_grades.get(subj, "0")
            corrected_grades[subj] = st.selectbox(f"{subj}:", grade_options,
                                                  index=grade_options.index(grade_val) if grade_val in grade_options else len(grade_options)-1)

        if st.button("Get Foundation Recommendation"):
            user_input = {"Qualification": qualification, **corrected_grades}
            X_new = preprocess_foundation(user_input)
            probs = foundation_model.predict(X_new, verbose=0)[0]
            top2_idx = probs.argsort()[-2:][::-1]
            top2_progs = foundation_encoder.inverse_transform(top2_idx)
            st.success(f"Top Recommendation: {top2_progs[0]}")
            st.info(f"Alternative Recommendation: {top2_progs[1]}")

# ===== Degree Path =====
else:
    st.header("Degree Programme Recommendation")
    uploaded_file = st.file_uploader("Upload your academic result (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    qualification = st.selectbox("Qualification:", list(qualification_mapping_degree.keys()))
    cgpa = st.number_input("Enter CGPA:", min_value=0.0, max_value=4.0, step=0.01, value=0.0)

    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        extracted_grades = parse_grades(text, mode="degree")

        st.subheader("Validate Extracted Results")
        grade_options = list(grade_mapping_degree.keys())
        corrected_grades = {}
        for subj in degree_subjects:
            grade_val = extracted_grades.get(subj, "0")
            corrected_grades[subj] = st.selectbox(f"{subj}:", grade_options,
                                                  index=grade_options.index(grade_val) if grade_val in grade_options else len(grade_options)-1)

        if st.button("Continue to Questionnaire"):
            user_input = {"Qualification": qualification, "CGPA": cgpa, **corrected_grades}
            X_new = preprocess_degree(user_input)
            top10 = get_top_n_programmes(degree_model, X_new, degree_encoder, n=10)
            st.session_state["top_predicted"] = top10

    # === Questionnaire stage ===
if "top_predicted" in st.session_state:
    st.header("General Interest Questionnaire")

    if "field" not in st.session_state:
        scores = {"Maths": 0, "Engineering": 0, "Software Engineering": 0, "Architecture": 0}

        for q, fields in general_questions:
            ans = st.radio(q, fields, key=f"general_{q}")
            scores[ans] += 1

        if st.button("Submit General Questionnaire"):
            max_score = max(scores.values())
            winners = [k for k, v in scores.items() if v == max_score]

            if len(winners) > 1:
                st.warning(f"Tie detected! Possible fields: {', '.join(winners)}")
            else:
                st.session_state.field = winners[0]
                st.success(f"Your strongest interest field: {st.session_state.field}")

    # --- Follow-up questionnaire ---
    if "field" in st.session_state:
        field = st.session_state.field

        final_recommendations = []

        if field == "Software Engineering":
            final_recommendations = ["Software Engineering"]

        elif field == "Architecture":
            final_recommendations = ["Architecture"]

        elif field == "Maths":
            st.subheader("Maths Detailed Questionnaire")
            if "maths_results" not in st.session_state:
                st.session_state.maths_results = {}

            for q, progs in maths_questions.items():
                ans = st.radio(q, progs, key=f"maths_{q}")
                st.session_state.maths_results[ans] = st.session_state.maths_results.get(ans, 0) + 1

            if st.button("Submit Maths Questionnaire"):
                max_prog = max(st.session_state.maths_results.values())
                final_recommendations = [p for p, v in st.session_state.maths_results.items() if v == max_prog]

        elif field == "Engineering":
            st.subheader("Engineering Detailed Questionnaire")
            if "eng_results" not in st.session_state:
                st.session_state.eng_results = {}

            for q, progs in engineering_questions.items():
                ans = st.radio(q, progs, key=f"eng_{q}")
                st.session_state.eng_results[ans] = st.session_state.eng_results.get(ans, 0) + 1

            if st.button("Submit Engineering Questionnaire"):
                max_prog = max(st.session_state.eng_results.values())
                final_recommendations = [p for p, v in st.session_state.eng_results.items() if v == max_prog]

        # ✅ Intersect with Top-5 predicted
        if final_recommendations:
            filtered = [p for p in final_recommendations if p in st.session_state["top_predicted"]]
            if filtered:
                st.success(f"🎯 Final Recommended Programme(s): {', '.join(filtered)}")
            else:
                st.warning("No overlap between academic results and interests. Please review your grades or answers.")
