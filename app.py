import streamlit as st
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
import pdfplumber
import pytesseract
from PIL import Image

# ========================
# Load Models, Scalers, Encoders
# ========================
foundation_model = load_model("foundation_model.h5")
degree_model = load_model("degree_model.h5")

foundation_scaler = joblib.load("foundation_scaler.pkl")
degree_scaler = joblib.load("degree_scaler.pkl")

with open("foundation_encoder.pkl", "rb") as f:
    foundation_encoder = pickle.load(f)

with open("degree_encoder.pkl", "rb") as f:
    degree_encoder = pickle.load(f)

# ========================
# Subject lists (must match training order!)
# ========================
foundation_subjects = [
    "Mathematics", "English", "Science", "Physics", "Chemistry", "Biology",
    "Additional Mathematics", "Accounting", "Chinese", "Art", "Pendidikan Seni"
]

degree_subjects = [
    'Mathematics', 'Additional Mathematics', 'English', 'Physics',
    'Biology', 'Chemistry', 'ICT', 'Technology', 'Pendidikan Seni',
    'Advanced Mathematics I', 'Advanced Mathematics II'
]

# ========================
# Grade Mappings
# ========================
# Foundation (Ordinal A=6 to F=1)
foundation_grade_mapping = {
    "A": 6, "A+": 6, "A-": 6,
    "B": 5, "B+": 5, "B-": 5,
    "C": 4, "C+": 4, "C-": 4,
    "D": 3, "D+": 3, "D-": 3,
    "E": 2, "E+": 2, "E-": 2,
    "F": 1, "F+": 1, "F-": 1
}

# Degree (GPA 4.0 → 0.0)
degree_grade_mapping = {
    'A+': 4.0, 'A': 4.0, 'A-': 3.67,
    'B+': 3.33, 'B': 3.0, 'B-': 2.67,
    'C+': 2.33, 'C': 2.0, 'C-': 1.67,
    'D+': 1.33, 'D': 1.0,
    'E': 0.67, 'F': 0.0,
    0: 0
}

# ========================
# Helper functions
# ========================

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def parse_grades(text, subject_list, grade_mapping):
    """Extract subjects and grades from OCR text"""
    results = {subj: 0.0 for subj in subject_list}
    for subj in subject_list:
        for grade in grade_mapping.keys():
            if subj.lower() in text.lower() and str(grade) in text:
                results[subj] = grade_mapping[grade]
    return results

# ========================
# Streamlit UI
# ========================

st.title("🎓 Programme Recommendation System")

mode = st.radio("Choose recommendation type:", ["Foundation Programme", "Degree Programme"])

uploaded_file = st.file_uploader("Upload your academic result (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Step 1: OCR
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    st.subheader("📄 Extracted Text")
    st.text_area("OCR Output", text, height=200)

    if mode == "Foundation Programme":
        # Step 2: Extract grades
        grades = parse_grades(text, foundation_subjects, foundation_grade_mapping)
        st.subheader("📝 Extracted Grades")
        st.write(grades)

        # Step 3: Prepare features
        features = [grades[subj] for subj in foundation_subjects]
        X = foundation_scaler.transform([features])

        # Step 4: Predict
        prediction = foundation_model.predict(X)[0]
        top_indices = prediction.argsort()[-3:][::-1]

        st.subheader("🎯 Recommended Foundation Programmes")
        for idx in top_indices:
            prog_name = foundation_encoder.inverse_transform([idx])[0]
            st.write(f"- {prog_name} (prob: {prediction[idx]:.2f})")

    elif mode == "Degree Programme":
        # Step 2: Extract grades
        grades = parse_grades(text, degree_subjects, degree_grade_mapping)
        st.subheader("📝 Extracted Grades")
        st.write(grades)

        # Step 3: Questionnaire
        st.subheader("🧠 Interest Questionnaire")
        q1 = st.radio("I like solving technical problems:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])
        q2 = st.radio("I enjoy working with computers:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])
        q3 = st.radio("I want to help people through healthcare:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])

        mapping = {"Strongly Agree": 3, "Agree": 2, "Neutral": 1, "Disagree": 0}
        questionnaire_scores = [mapping[q1], mapping[q2], mapping[q3]]

        # Step 4: Prepare features
        features = [grades[subj] for subj in degree_subjects] + questionnaire_scores
        X = degree_scaler.transform([features])

        # Step 5: Predict
        prediction = degree_model.predict(X)[0]
        best_idx = np.argmax(prediction)
        prog_name = degree_encoder.inverse_transform([best_idx])[0]

        st.subheader("🎯 Recommended Degree Programme")
        st.success(f"{prog_name} (prob: {prediction[best_idx]:.2f})")
