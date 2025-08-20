import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import pdfplumber
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# ========================
# Load models & tools
# ========================
foundation_model = load_model("foundation_model.h5")
degree_model = load_model("degree_model.h5")

with open("foundation_scaler.pkl", "rb") as f:
    foundation_scaler = pickle.load(f)

with open("degree_scaler.pkl", "rb") as f:
    degree_scaler = pickle.load(f)

# Subject list (must match training order!)
subject_list = [
    "Mathematics", "English", "Science", "Physics", "Chemistry", "Biology",
    "Additional Mathematics", "Accounting", "Chinese", "Art", "Pendidikan Seni"
]

# Grade mapping
grade_mapping = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0,
    "E": 0.7, "F": 0.0
}

# Example programme labels
foundation_programmes = ["Foundation in Science", "Foundation in Arts", "Foundation in IT"]
degree_programmes = ["Mechanical Engineering", "Civil Engineering", "Computer Science", "Accounting"]

# ========================
# Helper functions
# ========================

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

def parse_grades(text):
    results = {subj: 0.0 for subj in subject_list}
    for subj in subject_list:
        for grade in grade_mapping.keys():
            if subj.lower() in text.lower() and grade in text:
                results[subj] = grade_mapping[grade]
    return results

# ========================
# Streamlit App
# ========================

st.title("🎓 Programme Recommendation System")
mode = st.radio("Choose recommendation type:", ["Foundation Programme", "Degree Programme"])

uploaded_file = st.file_uploader("Upload your academic result (PDF/Image)", type=["pdf","png","jpg","jpeg"])

if uploaded_file:
    # Step 1: OCR
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    st.subheader("📄 Extracted Text")
    st.text_area("OCR Output", text, height=200)

    # Step 2: Extract grades
    grades = parse_grades(text)
    st.subheader("📝 Extracted Grades")
    st.write(grades)

    # Step 3: Prepare features
    features = [grades[subj] for subj in subject_list]

    if mode == "Foundation Programme":
        # Scale and predict foundation
        X = foundation_scaler.transform([features])
        prediction = foundation_model.predict(X)[0]
        top_indices = prediction.argsort()[-3:][::-1]

        st.subheader("🎯 Recommended Foundation Programmes")
        for idx in top_indices:
            st.write(f"- {foundation_programmes[idx]} (prob: {prediction[idx]:.2f})")

    elif mode == "Degree Programme":
        # Add questionnaire
        st.subheader("🧠 Interest Questionnaire")
        q1 = st.radio("I like solving technical problems:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])
        q2 = st.radio("I enjoy working with computers:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])
        q3 = st.radio("I want to help people through healthcare:", ["Strongly Agree", "Agree", "Neutral", "Disagree"])

        mapping = {"Strongly Agree": 3, "Agree": 2, "Neutral": 1, "Disagree": 0}
        questionnaire_scores = [mapping[q1], mapping[q2], mapping[q3]]

        # Combine academic + questionnaire
        full_features = features + questionnaire_scores
        X = degree_scaler.transform([full_features])

        # Predict single best degree
        prediction = degree_model.predict(X)[0]
        best_idx = np.argmax(prediction)

        st.subheader("🎯 Recommended Degree Programme")
        st.success(f"{degree_programmes[best_idx]} (prob: {prediction[best_idx]:.2f})")
