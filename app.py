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
import random

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
@st.cache_resource
def load_easyocr():
    # Load EasyOCR once, cache across reruns
    return easyocr.Reader(["en"], gpu=False)

reader = load_easyocr()

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
# Questionnaire
# =========================================
general_questions = [
    {
        "question": "Q1: Which activity do you enjoy the most?",
        "options": {
            "Solving tricky math problems or puzzles": "Maths",
            "Figuring out how machines or systems work": "Engineering",
            "Creating things using a computer": "Software Engineering",
            "Drawing buildings or planning spaces": "Architecture"
        }
    },
    {
        "question": "Q2: Which type of task do you usually enjoy?",
        "options": {
            "Solving calculations or puzzles": "Maths",
            "Testing how machines, tools or systems work and recording the results": "Engineering",
            "Building something using software or code": "Software Engineering",
            "Drawing, crafting or designing": "Architecture"
        }
    },
    {
        "question": "Q3: What kind of schoolwork feels most satisfying?",
        "options": {
            "Getting the right answer in maths or logic problems": "Maths",
            "Doing experiments and recording observations": "Engineering",
            "Writing or debugging computer code": "Software Engineering",
            "Designing models or creative projects": "Architecture"
        }
    },
    {
        "question": "Q4: Which hobby sounds most fun to you?",
        "options": {
            "Solving Sudoku, chess or brain teasers": "Maths",
            "Fixing or assembling gadgets and machines": "Engineering",
            "Developing apps, websites or games": "Software Engineering",
            "Sketching, painting or 3D modelling": "Architecture"
        }
    },
    {
        "question": "Q5: Which kind of job would you enjoy more?",
        "options": {
            "Analysing data or working as a financial analyst": "Maths",
            "Designing or testing engines, bridges or circuits": "Engineering",
            "Becoming a software developer or IT specialist": "Software Engineering",
            "Becoming an architect or interior designer": "Architecture"
        }
    },
    {
        "question": "Q6: Which topics interest you the most?",
        "options": {
            "Numbers, patterns or statistics": "Maths",
            "Machines, electricity or construction": "Engineering",
            "Computers, coding or AI": "Software Engineering",
            "Art, space planning or creative design": "Architecture"
        }
    },
    {
        "question": "Q7: Which project would you most enjoy?",
        "options": {
            "Solving a complex mathematical model": "Maths",
            "Designing and testing a new robot": "Engineering",
            "Creating a mobile app or website": "Software Engineering",
            "Designing a new building layout": "Architecture"
        }
    },
    {
        "question": "Q8: Which type of problem do you prefer solving?",
        "options": {
            "Math puzzles, equations or logic": "Maths",
            "Fixing machines, systems or structures": "Engineering",
            "Troubleshooting computer code": "Software Engineering",
            "Design challenges in spaces or layouts": "Architecture"
        }
    },
    {
        "question": "Q9: Which would you most like to learn more about?",
        "options": {
            "Advanced mathematics and statistics": "Maths",
            "How engines, circuits or materials work": "Engineering",
            "Programming languages and AI": "Software Engineering",
            "Architectural design and modelling": "Architecture"
        }
    }
]

maths_questions = {
    "Q1: Which of the following sounds most interesting to you?": {
        "Solving real-world problems using maths, like planning the fastest way to deliver packages or understanding how diseases spread": "Bachelor of Science (Honours) Applied Mathematics with Computing",
        "Figuring out how to grow money, manage investments or understand how banks work": "Bachelor of Science (Honours) Financial Mathematics",
        "Using maths and statistics to predict risks, like how likely someone is to get sick or how long a machine will last": "Bachelor of Science (Honours) Actuarial Science",
        "Calculating building costs and making sure construction projects stay on budget": "Bachelor of Science (Honours) Quantity Surveying",
        "Studying how the universe works, like how planets move or how light behaves": "Bachelor of Science (Honours) Physics"
    },
    "Q2: What kind of problems do you enjoy solving most?": {
        "Figuring out how to plan the fastest delivery routes for food orders": "Bachelor of Science (Honours) Applied Mathematics with Computing",
        "Deciding how to split your monthly allowance to save and spend wisely": "Bachelor of Science (Honours) Financial Mathematics",
        "Calculating the chances of someone getting a loan based on their background": "Bachelor of Science (Honours) Actuarial Science",
        "Estimating how much it will cost to build a school and making sure it stays within budget": "Bachelor of Science (Honours) Quantity Surveying",
        "Solving puzzles about motion, energy, or forces in the physical world": "Bachelor of Science (Honours) Physics"
    },
    "Q3: Which innovation would you be most excited to work on?": {
        "A system that helps a city manage traffic using math models": "Bachelor of Science (Honours) Applied Mathematics with Computing",
        "A mobile app that tracks personal spending and predicts savings growth": "Bachelor of Science (Honours) Financial Mathematics",
        "A tool that helps insurance companies estimate accident risks accurately": "Bachelor of Science (Honours) Actuarial Science",
        "A system that calculates the total cost for building a house, including labour and materials": "Bachelor of Science (Honours) Quantity Surveying",
        "A telescope system that can detect planets outside our solar system": "Bachelor of Science (Honours) Physics"
    },
    "Q4: Which type of task would you enjoy most?": {
        "Figuring out how to calculate the best angle to kick a ball for a perfect goal": "Bachelor of Science (Honours) Applied Mathematics with Computing",
        "Planning how to save and grow money for a big purchase like a car or house": "Bachelor of Science (Honours) Financial Mathematics",
        "Estimating the chances of someone getting sick based on their habits": "Bachelor of Science (Honours) Actuarial Science",
        "Calculating the cost of building a new school and making sure it doesn’t go over budget": "Bachelor of Science (Honours) Quantity Surveying",
        "Designing an experiment to measure the speed of light": "Bachelor of Science (Honours) Physics"
    },
    "Q5: What motivates you most about maths?": {
        "Using maths to solve real-world problems like traffic jams or population growth": "Bachelor of Science (Honours) Applied Mathematics with Computing",
        "Helping people make smarter money decisions and manage finances": "Bachelor of Science (Honours) Financial Mathematics",
        "Predicting future risks like accidents or illness using data and statistics": "Bachelor of Science (Honours) Actuarial Science",
        "Making sure construction projects stay on budget and are cost-efficient": "Bachelor of Science (Honours) Quantity Surveying",
        "Understanding the laws of nature, from tiny particles to the universe itself": "Bachelor of Science (Honours) Physics"
    }
}

engineering_questions = {
    "Q1: Which of these jobs sounds the most exciting to you?": {
        "Making machines that help doctors save lives": "Bachelor of Biomedical Engineering with Honours",
        "Creating useful liquids like shampoo or glue": "Bachelor of Chemical Engineering with Honours",
        "Designing bridges, roads or tall buildings": "Bachelor of Civil Engineering with Honours",
        "Building circuits, fixing electronics or working with electricity": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Making phone screens that don’t break easily when dropped": "Bachelor of Materials Engineering with Honours",
        "Redesigning a motorbike engine so it uses less fuel but still runs fast and smooth": "Bachelor of Mechanical Engineering with Honours",
        "Making smart robots that can move or do tasks": "Bachelor of Mechatronics Engineering with Honours",
        "Helping people stay connected through phones and the internet": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q2: What school subject do you enjoy the most?": {
        "Biology – learning about the human body": "Bachelor of Biomedical Engineering with Honours",
        "Chemistry – mixing and experimenting with chemicals": "Bachelor of Chemical Engineering with Honours",
        "Mathematics – calculating and solving problems": "Bachelor of Civil Engineering with Honours",
        "Physics – electricity, circuits, and energy": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Science of materials – metals, plastics, and how they are used": "Bachelor of Materials Engineering with Honours",
        "Mechanics – motion, forces, and machines": "Bachelor of Mechanical Engineering with Honours",
        "Technology – robots, sensors, and automation": "Bachelor of Mechatronics Engineering with Honours",
        "Communication – signals, networks, and data": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q3: Which project would you prefer to work on?": {
        "Designing an artificial arm for someone who lost theirs": "Bachelor of Biomedical Engineering with Honours",
        "Finding a way to make fuel from waste cooking oil": "Bachelor of Chemical Engineering with Honours",
        "Planning a new highway for a busy city": "Bachelor of Civil Engineering with Honours",
        "Developing a solar-powered phone charger": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Creating stronger but lighter airplane parts": "Bachelor of Materials Engineering with Honours",
        "Building a faster and safer car engine": "Bachelor of Mechanical Engineering with Honours",
        "Designing a robot vacuum cleaner that avoids obstacles": "Bachelor of Mechatronics Engineering with Honours",
        "Improving 5G internet connection for rural areas": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q4: What motivates you the most about engineering?": {
        "Helping doctors treat patients with new technology": "Bachelor of Biomedical Engineering with Honours",
        "Solving pollution or energy problems with science": "Bachelor of Chemical Engineering with Honours",
        "Designing safe and useful buildings for people": "Bachelor of Civil Engineering with Honours",
        "Making homes and cities smarter with electricity": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Finding new materials to improve products like phones or clothes": "Bachelor of Materials Engineering with Honours",
        "Designing powerful machines that improve transportation": "Bachelor of Mechanical Engineering with Honours",
        "Making robots that help humans with everyday tasks": "Bachelor of Mechatronics Engineering with Honours",
        "Keeping the world connected through communication systems": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q5: Which tool or equipment would you most like to work with?": {
        "MRI scanners and prosthetic devices": "Bachelor of Biomedical Engineering with Honours",
        "Laboratory equipment for chemical experiments": "Bachelor of Chemical Engineering with Honours",
        "Surveying and construction tools": "Bachelor of Civil Engineering with Honours",
        "Wires, circuits, and electrical components": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Microscopes and machines for testing materials": "Bachelor of Materials Engineering with Honours",
        "Engines, turbines, and mechanical parts": "Bachelor of Mechanical Engineering with Honours",
        "Robots, sensors, and automation kits": "Bachelor of Mechatronics Engineering with Honours",
        "Satellites, antennas, and communication devices": "Bachelor of Telecommunications Engineering with Honours",
    }
}

# =========================================
# Helper for running questionnaire
# =========================================
def run_detailed_questionnaire(questions, key_prefix):
    results = {}
    for idx, (q, options) in enumerate(questions.items()):
        # ✅ Shuffle only once and store in session_state
        if f"{key_prefix}_options_{idx}" not in st.session_state:
            shuffled = list(options.keys())
            random.shuffle(shuffled)
            st.session_state[f"{key_prefix}_options_{idx}"] = shuffled
        else:
            shuffled = st.session_state[f"{key_prefix}_options_{idx}"]

        ans = st.radio(q, shuffled, key=f"{key_prefix}_{idx}")
        chosen = options[ans]
        results[chosen] = results.get(chosen, 0) + 1
    return results

# =========================================
# Streamlit App
# =========================================
st.title("🎓 UTAR Programme Recommendation System")
option = st.radio("Choose Recommendation Type:", ["Foundation", "Degree Programme"])

programme_mapping = {
    'FOUNDATION IN ARTS Arts & Social Science stream (Stream X)': 0,
    'FOUNDATION IN ARTS Management and Accountancy stream (Stream Y)': 1,
    'FOUNDATION IN SCIENCE Biological Science stream (Stream S)': 2,
    'FOUNDATION IN SCIENCE Physical Science stream (Stream P)': 3
}

# Reverse mapping: number → name
programme_reverse_mapping = {v: k for k, v in programme_mapping.items()}

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
        
            # Map numbers back to names
            top2_progs = [programme_reverse_mapping[idx] for idx in top2_idx]
        
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
        
            # Store in session_state so it persists across reruns
            st.session_state["top_predicted"] = top10
        
        # ✅ Always show top-10 if available
        if "top_predicted" in st.session_state:
            st.info("📊 Top 10 Academic-based Recommendations:\n\n" + 
                    "\n".join([f"{i+1}. {p}" for i, p in enumerate(st.session_state['top_predicted'])]))

# =========================================
# Normalization helper for programme names
# =========================================
def normalize_programme(name: str) -> str:
    """
    Normalize programme names by removing degree prefixes/suffixes
    and converting to lowercase for comparison.
    """
    name = name.lower()
    name = name.replace("bachelor of ", "")
    name = name.replace("with honours", "")
    name = name.replace("(honours)", "")
    name = name.replace("honours", "")
    return name.strip()

# === Questionnaire stage ===
if "top_predicted" in st.session_state:
    st.header("General Interest Questionnaire")

    # Run general questionnaire only if we haven't picked a field yet
    if "field" not in st.session_state:
        scores = {"Maths": 0, "Engineering": 0, "Software Engineering": 0, "Architecture": 0}
    
        for idx, item in enumerate(general_questions):
            q = item["question"]
            options_map = item["options"]
    
            # ✅ Shuffle once and store in session_state
            if f"options_{idx}" not in st.session_state:
                shuffled = list(options_map.keys())
                random.shuffle(shuffled)
                st.session_state[f"options_{idx}"] = shuffled
            else:
                shuffled = st.session_state[f"options_{idx}"]
    
            ans = st.radio(q, shuffled, key=f"general_{idx}")
            chosen_field = options_map[ans]
            scores[chosen_field] += 1

        if st.button("Submit General Questionnaire"):
            max_score = max(scores.values())
            winners = [k for k, v in scores.items() if v == max_score]

            if len(winners) > 1:
                st.warning(f"Tie detected! Possible fields: {', '.join(winners)}")
                pick = st.radio("Pick one field to continue with:", winners, key="tie_pick")
                if st.button("Continue with selected field"):
                    st.session_state.field = pick
                    st.success(f"Continuing with: {pick}")
            else:
                st.session_state.field = winners[0]
                st.success(f"Your strongest interest field: {st.session_state.field}")

    # --- Follow-up questionnaire (single chosen field) ---
    if "field" in st.session_state:
        field = st.session_state.field
        final_recommendations = []

        if field == "Software Engineering":
            final_recommendations = ["Software Engineering"]

        elif field == "Architecture":
            final_recommendations = ["Architecture"]

        elif field == "Maths":
            st.subheader("Maths Detailed Questionnaire")
            maths_results = run_detailed_questionnaire(maths_questions, "maths")
            if st.button("Submit Maths Questionnaire"):
                max_val = max(maths_results.values())
                final_recommendations = [p for p, v in maths_results.items() if v == max_val]

        elif field == "Engineering":
            st.subheader("Engineering Detailed Questionnaire")
            eng_results = run_detailed_questionnaire(engineering_questions, "eng")
            if st.button("Submit Engineering Questionnaire"):
                max_val = max(eng_results.values())
                final_recommendations = [p for p, v in eng_results.items() if v == max_val]

        # ✅ Intersect with Top-10 predicted (with normalization)
        if final_recommendations:
            normalized_top10 = [normalize_programme(p) for p in st.session_state["top_predicted"]]
            normalized_finals = [normalize_programme(p) for p in final_recommendations]

            # keep original names for display
            filtered = [
                original for original in st.session_state["top_predicted"]
                if normalize_programme(original) in normalized_finals
            ]

            if filtered:
                st.success(f"🎯 Final Recommended Programme(s): {', '.join(filtered)}")
            else:
                st.warning(
                    "No direct overlap between academic results and interests. "
                    "But don’t worry — you can adjust grades above or explore other fields in the questionnaire."
                )
