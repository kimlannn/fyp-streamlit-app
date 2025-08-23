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
        "Solving real-world problems using maths, like planning the fastest way to deliver packages or understanding how diseases spread": "Applied Mathematics",
        "Figuring out how to grow money, manage investments or understand how banks work": "Financial Mathematics",
        "Using maths and statistics to predict risks, like how likely someone is to get sick or how long a machine will last": "Actuarial Science",
        "Calculating building costs and making sure construction projects stay on budget": "Quantity Surveying",
        "Studying how the universe works, like how planets move or how light behaves": "Physics"
    },
    "Q2: What kind of problems do you enjoy solving most?": {
        "Figuring out how to plan the fastest delivery routes for food orders": "Applied Mathematics",
        "Deciding how to split your monthly allowance to save and spend wisely": "Financial Mathematics",
        "Calculating the chances of someone getting a loan based on their background": "Actuarial Science",
        "Estimating how much it will cost to build a school and making sure it stays within budget": "Quantity Surveying",
        "Solving puzzles about motion, energy, or forces in the physical world": "Physics"
    },
    "Q3: Which innovation would you be most excited to work on?": {
        "A system that helps a city manage traffic using math models": "Applied Mathematics",
        "A mobile app that tracks personal spending and predicts savings growth": "Financial Mathematics",
        "A tool that helps insurance companies estimate accident risks accurately": "Actuarial Science",
        "A system that calculates the total cost for building a house, including labour and materials": "Quantity Surveying",
        "A telescope system that can detect planets outside our solar system": "Physics"
    },
    "Q4: Which type of task would you enjoy most?": {
        "Figuring out how to calculate the best angle to kick a ball for a perfect goal": "Applied Mathematics",
        "Planning how to save and grow money for a big purchase like a car or house": "Financial Mathematics",
        "Estimating the chances of someone getting sick based on their habits": "Actuarial Science",
        "Calculating the cost of building a new school and making sure it doesn’t go over budget": "Quantity Surveying",
        "Designing an experiment to measure the speed of light": "Physics"
    },
    "Q5: What motivates you most about maths?": {
        "Using maths to solve real-world problems like traffic jams or population growth": "Applied Mathematics",
        "Helping people make smarter money decisions and manage finances": "Financial Mathematics",
        "Predicting future risks like accidents or illness using data and statistics": "Actuarial Science",
        "Making sure construction projects stay on budget and are cost-efficient": "Quantity Surveying",
        "Understanding the laws of nature, from tiny particles to the universe itself": "Physics"
    }
}

engineering_questions = {
    "Q1: Which of these jobs sounds the most exciting to you?": {
        "Making machines that help doctors save lives": "Biomedical Engineering",
        "Creating useful liquids like shampoo or glue": "Chemical Engineering",
        "Designing bridges, roads or tall buildings": "Civil Engineering",
        "Building circuits, fixing electronics or working with electricity": "Electrical & Electronic Engineering",
        "Making phone screens that don’t break easily when dropped": "Materials Engineering",
        "Redesigning a motorbike engine so it uses less fuel but still runs fast and smooth": "Mechanical Engineering",
        "Making smart robots that can move or do tasks": "Mechatronics Engineering",
        "Helping people stay connected through phones and the internet": "Telecommunications Engineering",
    },
    "Q2: What kind of problems do you enjoy solving most?": {
        "Figuring out how machines or tools can help treat patients better": "Biomedical Engineering",
        "Finding ways to make products like plastic or fuel in a faster or cheaper way": "Chemical Engineering",
        "Solving how to make buildings safer during earthquakes or bad weather": "Civil Engineering",
        "Working on how to send electricity to homes and buildings without losing power": "Electrical & Electronic Engineering",
        "Creating new materials that are super light, strong, or heat-resistant for special uses": "Materials Engineering",
        "Designing better systems to keep car engines or machines from overheating": "Mechanical Engineering",
        "Combining machines, sensors, and computer controls to build smart robots or gadgets": "Mechatronics Engineering",
        "Finding ways to make the internet or mobile networks faster and more stable": "Telecommunications Engineering",
    },
    "Q3: Which innovation would you be most excited to work on?": {
        "A wearable health tracker that can detect illness early": "Biomedical Engineering",
        "A plastic-free packaging that naturally breaks down in the environment": "Chemical Engineering",
        "A stadium with a roof that opens and closes automatically based on weather": "Civil Engineering",
        "A wireless charging road that powers electric cars while they drive": "Electrical & Electronic Engineering",
        "A super-lightweight bicycle frame that is strong but easy to carry": "Materials Engineering",
        "A drone that can fly longer and faster using a new engine design": "Mechanical Engineering",
        "A robot pet that responds to voice commands and can play games": "Mechatronics Engineering",
        "A phone app that uses satellites to give signal even in remote jungles": "Telecommunications Engineering",
    },
    "Q4: Which task would you most enjoy?": {
        "Testing how the human body reacts to new medical devices": "Biomedical Engineering",
        "Mixing and observing chemical reactions in a science lab": "Chemical Engineering",
        "Designing a safe and stable bridge for a busy road": "Civil Engineering",
        "Designing lights that turn on or off depending on how people use a room": "Electrical & Electronic Engineering",
        "Studying how and when airplane parts start to wear out": "Materials Engineering",
        "Putting together a small model of a car engine to see how it works and moves": "Mechanical Engineering",
        "Making an automatic hand sanitizer dispenser using sensors": "Mechatronics Engineering",
        "Fixing problems in a mobile network tower so people can get signal": "Telecommunications Engineering",
    },
    "Q5: What motivates you most about engineering?": {
        "Creating medical tools or devices that help people feel better": "Biomedical Engineering",
        "Using science to solve pollution or energy problems": "Chemical Engineering",  # or Civil if you want dual
        "Designing buildings or roads that make life easier for others": "Civil Engineering",
        "Building smart and energy-saving electrical systems": "Electrical & Electronic Engineering",
        "Finding new materials to improve everyday products like phones or clothes": "Materials Engineering",
        "Making engines and machines work better and faster": "Mechanical Engineering",
        "Mixing machines and coding to build things that work automatically": "Mechatronics Engineering",
        "Improving how we connect through the internet or phone networks": "Telecommunications Engineering",
    }
}

# =========================================
# Helper for running questionnaire
# =========================================
def run_detailed_questionnaire(questions, key_prefix):
    results = {}
    for q, options in questions.items():
        shuffled = list(options.keys())
        random.shuffle(shuffled)
        ans = st.radio(q, shuffled, key=f"{key_prefix}_{q}")
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

# === Questionnaire stage ===
if "top_predicted" in st.session_state:
    st.header("General Interest Questionnaire")

    # Run general questionnaire only if we haven't picked a field yet
    if "field" not in st.session_state:
        scores = {"Maths": 0, "Engineering": 0, "Software Engineering": 0, "Architecture": 0}

        # general_questions is a list of dicts; access by keys
        for item in general_questions:
            q = item["question"]
            options_map = item["options"]
            opts = list(options_map.keys())
            random.shuffle(opts)
            ans = st.radio(q, opts, key=f"general_{q}")
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

        # ✅ Intersect with Top-10 predicted
        if final_recommendations:
            filtered = [p for p in final_recommendations if p in st.session_state["top_predicted"]]
            if filtered:
                st.success(f"🎯 Final Recommended Programme(s): {', '.join(filtered)}")
            else:
                st.warning(
                    "No overlap between academic results and interests. "
                    "You can adjust grades above or pick another field in the tie step."
                )

