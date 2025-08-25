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
            "Analyzing data to make smart financial decisions": "Maths",
            "Designing or improving machines, bridges or electronics": "Engineering",
            "Building useful apps or software": "Software Engineering",
            "Designing homes, buildings or creative spaces": "Architecture"
        }
    },
    {
        "question": "Q6: Which topics interest you the most?",
        "options": {
            "Money, risk, statistics and patterns": "Maths",
            "Machines, technology or how things are built": "Engineering",
            "Apps, games or how websites work": "Software Engineering",
            "Art, design or how spaces look and feel": "Architecture"
        }
    },
    {
        "question": "Q7: Which project would you most enjoy?",
        "options": {
            "Solving a complex mathematical model": "Maths",
            "Designing and testing a new robot": "Engineering",
            "Creating a mobile app or website": "Software Engineering",
            "Creating a model of a house or a park": "Architecture"
        }
    },
    {
        "question": "Q8: Which type of problem do you prefer solving?",
        "options": {
            "Figuring out the best deal or investment": "Maths",
            "Finding smart ways to make machines, systems or structures work better": "Engineering",
            "Troubleshooting computer code": "Software Engineering",
            "Making a space look both useful and beautiful": "Architecture"
        }
    },
    {
        "question": "Q9: Which would you most like to learn more about?",
        "options": {
            "How insurance, banks, or investments use maths": "Maths",
            "How engines, circuits or materials work": "Engineering",
            "How to build your own app or website": "Software Engineering",
            "How to design a beautiful and functional building": "Architecture"
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

    "Q2: What kind of problems do you enjoy solving most?": {
        "Figuring out how machines or tools can help treat patients better": "Bachelor of Biomedical Engineering with Honours",
        "Finding ways to make products like plastic or fuel in a faster or cheaper way": "Bachelor of Chemical Engineering with Honours",
        "Solving how to make buildings safer during earthquakes or bad weather": "Bachelor of Civil Engineering with Honours",
        "Working on how to send electricity to homes and buildings without losing power": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Creating new materials that are super light, strong, or heat-resistant for special uses": "Bachelor of Materials Engineering with Honours",
        "Designing better systems to keep car engines or machines from overheating": "Bachelor of Mechanical Engineering with Honours",
        "Combining machines, sensors, and computer controls to build smart robots or gadgets": "Bachelor of Mechatronics Engineering with Honours",
        "Finding ways to make the internet or mobile networks faster and more stable": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q3: Which innovation would you be most excited to work on?": {
        "A wearable health tracker that can detect illness early": "Bachelor of Biomedical Engineering with Honours",
        "A plastic-free packaging that naturally breaks down in the environment": "Bachelor of Chemical Engineering with Honours",
        "A stadium with a roof that opens and closes automatically based on weather": "Bachelor of Civil Engineering with Honours",
        "A wireless charging road that powers electric cars while they drive": "Bachelor of Electrical and Electronic Engineering with Honours",
        "A super-lightweight bicycle frame that is strong but easy to carry": "Bachelor of Materials Engineering with Honours",
        "A drone that can fly longer and faster using a new engine design": "Bachelor of Mechanical Engineering with Honours",
        "A robot pet that responds to voice commands and can play games": "Bachelor of Mechatronics Engineering with Honours",
        "A phone app that uses satellites to give signal even in remote jungles": "Bachelor of Telecommunications Engineering with Honours",
    },

    "Q4: What motivates you the most about engineering?": {
        "Creating medical tools or devices that help people feel better": "Bachelor of Biomedical Engineering with Honours",
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
        "Wires, circuits and electrical components": "Bachelor of Electrical and Electronic Engineering with Honours",
        "Microscopes and machines for testing materials": "Bachelor of Materials Engineering with Honours",
        "Engines, turbines and mechanical parts": "Bachelor of Mechanical Engineering with Honours",
        "Robots, sensors and automation kits": "Bachelor of Mechatronics Engineering with Honours",
        "Satellites, antennas and communication devices": "Bachelor of Telecommunications Engineering with Honours",
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
        
            # Persist top-10 across reruns
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

# =========================================
# Final selection helpers
# =========================================
def pick_two(programmes):
    """Pick up to 2 programmes according to tie rules."""
    unique = list(dict.fromkeys(programmes))  # keep order, dedupe
    if len(unique) <= 2:
        return unique
    return random.sample(unique, 2)

# =========================================
# Questionnaire Flow (Degree only)
# =========================================
if "top_predicted" in st.session_state:
    st.header("General Interest Questionnaire")

    # General questionnaire (stable radios) only if not yet finalized/answered
    if "general_scores" not in st.session_state and "general_winners" not in st.session_state:
        scores = {"Maths": 0, "Engineering": 0, "Software Engineering": 0, "Architecture": 0}

        for idx, item in enumerate(general_questions):
            q = item["question"]
            options_map = item["options"]

            # Shuffle once and keep stable
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

            st.session_state.general_winners = winners
            st.session_state.general_scores = scores

            # If the tie is only Architecture/Software Engineering → output directly (no detailed step)
            if all(w in ["Architecture", "Software Engineering"] for w in winners):
                chosen = pick_two(winners)
                st.session_state.final_general = chosen
                st.success(f"General Recommendation: {', '.join(chosen)}")

            # If Maths/Engineering are present in the winners → proceed to detailed questionnaire
            elif any(w in ["Maths", "Engineering"] for w in winners):
                # Store all winners; we’ll ask detailed for any of Maths/Engineering present
                st.session_state.field = winners
                st.success(f"Proceeding to detailed questionnaire for: {', '.join(winners)}")

            # Single winner Architecture or Software Engineering
            else:
                st.session_state.final_general = winners
                st.success(f"General Recommendation: {', '.join(winners)}")

    # Detailed Questionnaire Stage (only if needed)
    if "field" in st.session_state:
        chosen_fields = st.session_state.field

        # Ask Maths detailed if present
        if "Maths" in chosen_fields and "maths_done" not in st.session_state:
            st.subheader("Maths Detailed Questionnaire")
            maths_results = run_detailed_questionnaire(maths_questions, "maths")
            if st.button("Submit Maths Questionnaire"):
                max_val = max(maths_results.values())
                winners = [p for p, v in maths_results.items() if v == max_val]
                st.session_state.maths_detail = pick_two(winners)
                st.session_state.maths_done = True
                st.success(f"Maths focus: {', '.join(st.session_state.maths_detail)}")

        # Ask Engineering detailed if present
        if "Engineering" in chosen_fields and "eng_done" not in st.session_state:
            st.subheader("Engineering Detailed Questionnaire")
            eng_results = run_detailed_questionnaire(engineering_questions, "eng")
            if st.button("Submit Engineering Questionnaire"):
                max_val = max(eng_results.values())
                winners = [p for p, v in eng_results.items() if v == max_val]
                st.session_state.eng_detail = pick_two(winners)
                st.session_state.eng_done = True
                st.success(f"Engineering focus: {', '.join(st.session_state.eng_detail)}")

        # When all needed detailed sections are answered, compute final recommendations
        need_maths = "Maths" in chosen_fields
        need_eng = "Engineering" in chosen_fields
        maths_ready = (not need_maths) or ("maths_done" in st.session_state)
        eng_ready = (not need_eng) or ("eng_done" in st.session_state)

        if maths_ready and eng_ready and ("finalized" not in st.session_state):
            detailed_results = []
            if "maths_detail" in st.session_state:
                detailed_results.extend(st.session_state.maths_detail)
            if "eng_detail" in st.session_state:
                detailed_results.extend(st.session_state.eng_detail)

            final_recommendations = []

            # Only combine general non-detailed fields (Arch/SE) with detailed results
            # if the general winners contained a mix (Arch/SE together with Maths/Eng).
            if "general_winners" in st.session_state:
                gen_winners = st.session_state.general_winners
                mixed = any(g in ["Architecture", "Software Engineering"] for g in gen_winners) and \
                        any(g in ["Maths", "Engineering"] for g in gen_winners)
                if mixed:
                    non_detailed = [g for g in gen_winners if g in ["Architecture", "Software Engineering"]]
                    if non_detailed:
                        final_recommendations.extend(pick_two(non_detailed))
                    if detailed_results:
                        final_recommendations.extend(pick_two(detailed_results))
                    final_recommendations = pick_two(final_recommendations)  # cap to 2
                else:
                    # If winners are only Maths/Engineering, final comes from detailed results only
                    final_recommendations = pick_two(detailed_results)
            else:
                final_recommendations = pick_two(detailed_results)

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
                    # fallback to top-2 academics
                    top2 = st.session_state["top_predicted"][:2]
                    st.warning(
                        f"⚠️ Your answers do not overlap with academic prediction. "
                        f"Suggesting top academic matches instead: {', '.join(top2)}"
                    )
            st.session_state.finalized = True

    # If user ended with Architecture/Software Engineering only (no detailed step)
    if "final_general" in st.session_state and "finalized" not in st.session_state:
        finals = pick_two(st.session_state.final_general)

        normalized_top10 = [normalize_programme(p) for p in st.session_state["top_predicted"]]
        normalized_finals = [normalize_programme(p) for p in finals]

        filtered = [
            original for original in st.session_state["top_predicted"]
            if normalize_programme(original) in normalized_finals
        ]

        if filtered:
            st.success(f"🎯 Final Recommended Programme(s): {', '.join(pick_two(filtered))}")
        else:
            fallback = st.session_state["top_predicted"][:2]
            st.warning(
                "⚠️ Your answers do not overlap with academic prediction. "
                f"Suggesting top academic matches instead: {', '.join(fallback)}"
            )
        st.session_state.finalized = True
