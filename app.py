import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU init on Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import pdfplumber
import cv2
import re
import random
from io import BytesIO

# Fuzzy matching for noisy OCR subject names
from rapidfuzz import fuzz, process

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

def get_top_n_programmes(model, X_new, encoder, n=10):
    pred_probs = model.predict(X_new, verbose=0)[0]
    top_idx = np.argsort(pred_probs)[::-1][:n]
    return encoder.inverse_transform(top_idx)

# =========================================
# OCR helpers (Tesseract + OpenCV + fuzzy matching)
# =========================================

# Subject aliases -> canonical subject name
subject_aliases = {
    "bahasa inggeris": "English",
    "BAHASAINGGERIS": "English",
    "english": "English",
    "b. inggeris": "English",
    "bi": "English",

    "accounting": "Accounting",
    "akaun": "Accounting",
    "accounts": "Accounting",
    
    "additional mathematics": "Additional Mathematics",
    "add maths": "Additional Mathematics",
    "additional mathematic": "Additional Mathematics",
    "matematik tambahan": "Additional Mathematics",
    
    "mathematics": "Mathematics",
    "mathematik": "Mathematics",
    "maths": "Mathematics",
    "mathematic": "Mathematics",
    "matematik": "Mathematics",

    "science": "Science",
    "sains": "Science",
    "physics": "Physics",
    "fizik": "Physics",
    "chemistry": "Chemistry",
    "kimia": "Chemistry",
    "biology": "Biology",
    "biologi": "Biology",

    "bahasa cina": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",

    "pendidikan seni": "Art",
    "art": "Art",
    "seni": "Art",

    "ict": "ICT",
    "information and communication technology": "ICT",

    "technology": "Technology",
    "advanced mathematics i": "Advanced Mathematics I",
    "advanced mathematics 1": "Advanced Mathematics I",
    "advanced mathematics ii": "Advanced Mathematics II",
    "advanced mathematics 2": "Advanced Mathematics II",
}

# SPM-style keywords → canonical grade (checked first in the substring to the right of subject)
grade_keywords = {
    # UEC
    "A1": "A",
    "A2": "B", "B3": "B",
    "B4": "C", "B5": "C",
    "B6": "D", "C7": "E", "C8": "E", "F9": "F",
    # SPM
    "CEMERLANG TERTINGGI": "A+",
    "CEMERLANG TINGGI": "A",
    "CEMERLANG": "A-",
    "KEPUJIAN TERTINGGI": "B+",
    "KEPUJIAN TINGGI": "B",
    "KEPUJIAN ATAS": "C+",
    "KEPUJIAN": "C",
    "LULUS ATAS": "D",
    "LULUS": "E",
    "GAGAL": "F",
}

# Keep a robust grade regex that captures +/- when present
GRADE_AFTER_SUBJ_RE = re.compile(r"\b([A-F](?:\s*\+|\s*\-)?)(?![A-Z])", re.IGNORECASE)

# Match A+, A, A-, B+, B, B-, ..., F (with optional spaces/newline in between)
grade_pattern = re.compile(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E|F)\b", re.IGNORECASE)

# More forgiving: captures grades even if extra spaces/newlines inside
grade_like_but_messy = re.compile(
    r"(A\s*\+|A\s*-?|B\s*\+|B\s*-?|C\s*\+|C\s*-?|D\s*\+|D\s*-?|E|F)",
    re.IGNORECASE
)

@st.cache_resource
def load_doctr_model():
    return ocr_predictor(pretrained=True)

doctr_model = load_doctr_model()

def doctr_extract_lines(pil_img):
    """Run docTR OCR on a PIL image and return line-wise text.

    docTR's DocumentFile.from_images expects a list of numpy arrays or file paths,
    not a raw PIL.Image, so we convert the PIL image to a numpy array and pass
    it as a single-item list.
    """
    import numpy as _np
    import tempfile
    try:
        # Convert PIL -> numpy and pass as a list
        img_array = _np.array(pil_img)
        doc = DocumentFile.from_images([img_array])
        result = doctr_model(doc)
        json_out = result.export()
    except Exception:
        # Fallback: save to a temp file and read from path (more robust on some envs)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name)
                doc = DocumentFile.from_images(tmp.name)
                result = doctr_model(doc)
                json_out = result.export()
        except Exception as e:
            # Give up gracefully and return empty text
            st.warning(f"docTR OCR failed: {e}")
            return ""

    # Collect line texts from exported structure
    lines = []
    for page in json_out.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                text = " ".join([w.get("value", "") for w in line.get("words", [])])
                if text.strip():
                    lines.append(text.strip())
    return "\n".join(lines)

def normalize_str(s: str) -> str:
    return re.sub(r"[^a-z0-9+\- ]", " ", s.lower()).strip()

def deskew_image(gray):
    # Try Tesseract OSD; if it fails, return as-is
    try:
        osd = pytesseract.image_to_osd(gray)
        angle_match = re.search(r'Rotate: (\d+)', osd)
        if angle_match:
            angle = int(angle_match.group(1)) % 360
            if angle != 0:
                (h, w) = gray.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return gray

def preprocess_for_ocr(pil_img: Image.Image, high_contrast=False) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    # Scale up small images
    h, w = img.shape[:2]
    if min(h, w) < 900:
        scale = int(900 / min(h, w))
        img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = deskew_image(gray)

    if high_contrast:
        # Strong binarization + denoise for faint prints
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 15)
        gray = cv2.medianBlur(gray, 3)
    else:
        # Mild clean-up
        gray = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def combine_tokens_to_lines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["page_num","block_num","par_num","line_num","left","top","right","bottom","text"])
    # group by line and combine
    groups = df.groupby(["page_num","block_num","par_num","line_num"], as_index=False)
    lines = []
    for keys, g in groups:
        left = int(g["left"].min())
        top = int(g["top"].min())
        right = int((g["left"] + g["width"]).max())
        bottom = int((g["top"] + g["height"]).max())
        text = " ".join([t for t in g["text_norm"].tolist()])
        rec = {**dict(zip(["page_num","block_num","par_num","line_num"], keys)),
               "left": left, "top": top, "right": right, "bottom": bottom,
               "text": text, "text_norm": normalize_str(text)}
        lines.append(rec)
    return pd.DataFrame(lines)

def extract_text_from_pdf(file_bytes: bytes):
    # Try selectable text first
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            pages_text = []
            for p in pdf.pages:
                pages_text.append(p.extract_text() or "")
            text = "\n".join(pages_text).strip()
    except Exception:
        text = ""

    # If poor text or empty, rasterize pages and OCR them
    if (not text or len(text) < 50) and PDF2IMAGE_OK:
        try:
            images = convert_from_bytes(file_bytes, dpi=300)
            ocr_texts = []
            for im in images:
                df, lines, _ = ocr_image_two_pass(im)
                ocr_texts.append(" ".join(lines["text"].tolist()) if not lines.empty else "")
            text = "\n".join(ocr_texts).strip()
        except Exception:
            pass
    return text

def extract_tokens_from_image(uploaded_file):
    """Use docTR to extract line text from uploaded image file.

    Returns (full_text, token_df, line_df). docTR does not give a token_df
    in the same format as our Tesseract pipeline, so token_df/line_df are None.
    """
    pil_img = Image.open(uploaded_file).convert("RGB")
    text = doctr_extract_lines(pil_img)
    # docTR export contains word boxes if you later want to build line/token dfs.
    return text, None, None

def extract_text_from_file(uploaded_file):
    # Unified: for PDFs return text; for images return text + dfs
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        file_bytes = uploaded_file.read()
        text = extract_text_from_pdf(file_bytes)
        return text, None, None
    else:
        text, token_df, line_df = extract_tokens_from_image(uploaded_file)
        return text, token_df, line_df

# =========================================
# Grade parsing (layout-aware)
# =========================================
def find_grade_near_subject(line_df: pd.DataFrame, subject_alias: str, grade_regex=grade_pattern, fuzz_threshold=80):
    """
    Look for a grade on lines that likely mention subject_alias. Prefer grades
    that appear to the RIGHT of the subject occurrence (small local window).
    Works when you have Tesseract-style line_df.
    """
    if line_df is None or line_df.empty:
        return None

    subj_norm = normalize_str(subject_alias)
    best = None
    best_score = -1

    for _, row in line_df.iterrows():
        line = row["text"]
        line_norm = row.get("text_norm", normalize_str(line))
        score = fuzz.partial_ratio(subj_norm, line_norm)
        if score < fuzz_threshold:
            continue

        # try to locate the subject substring in the original line (case-insensitive)
        mo = re.search(re.escape(subject_alias), line, re.IGNORECASE)
        tail = line[mo.end():] if mo else line  # text to the right of subject (if found)

        # 1) Check SPM-style keyword phrases first (higher confidence)
        tail_upper = tail.upper()
        grade = None
        for k, v in grade_keywords.items():
            if k in tail_upper:
                grade = v
                break

        # 2) If no keyword, look for explicit grade token in the tail
        if not grade:
            m = GRADE_AFTER_SUBJ_RE.search(tail_upper)
            if m:
                grade = m.group(1).upper()

        # 3) Keep best-scoring candidate
        if grade:
            sc = score + (10 if grade_regex.search(tail) else 0)
            if sc > best_score:
                best = grade
                best_score = sc

    return best
    
def preprocess_lines(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merged = []
    skip_next = False

    for i in range(len(lines)):
        if skip_next:
            skip_next = False
            continue

        # If this line is subject and next line looks like a grade → merge
        if i + 1 < len(lines) and re.match(r"^[A-F][+-]?$", lines[i+1].strip().upper()):
            merged.append(f"{lines[i]} {lines[i+1]}")
            skip_next = True
        else:
            merged.append(lines[i])

    return merged
    
def parse_grades(text, mode="foundation", line_df=None, debug=True):
    subjects = foundation_subjects if mode == "foundation" else degree_subjects
    results = {}

    alias_map = {}
    for alias, canon in subject_aliases.items():
        alias_map.setdefault(canon, []).append(alias)

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    def log(msg):
        if debug:
            st.write(msg)

    for subj in subjects:
        found_grade = None
        matched_line = None

        # -------- Only exact alias match --------
        for alias in alias_map.get(subj, [subj.lower()]):
            alias_norm = normalize_str(alias)
            for i, ln in enumerate(lines):
                ln_norm = normalize_str(ln)
                if alias_norm in ln_norm:
                    # only check same line + next line
                    for j in range(0, 2):
                        if i + j >= len(lines):
                            break
                        tail = lines[i + j].upper()

                        grade = None
                        for k, v in grade_keywords.items():
                            if k in tail:
                                grade = v
                                break
                        if not grade:
                            m = GRADE_AFTER_SUBJ_RE.search(tail)
                            if m:
                                grade = m.group(1).replace(" ", "").upper()

                        if grade:
                            found_grade = grade
                            matched_line = tail
                            break
                    if found_grade:
                        break
            if found_grade:
                break

        results[subj] = found_grade if found_grade else "0"

        if found_grade:
            log(f"✅ {subj}: {found_grade} (EXACT) → from line: '{matched_line}'")
        else:
            log(f"❌ {subj}: no grade found")

    return results

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
        # Shuffle only once and store in session_state
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
        text, token_df, line_df = extract_text_from_file(uploaded_file)
        extracted_grades = parse_grades(text, mode="foundation", line_df=line_df)

        with st.expander("🔍 OCR Debug (optional)"):
            st.write("Raw text (first 2000 chars):")
            st.code((text or "")[:2000] + ("..." if text and len(text) > 2000 else ""))
            if token_df is not None and not token_df.empty:
                st.write("Token samples:")
                st.dataframe(token_df.head(20))

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
        text, token_df, line_df = extract_text_from_file(uploaded_file)
        extracted_grades = parse_grades(text, mode="degree", line_df=line_df)

        with st.expander("🔍 OCR Debug (optional)"):
            st.write("Raw text (first 600 chars):")
            st.code((text or "")[:600] + ("..." if text and len(text) > 600 else ""))
            if token_df is not None and not token_df.empty:
                st.write("Token samples:")
                st.dataframe(token_df.head(20))

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

    # --- First run: show questionnaire ---
    if "general_submitted" not in st.session_state:
        scores = {"Maths": 0, "Engineering": 0, "Software Engineering": 0, "Architecture": 0}

        for idx, item in enumerate(general_questions):
            q = item["question"]
            options_map = item["options"]

            # shuffle once per question
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
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            winners = [sorted_scores[0][0]]
            if len(sorted_scores) > 1:
                winners.append(sorted_scores[1][0])

            st.session_state.general_winners = winners
            st.session_state.general_scores = scores
            st.session_state.general_submitted = True   # lock answers

            # Handle outcome
            if all(w in ["Architecture", "Software Engineering"] for w in winners):
                st.session_state.final_general = winners
                st.success(f"General Recommendation: {', '.join(winners)}")

            elif any(w in ["Maths", "Engineering"] for w in winners):
                st.session_state.field = winners
                st.info(f"Proceeding to detailed questionnaire for: {', '.join(winners)}")

            else:
                st.session_state.final_general = winners
                st.success(f"General Recommendation: {', '.join(winners)}")

    # --- After submission: show locked version ---
    else:
        st.subheader("✅ General Questionnaire (Your Answers)")
        for idx, item in enumerate(general_questions):
            q = item["question"]
            options_map = item["options"]
            options_list = st.session_state[f"options_{idx}"]
            prev_answer = st.session_state.get(f"general_{idx}")
            if prev_answer:
                st.radio(q, options_list, index=options_list.index(prev_answer),
                         key=f"general_locked_{idx}", disabled=True)

        st.write("**Top Fields of Interest:** " + ", ".join(st.session_state.general_winners))

    # ===== Detailed Questionnaire Stage =====
    if "field" in st.session_state:
        chosen_fields = st.session_state.field

        # --- Maths detailed ---
        if "Maths" in chosen_fields and "maths_done" not in st.session_state:
            st.subheader("Maths Detailed Questionnaire")
            maths_results = run_detailed_questionnaire(maths_questions, "maths")
            if st.button("Submit Maths Questionnaire"):
                max_val = max(maths_results.values())
                winners = [p for p, v in maths_results.items() if v == max_val]
                st.session_state.maths_detail = pick_two(winners)
                st.session_state.maths_done = True
                st.success(f"Maths focus: {', '.join(st.session_state.maths_detail)}")

        # --- Engineering detailed ---
        if "Engineering" in chosen_fields and "eng_done" not in st.session_state:
            st.subheader("Engineering Detailed Questionnaire")
            eng_results = run_detailed_questionnaire(engineering_questions, "eng")
            if st.button("Submit Engineering Questionnaire"):
                max_val = max(eng_results.values())
                winners = [p for p, v in eng_results.items() if v == max_val]
                st.session_state.eng_detail = pick_two(winners)
                st.session_state.eng_done = True
                st.success(f"Engineering focus: {', '.join(st.session_state.eng_detail)}")

        # --- Finalize only when both ready ---
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

            # merge Arch/SE with detailed if mixed
            if "general_winners" in st.session_state:
                gen_winners = st.session_state.general_winners
                mixed = any(g in ["Architecture", "Software Engineering"] for g in gen_winners) and \
                        any(g in ["Maths", "Engineering"] for g in gen_winners)
                if mixed:
                    non_detailed = [g for g in gen_winners if g in ["Architecture", "Software Engineering"]]
                    final_recommendations.extend(non_detailed)
                    final_recommendations.extend(detailed_results)
                else:
                    final_recommendations.extend(detailed_results)
            else:
                final_recommendations.extend(detailed_results)

            final_recommendations = pick_two(final_recommendations)

            if final_recommendations:
                st.success(f"🎯 Final Recommended Programme(s): {', '.join(final_recommendations)}")
            else:
                fallback = st.session_state["top_predicted"][:2]
                st.warning(
                    f"⚠️ Your answers do not overlap with academic prediction. "
                    f"Suggesting top academic matches instead: {', '.join(fallback)}"
                )
            st.session_state.finalized = True

    # ===== Arch/SE only (no detailed needed) =====
    if "final_general" in st.session_state and "finalized" not in st.session_state:
        finals = pick_two(st.session_state.final_general)
        st.success(f"🎯 Final Recommended Programme(s): {', '.join(finals)}")
        st.session_state.finalized = True

# =========================================
# 🔄 Run Again Button (reset + scroll top)
# =========================================
def reset_all():
    # clear questionnaire + predictions
    keys_to_clear = [k for k in st.session_state.keys()]
    for k in keys_to_clear:
        del st.session_state[k]
    st.toast("Recommendation system has been reset!", icon="🔄")

if st.button("🔄 Run Again"):
    reset_all()
    st.rerun()
