# app.py
# MILTRANS: Context-Aware AI Translation Engine for Military SOPs
# Edit this file in Spyder, save as app.py, then run:  streamlit run app.py

import streamlit as st

# ---------------------------
# Streamlit Page Config - MUST BE FIRST
# ---------------------------
st.set_page_config(page_title="MILTRANS: Hindi → Multi‑Lang Translator", page_icon="🌐", layout="wide")

import os
import re
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from bs4 import BeautifulSoup

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import easyocr

from transformers import pipeline

try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.utils import which
    from audio_recorder_streamlit import audio_recorder
    AUDIO_IMPORTS_AVAILABLE = True
except ImportError:
    AUDIO_IMPORTS_AVAILABLE = False

from pymongo import MongoClient
from pymongo.server_api import ServerApi


# ---------------------------
# Configure pydub to use ffmpeg
# ---------------------------
AUDIO_ENABLED = False
if AUDIO_IMPORTS_AVAILABLE:
    ffmpeg_path = which("ffmpeg")
    AUDIO_ENABLED = ffmpeg_path is not None
    if AUDIO_ENABLED:
        AudioSegment.converter = ffmpeg_path
    else:
        st.sidebar.warning("⚠ FFmpeg not found. Audio features disabled.")
else:
    st.sidebar.warning("⚠ Audio libraries not installed. Audio features disabled.")


# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #000000, #6a0dad);
        color: #FFFFFF !important;
        font-family: 'Arial Black', sans-serif;
    }
    /* Make all text white */
    .stRadio > label, .stRadio > div, .stRadio label {
        color: white !important;
    }
    .stTextArea > label, .stTextInput > label {
        color: white !important;
        font-weight: bold !important;
    }
    .stMarkdown, .stMarkdown p {
        color: white !important;
    }
    /* Radio button text */
    .stRadio > div > label > div {
        color: white !important;
    }
    /* Input labels and text */
    label, .stSelectbox label, .stMultiSelect label {
        color: white !important;
    }
    /* Text input and textarea - black text for visibility */
    .stTextInput input, .stTextArea textarea {
        color: black !important;
        background-color: white !important;
    }
    /* Multiselect dropdown - black text for visibility */
    .stMultiSelect > div > div > div {
        color: black !important;
        background-color: white !important;
    }
    .stMultiSelect [data-baseweb="select"] {
        color: black !important;
    }
    /* More specific multiselect styling */
    .stMultiSelect div[data-baseweb="select"] > div {
        color: black !important;
    }
    .stMultiSelect span {
        color: black !important;
    }
    .stMultiSelect [role="option"] {
        color: black !important;
    }
    /* Force all multiselect elements */
    div[data-testid="stMultiSelect"] * {
        color: black !important;
    }
    .stButton > button {
        background-color: #8A2BE2 !important;
        color: white !important;
        border: 2px solid #00FF00 !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #9370DB !important;
        color: white !important;
        border: 2px solid #00FF00 !important;
    }
    /* Download button - black text */
    .stDownloadButton > button {
        background-color: #8A2BE2 !important;
        color: black !important;
        border: 2px solid #00FF00 !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    .stDownloadButton > button:hover {
        background-color: #9370DB !important;
        color: black !important;
        border: 2px solid #00FF00 !important;
    }
    /* Make radio buttons visible */
    .stRadio > div > label > div > p {
        color: white !important;
        font-weight: bold !important;
    }
    .output-box {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #00FF00;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 15px #00FF00;
        color: white !important;
    }
    /* Ensure all text elements are white except inputs */
    div:not(.stTextInput):not(.stTextArea), p, span, label, h1, h2, h3 {
        color: white !important;
    }
    /* Override Streamlit's default colors */
    .stApp > header {
        background-color: transparent;
    }
    /* Sidebar background black */
    section[data-testid="stSidebar"] {
        background-color: black !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    .loading {
        animation: pulse 1.5s infinite;
        color: white !important;
    }
    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }
    .status-badge {
        background-color: #00FF00;
        color: white !important;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Force white text on all elements */
    div, p, span, label, h1, h2, h3 {
        color: white !important;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .stSidebar {
        background-color: black !important;
    }
    .css-1d391kg .stSelectbox, .css-1d391kg .stMultiSelect {
        background-color: black !important;
    }
/* Sidebar text dark black for all languages */
    .css-1d391kg {
        color: #000000 !important;
    }
    .css-1d391kg *:not(.stMultiSelect *) {
        color: #000000 !important;
    }
    /* Sidebar multiselect - black text */
    section[data-testid="stSidebar"] .stMultiSelect * {
        color: black !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        color: black !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: black !important;
    }
    /* Force black text on all multiselect options - aggressive approach */
    section[data-testid="stSidebar"] [role="option"] {
        color: #000000 !important;
        background-color: white !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect span {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect div {
        color: #000000 !important;
    }
    /* Target specific multiselect elements */
    section[data-testid="stSidebar"] [data-baseweb="popover"] {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="popover"] * {
        color: #000000 !important;
    }
    /* Override all text in sidebar multiselect - comprehensive */
    .css-1d391kg .stMultiSelect {
        color: #000000 !important;
    }
    .css-1d391kg .stMultiSelect * {
        color: #000000 !important;
    }
    /* Target all possible multiselect elements */
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect p {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stMultiSelect [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    /* Force black on selected items */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        color: #000000 !important;
        background-color: #f0f0f0 !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="tag"] * {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("MILTRANS: Context-Aware AI Translation Engine for Military SOPs")
st.sidebar.markdown('<span class="status-badge">Connected to MongoDB</span>', unsafe_allow_html=True)


# ---------------------------
# MongoDB Connection (secrets or env)
# ---------------------------
# Preferred: put your connection string in .streamlit/secrets.toml under:
# [mongodb]
# uri = "mongodb+srv://<user>:<pass>@<cluster-url>/?retryWrites=true&w=majority&appName=<AppName>"
MONGODB_URI = None
try:
    MONGODB_URI = st.secrets["mongodb"]["uri"]
except Exception:
    MONGODB_URI = os.getenv("MONGODB_URI", None)

client = None
collection = None

if MONGODB_URI:
    try:
        client = MongoClient(MONGODB_URI, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        st.sidebar.success("✅ Connected to MongoDB")
        db = client["translations_db"]
        collection = db["translations"]
    except Exception as e:
        st.sidebar.warning(f"⚠ MongoDB unavailable: Running without database")
        client = None
        collection = None
# Database is optional - app works without it


def save_translation(source_text, translated_text, source_lang="hi", target_lang="en"):
    """Save translation to MongoDB if it doesn't already exist."""
    if collection is None:
        return None
    existing_doc = collection.find_one({
        "source_text": source_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    })
    if existing_doc:
        st.sidebar.info(f"ℹ Translation for {target_lang} already exists in DB.")
        return existing_doc
    new_doc = {
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    collection.insert_one(new_doc)
    st.sidebar.success(f"✅ Saved new translation ({target_lang}) to DB.")
    return new_doc


# ---------------------------
# Supported Languages (NLLB codes)
# ---------------------------
LANGUAGES = {
    "English": "eng_Latn",
    "Kannada": "kan_Knda",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Marathi": "mar_Deva",
    "Malayalam": "mal_Mlym",
    "Odia": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Urdu": "urd_Arab",
    "Gujarati": "guj_Gujr",
    "Assamese": "asm_Beng",
    "Bhojpuri": "bho_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Nepali": "npi_Deva",
    "Manipuri (Meitei)": "mni_Beng"
}

# ---------------------------
# Sidebar - select multiple target languages
# ---------------------------
target_languages = st.sidebar.multiselect(
    "Select target languages:",
    options=list(LANGUAGES.keys()),
    default=["English"]
)

# Ensure English is always included
if "English" not in target_languages:
    target_languages.insert(0, "English")


# ---------------------------
# Initialize EasyOCR Reader
# ---------------------------
@st.cache_resource
def _init_reader():
    # EasyOCR uses ISO-639-1 codes; 'hi' for Hindi
    return easyocr.Reader(['hi'])

reader = _init_reader()


# ---------------------------
# Hindi Normalizer
# ---------------------------
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("hi")

def normalize_hindi(text: str) -> str:
    if not text:
        return ""
    return normalizer.normalize(text)


# ---------------------------
# Load translator
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_nllb_translator(src_code: str, tgt_code: str):
    try:
        return pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_code,
            tgt_lang=tgt_code
        )
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def translate_text(text: str, tgt_lang_code: str) -> str:
    try:
        translator = load_nllb_translator("hin_Deva", tgt_lang_code)
        if translator is None:
            return "Translation service unavailable. Please restart the app."
        result = translator(text)
        return result[0]["translation_text"]
    except Exception as e:
        return f"Translation failed: {str(e)[:100]}..."


# ---------------------------
# OCR Function
# ---------------------------
def extract_text_from_image(image_file) -> str:
    img = Image.open(image_file).convert("RGB")
    result = reader.readtext(np.array(img), detail=0)
    text = " ".join(result).strip()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s\.,;!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------------------
# Web scraping
# ---------------------------
def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text(" ", strip=True) for p in paragraphs])
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\.,;!?-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


# ---------------------------
# Audio to text
# ---------------------------
def extract_text_from_audio(audio_file) -> str:
    recognizer = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        wav_io = BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            if len(audio_data.frame_data) == 0:
                return "Error: No speech detected in the audio."
            try:
                return recognizer.recognize_google(audio_data, language="hi-IN")
            except sr.UnknownValueError:
                return "Error: Could not understand audio."
            except sr.RequestError as e:
                return f"Error: Google Speech Recognition request failed; {e}"
    except Exception as e:
        return f"Error extracting text from audio: {e}"


# ---------------------------
# UI - Input selection
# ---------------------------
available_inputs = ["Text", "File (.txt)", "Image", "Web URL"]
if AUDIO_ENABLED:
    available_inputs.append("Audio")

input_type = st.radio(
    "Choose input type:",
    available_inputs,
    index=0,
    horizontal=True
)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "editable_text" not in st.session_state:
    st.session_state.editable_text = ""

hindi_text = ""

if input_type == "Text":
    st.session_state.text_input = st.text_area(
        "Enter Hindi text:",
        value=st.session_state.text_input,
        height=150
    )
    hindi_text = st.session_state.text_input

elif input_type == "File (.txt)":
    uploaded_file = st.file_uploader("Upload Hindi text file", type=["txt"])
    if uploaded_file:
        extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload image with Hindi text", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        extracted_text = extract_text_from_image(uploaded_image)
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Web URL":
    url_input = st.text_input("Enter Web URL:")
    if url_input:
        extracted_text = extract_text_from_url(url_input)
        st.session_state.editable_text = st.text_area(
            "Edit extracted text:",
            value=extracted_text,
            height=150
        )
        hindi_text = st.session_state.editable_text

elif input_type == "Audio" and AUDIO_ENABLED:
    st.info("Upload an audio file or record from mic.")
    audio_option = st.radio("Audio input type:", ["Upload File", "Record from Mic"], index=0, horizontal=True)

    if audio_option == "Upload File":
        uploaded_audio = st.file_uploader("Upload audio (mp3/wav)", type=["mp3", "wav"])
        if uploaded_audio:
            extracted_text = extract_text_from_audio(uploaded_audio)
            st.session_state.editable_text = st.text_area(
                "Edit extracted text:",
                value=extracted_text,
                height=150
            )
            hindi_text = st.session_state.editable_text

    elif audio_option == "Record from Mic":
        audio_bytes = audio_recorder()
        if audio_bytes:
            audio_file = BytesIO(audio_bytes)
            extracted_text = extract_text_from_audio(audio_file)
            st.session_state.editable_text = st.text_area(
                "Edit extracted text:",
                value=extracted_text,
                height=150
            )
            hindi_text = st.session_state.editable_text


# ---------------------------
# Translate Button
# ---------------------------
if st.button("Translate"):
    if hindi_text.strip():
        normalized_text = normalize_hindi(hindi_text)

        translations = {}
        for lang in target_languages:
            with st.spinner(f"Translating to {lang}..."):
                translated_text = translate_text(normalized_text, LANGUAGES[lang])
                translations[lang] = translated_text
                # save to DB if available
                save_translation(normalized_text, translated_text, "hi", lang)

        st.subheader("✅ Translated Texts:")
        for lang, text in translations.items():
            st.markdown(f"**{lang}:** {text}")

        # Download Button
        download_data = f"Original Hindi Text:\n{normalized_text}\n\n"
        for lang, text in translations.items():
            download_data += f"{lang}:\n{text}\n\n"

        st.download_button(
            label="⬇ Download All Translations",
            data=download_data.encode("utf-8"),
            file_name="translations.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please provide text input.")
