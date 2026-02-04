# rag_ui.py - UPDATED WITH ESP32 INTEGRATION
import os
import io
import json
import tempfile
import streamlit as st
from gtts import gTTS
import requests
from sentence_transformers import SentenceTransformer, util
import pickle
import faiss
from config import STT_ENGINE, TTS_ENGINE
from pytector import PromptInjectionDetector
from datetime import datetime
import sounddevice as sd
import queue
import numpy as np
import whisper
import wave
import subprocess
import sys
import time
import webbrowser
from uart_client import send_component_request


# Configuration Paths
API_KEY = os.getenv("TAMUS_AI_CHAT_API_KEY")
API_URL = os.getenv("TAMUS_AI_CHAT_API_ENDPOINT")

INDEX_PATH = "index/vector_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
FAQS_PATH = "faqs.json"
INAPPROPRIATE_LOG = "inappropriate_queries.txt"
WHISPER_MODEL = "base"  # Options: tiny, base, small (small+ may be slow on Jetson)
WHISPER_DEVICE = "cuda"  # Use GPU on Jetson

# Whisper Model Loading - Optimized for Jetson
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once and cache it. Optimized for Jetson GPU."""
    try:
        import torch
        device = WHISPER_DEVICE if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper on device: {device}")
        model = whisper.load_model(WHISPER_MODEL, device=device)
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

#UI CSS
st.markdown("""
<style>
    * { color: black !important; }
    .stApp { background-color: white !important; }
    section[data-testid="stSidebar"] { background-color: white !important; }
    
    div.stButton > button {
        color: white !important;
        background-color: #444444 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="input"] > div {
        background-color: black !important;
    }
    
    div[data-baseweb="input"] input {
        color: white !important;
    }
    
    div[data-baseweb="input"] input::placeholder {
        color: #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

# Prompt Injection Setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    detector = PromptInjectionDetector(use_groq=True, api_key=GROQ_API_KEY)
else:
    detector = PromptInjectionDetector(model_name_or_url="deberta")

detector.enable_keyword_blocking = True
detector.add_input_keywords(["ignore all previous", "bypass", "system prompt", "jailbreak", "override"])
detector.add_output_keywords(["i am hacked", "i am compromised", "system instructions"])
detector.set_input_block_message("Input blocked for security reasons: {matched_keywords}")
detector.set_output_block_message("Output contained unsafe content: {matched_keywords}")

#FAQ Disk Helpers
def load_faqs():
    if os.path.exists(FAQS_PATH):
        try:
            with open(FAQS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    default = {
        "ECEN 214": [
            {
                "question": "How do I measure the output of an op-amp circuit in the lab?",
                "answer": "Connect the oscilloscope probe across the output terminal and ground. Use correct probe attenuation and verify DC biasing."
            }
        ],
        "Equipment Troubleshooting": [
            {
                "question": "The oscilloscope is not displaying a waveform — what should I check?",
                "answer": "Confirm probe connection, vertical scale, time base settings, and trigger level."
            }
        ]
    }
    with open(FAQS_PATH, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=2)
    return default

def save_faqs(faqs):
    with open(FAQS_PATH, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2)

# Load FAISS and INDEX
def load_index():
    if not os.path.exists(INDEX_PATH):
        st.error(f"FAISS index not found at {INDEX_PATH}. Please run your index builder.")
        return None, None
    with open(INDEX_PATH, "rb") as f:
        index, texts = pickle.load(f)
    return index, texts

# Context Retrieval
def retrieve_context(query, index, texts, embed_model, k=3):
    if index is None or texts is None:
        return ""
    qvec = embed_model.encode([query])
    distances, indices = index.search(qvec, k)
    retrieved = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(texts):
            continue
        retrieved.append(texts[idx])
    return "\n\n".join(retrieved)

# LLM Call
def query_tamuai(prompt):
    """Query TAMU AI endpoint."""
    if not API_KEY or not API_URL:
        raise RuntimeError("API_KEY or API_URL not set in environment.")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "protected.llama3.2",
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(f"{API_URL}/api/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Text to Speech
def speak_text(text):
    """Generate TTS audio with offline support."""
    if TTS_ENGINE == "gtts":
        try:
            from gtts import gTTS
            tts = gTTS(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                audio_file = fp.name
                tts.save(audio_file)
                st.audio(audio_file, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.warning(f"gTTS failed: {e}")

# Voice Recording with Whisper
def transcribe_with_whisper(duration=5):
    """Record and transcribe audio using Whisper."""
    model = load_whisper_model()
    if not model:
        return None
    
    SAMPLE_RATE = 16000
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    audio_data = audio_data.flatten()
    result = model.transcribe(audio_data, fp16=False)
    return result["text"].strip()

# FAQ matching
def match_faq_local(query, embed_model, faqs):
    """Match query against local FAQs using embeddings."""
    qvec = embed_model.encode([query])
    best_score = 0
    best_answer = None
    
    for category, qa_list in faqs.items():
        for qa in qa_list:
            faq_vec = embed_model.encode([qa["question"]])
            score = util.cos_sim(qvec, faq_vec).item()
            if score > best_score and score > 0.7:
                best_score = score
                best_answer = qa["answer"]
    
    return best_answer

# Admin pages (placeholder - keeping existing implementation)
def admin_login_page():
    st.title("Admin Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == "admin123":
            st.session_state["admin_logged_in"] = True
            st.session_state["page"] = "Admin Dashboard"
            st.rerun()
        else:
            st.error("Invalid password")

def admin_dashboard_page(faqs):
    if not st.session_state.get("admin_logged_in"):
        st.error("Please login first")
        return
    
    st.title("Admin Dashboard")
    st.write("Manage FAQs")
    
    category = st.text_input("Category")
    question = st.text_input("Question")
    answer = st.text_area("Answer")
    
    if st.button("Add FAQ"):
        if category and question and answer:
            if category not in faqs:
                faqs[category] = []
            faqs[category].append({"question": question, "answer": answer})
            save_faqs(faqs)
            st.success("FAQ added")
            st.rerun()

# Session state initialization
st.session_state.setdefault("dispense_state", "IDLE")
st.session_state.setdefault("selected_component", None)
st.session_state.setdefault("selected_quantity", 1)
st.session_state.setdefault("uart_status", None)

# LLM CLASSIFICATION FOR COMPONENT REQUESTS
def extract_resistor_request(user_input):
    """
    Uses LLM to parse user input and detect component request
    Returns component identifier that maps to bin number
    
    Bin Mapping:
    0 -> 1kΩ Resistor
    1 -> 10kΩ Resistor
    2 -> 100Ω Resistor
    3 -> 100kΩ Resistor
    """

    prompt = f"""
You are a lab assistant. Analyze this query:
"{user_input}"

Determine if the user is asking for a resistor component.
Match the request to one of these exact values:
- 1kΩ (1k ohm, 1 kilo ohm, 1kohm, 1000 ohm)
- 10kΩ (10k ohm, 10 kilo ohm, 10kohm, 10000 ohm)
- 100Ω (100 ohm, hundred ohm)
- 100kΩ (100k ohm, 100 kilo ohm, 100kohm)

Respond with ONLY ONE of these exact strings:
"1kohm"
"10kohm"
"100ohm"
"100kohm"
"none"

If the user is not requesting a resistor, respond with "none".
"""

    try:
        llm_response = query_tamuai(prompt).strip().lower()
        llm_response = llm_response.replace("ω", "").replace(" ", "").replace('"', '')

        # Map LLM response to component identifier
        valid_responses = ["1kohm", "10kohm", "100ohm", "100kohm"]
        
        if llm_response in valid_responses:
            print(f"[LLM CLASSIFICATION] Detected: {llm_response}")
            return llm_response
        else:
            print(f"[LLM CLASSIFICATION] No component detected: {llm_response}")
            return None

    except Exception as e:
        print(f"[LLM ERROR] Classification failed: {e}")
        return None

def chatbot_page(index, texts, embed_model, faqs):
    st.title("ECEN Chatbot")
    st.write("Ask lab-related or course questions by typing or speaking")

    # Sidebar FAQ quick pick
    with st.sidebar:
        st.header("Frequently Asked Questions")
        for category, qa_list in faqs.items():
            with st.expander(category):
                for qa in qa_list:
                    if st.button(qa["question"], key=f"faq_{qa['question']}"):
                        st.session_state["prefilled_question"] = qa["question"]
                        st.session_state["prefilled_answer"] = qa["answer"]

    # Initialize STT result storage
    if "stt_result" not in st.session_state:
        st.session_state["stt_result"] = ""

    default_q = st.session_state.get("prefilled_question", "")
    
    col1, col2 = st.columns([4, 1])

    with col2:
        if st.button("Speak"):
            with st.spinner("Listening for 5 seconds... Speak naturally"):
                try:
                    transcribed = transcribe_with_whisper(duration=5)
                    if transcribed:
                        st.session_state["stt_result"] = transcribed
                        st.session_state.pop("prefilled_question", None)
                        st.session_state.pop("prefilled_answer", None)
                        st.success(f"You said: {transcribed}")
                        st.rerun()
                    else:
                        st.warning("No speech detected. Please try again.")
                except Exception as e:
                    st.error(f"STT error: {e}")

    with col1:
        current_input = st.session_state.get("stt_result", default_q)
        user_input = st.text_input("Type your question here:", value=current_input, key="user_input")

    # Component Dispenser Section
    st.subheader("Component Dispenser")
    
    # Display UART status if available
    if st.session_state.get("uart_status"):
        status = st.session_state["uart_status"]
        if status["success"]:
            st.success(f"✓ Component dispensed from Bin {status['bin']}")
        else:
            st.error(f"✗ Dispense failed: {status.get('error', 'Unknown error')}")
        
        if st.button("Clear Status"):
            st.session_state["uart_status"] = None
            st.rerun()

    # Auto-display prefilled FAQ answer
    if "prefilled_answer" in st.session_state and default_q and not st.session_state.get("stt_result"):
        st.subheader("FAQ Answer")
        st.write(st.session_state["prefilled_answer"])
        
        col_play, col_clear = st.columns([1, 1])
        with col_play:
            if st.button("Play FAQ answer"):
                speak_text(st.session_state["prefilled_answer"])
        with col_clear:
            if st.button("Clear FAQ"):
                st.session_state.pop("prefilled_question", None)
                st.session_state.pop("prefilled_answer", None)
                st.rerun()
        return

    if user_input:
        # LLM CLASSIFICATION TEST
        resistor_request = extract_resistor_request(user_input)

        if resistor_request:
            print(f"[LLM-INTEGRATION-TEST] Detected component request: {resistor_request}")
            
            # Show processing message
            with st.spinner(f"Dispensing {resistor_request}..."):
                try:
                    # Send to ESP32 via UART
                    success = send_component_request(resistor_request)
                    
                    # Map component to bin number for display
                    bin_map = {"1kohm": 0, "10kohm": 1, "100ohm": 2, "100kohm": 3}
                    bin_num = bin_map.get(resistor_request, -1)
                    
                    st.session_state["uart_status"] = {
                        "success": success,
                        "component": resistor_request,
                        "bin": bin_num,
                        "error": None if success else "No HIGH response from ESP32"
                    }
                    st.rerun()
                    
                except Exception as e:
                    print(f"[UART ERROR] {e}")
                    st.session_state["uart_status"] = {
                        "success": False,
                        "error": str(e)
                    }
                    st.rerun()

        # Clear STT result now that we're processing
        if "stt_result" in st.session_state:
            st.session_state["stt_result"] = ""
            
        # input safety
        is_injection, _ = detector.detect_injection(user_input)
        blocked, keywords = detector.check_input_keywords(user_input)
        if is_injection or blocked:
            st.error("Input blocked: Potential injection or unsafe content detected.")
            with open(INAPPROPRIATE_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] {user_input}\n")
            return

        # local FAQ match first
        local_answer = match_faq_local(user_input, embed_model, faqs)
        if local_answer:
            st.subheader("FAQ Match (Local)")
            st.write(local_answer)
            if st.button("Play FAQ Response"):
                speak_text(local_answer)
            return

        # retrieve context
        context = retrieve_context(user_input, index, texts, embed_model, k=3)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer clearly and concisely."

        try:
            with st.spinner("Thinking..."):
                answer = query_tamuai(prompt)
        except Exception as e:
            st.error(f"LLM request failed: {e}")
            return

        # check response safety
        safe, matched = detector.check_response_safety(answer)
        if not safe:
            st.error("Unsafe content detected in model output.")
            with open(INAPPROPRIATE_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now()}] {user_input}\n")
            return

        st.subheader("Response")
        st.write(answer)
        if st.button("Play Response"):
            speak_text(answer)

# Main
def main():
    st.set_page_config(page_title="ECEN Chatbot", layout="wide")

    # Load embedding model
    embed_model = None
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to load embedding model {MODEL_NAME}: {e}")

    # Load index and FAQs
    index, texts = load_index()
    faqs = load_faqs()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.session_state.setdefault("page", "Chatbot")
    page = st.sidebar.radio("Go to:", ["Chatbot", "Admin Login", "Admin Dashboard"], 
                            index=["Chatbot", "Admin Login", "Admin Dashboard"].index(st.session_state["page"]))
    st.session_state["page"] = page

    # Routing
    if page == "Admin Login":
        admin_login_page()
    elif page == "Admin Dashboard":
        admin_dashboard_page(faqs)
    else:
        if embed_model is None or index is None or texts is None:
            st.error("Embeddings or index not loaded - cannot serve RAG responses.")
            return
        chatbot_page(index, texts, embed_model, faqs)

if __name__ == "__main__":
    main()