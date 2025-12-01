# rag_ui.py
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
from vosk import Model, KaldiRecognizer
import threading

# Configuration Paths
API_KEY = os.getenv("TAMUS_AI_CHAT_API_KEY")
API_URL = os.getenv("TAMUS_AI_CHAT_API_ENDPOINT")

INDEX_PATH = "index/vector_index.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
FAQS_PATH = "faqs.json"
INAPPROPRIATE_LOG = "inappropriate_queries.txt"
VOSK_MODEL_PATH = "vosk-model-en-us-0.22-lgraph"

# Speech to text Vosk Model
@st.cache_resource
def load_vosk_model():
    """Load Vosk model once and cache it."""
    try:
        return Model(VOSK_MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load Vosk model from {VOSK_MODEL_PATH}: {e}")
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

# Text to Speech  - GTTS
def speak_text(text):
    if TTS_ENGINE == "gtts":
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    elif TTS_ENGINE == "edge_tts":
        import edge_tts
        import asyncio
        async def _speak():
            communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                await communicate.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        asyncio.run(_speak())

# VOSK STT Function - updated model to 1.8 GB eek
def transcribe_with_vosk(duration=10):
    """
    Record audio for specified duration and transcribe using Vosk.
    Returns transcribed text.
    IMPROVED: Better sample rate, larger model recommended, longer duration
    """
    vosk_model = load_vosk_model()
    if vosk_model is None:
        raise RuntimeError("Vosk model not loaded")
    
    # Use 16000 Hz for better accuracy (Vosk's recommended rate)
    recognizer = KaldiRecognizer(vosk_model, 16000)
    recognizer.SetMaxAlternatives(0)  # Only best result
    recognizer.SetWords(True)  # Enable word-level timestamps for better accuracy
    
    audio_q = queue.Queue()
    transcribed_text = []
    
    def callback(indata, frames, time, status):
        if status:
            print("Audio status:", status)
        audio_q.put(bytes(indata))
    
    # Record for specified duration with optimized settings
    with sd.RawInputStream(samplerate=16000, blocksize=2000, dtype="int16",
                           channels=1, callback=callback):
        import time
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                data = audio_q.get(timeout=0.1)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text.strip():
                        transcribed_text.append(text.strip())
            except queue.Empty:
                continue
        
        # Get final result
        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "")
        if final_text.strip():
            transcribed_text.append(final_text.strip())
    
    return " ".join(transcribed_text)

# Local FAQ Fuzzy match
def match_faq_local(user_input, embed_model, faqs, threshold=0.78):
    best_match = None
    best_score = 0.0
    user_vec = embed_model.encode(user_input)
    for cat, qalist in faqs.items():
        for qa in qalist:
            qvec = embed_model.encode(qa["question"])
            score = util.cos_sim(user_vec, qvec).item()
            if score > best_score:
                best_score = score
                best_match = qa
    if best_score >= threshold:
        return best_match["answer"]
    return None

# Admin Page
def admin_login_page():
    st.subheader("🔐 Admin Login")
    username = st.text_input("Username", key="admin_user")
    password = st.text_input("Password", type="password", key="admin_pass")

    col1, col2 = st.columns([1,1])
    with col1:
        login_clicked = st.button("Login")
    with col2:
        if st.button("Cancel"):
            st.session_state["page"] = "Chatbot"
            st.rerun()

    if login_clicked:
        if username == "admin" and password == "password":
            st.session_state["admin_logged_in"] = True
            st.session_state["page"] = "Admin Dashboard"
            st.success("Login successful! Redirecting to dashboard...")
            st.rerun()
        else:
            st.error("Invalid credentials")
            
def admin_dashboard_page(faqs):
    if not st.session_state.get("admin_logged_in", False):
        st.warning("You must log in as admin to view this page.")
        st.session_state["page"] = "Admin Login"
        st.rerun()

    st.title("Admin Dashboard")
    st.markdown("Use the dashboard to inspect logs, inventory and edit FAQs.")

    col1, col2, col3 = st.columns([1,2,2])
    with col1:
        if st.button("Logout"):
            st.session_state["admin_logged_in"] = False
            st.session_state["page"] = "Chatbot"
            st.success("Logged out.")
            st.rerun()
    with col2:
        st.metric("FAQ categories", len(faqs.keys()))
    with col3:
        count = 0
        if os.path.exists(INAPPROPRIATE_LOG):
            with open(INAPPROPRIATE_LOG, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
        st.metric("Inappropriate queries", count)

    st.markdown("---")
    st.subheader("🚫 Inappropriate Queries")
    if os.path.exists(INAPPROPRIATE_LOG):
        with open(INAPPROPRIATE_LOG, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            st.dataframe({"timestamped_query": lines})
            if st.button("Clear log"):
                open(INAPPROPRIATE_LOG, "w", encoding="utf-8").close()
                st.success("Log cleared.")
                st.rerun()
        else:
            st.info("No inappropriate queries logged.")
    else:
        st.info("No log file found.")

    st.markdown("---")
    st.subheader("📦 Component Stock (dummy)")
    stock = {
        "1kΩ Resistors": "High",
        "2kΩ Resistors": "Low",
        "10kΩ Resistors": "Medium",
        "Capacitors (0.1 µF)": "High"
    }
    st.table({"Component": list(stock.keys()), "Stock Level": list(stock.values())})

    st.markdown("---")
    st.subheader("FAQ Editor (Add / Edit / Delete)")
    categories = list(faqs.keys())
    edit_mode = st.radio("Mode:", ["Add FAQ", "Edit FAQ", "Delete FAQ"])

    if edit_mode == "Add FAQ":
        new_cat = st.text_input("Category (e.g. 'ECEN 214')", key="new_cat")
        new_q = st.text_input("Question", key="new_q")
        new_a = st.text_area("Answer", key="new_a")
        if st.button("Add FAQ"):
            if not new_cat or not new_q or not new_a:
                st.error("Fill category, question and answer.")
            else:
                if new_cat not in faqs:
                    faqs[new_cat] = []
                faqs[new_cat].append({"question": new_q, "answer": new_a})
                save_faqs(faqs)
                st.success("FAQ added.")
                st.rerun()

    elif edit_mode == "Edit FAQ":
        if not categories:
            st.info("No categories available.")
        else:
            sel_cat = st.selectbox("Select category", categories, key="edit_cat")
            q_list = faqs.get(sel_cat, [])
            if q_list:
                q_titles = [q["question"] for q in q_list]
                sel_q_idx = st.selectbox("Select question to edit", list(range(len(q_titles))), format_func=lambda i: q_titles[i], key="edit_qidx")
                sel_q = q_list[sel_q_idx]
                edited_q = st.text_input("Question", value=sel_q["question"], key="edited_q")
                edited_a = st.text_area("Answer", value=sel_q["answer"], key="edited_a")
                if st.button("Save changes"):
                    faqs[sel_cat][sel_q_idx] = {"question": edited_q, "answer": edited_a}
                    save_faqs(faqs)
                    st.success("FAQ updated.")
                    st.rerun()
            else:
                st.info("No questions in this category.")

    elif edit_mode == "Delete FAQ":
        if not categories:
            st.info("No categories available.")
        else:
            sel_cat = st.selectbox("Select category", categories, key="del_cat")
            q_list = faqs.get(sel_cat, [])
            if q_list:
                q_titles = [q["question"] for q in q_list]
                sel_q_idx = st.selectbox("Select question to delete", list(range(len(q_titles))), format_func=lambda i: q_titles[i], key="del_qidx")
                if st.button("Delete selected FAQ"):
                    faqs[sel_cat].pop(sel_q_idx)
                    if not faqs[sel_cat]:
                        del faqs[sel_cat]
                    save_faqs(faqs)
                    st.success("FAQ deleted.")
                    st.rerun()
            else:
                st.info("No questions to delete in this category.")

# Chatbot UI Page - IMPROVED
def chatbot_page(index, texts, embed_model, faqs):
    st.title("ECEN Chatbot")
    st.write("Ask lab-related or course questions by typing or speaking (powered by Vosk offline STT).")

    # Sidebar FAQ quick pick
    with st.sidebar:
        st.header("Frequently Asked Questions")
        for category, qa_list in faqs.items():
            with st.expander(category):
                for qa in qa_list:
                    if st.button(qa["question"], key=f"faq_{qa['question']}"):
                        st.session_state["prefilled_question"] = qa["question"]
                        st.session_state["prefilled_answer"] = qa["answer"]
                        st.rerun()

    # Initialize STT result storage
    if "stt_result" not in st.session_state:
        st.session_state["stt_result"] = ""

    default_q = st.session_state.get("prefilled_question", "")
    
    col1, col2 = st.columns([4, 1])

    with col2:
        if st.button("🎤 Speak"):
            with st.spinner("🎙️ Listening for 7 seconds... Speak clearly near your microphone"):
                try:
                    transcribed = transcribe_with_vosk(duration=7)
                    if transcribed:
                        st.session_state["stt_result"] = transcribed
                        # Clear any prefilled FAQ data
                        st.session_state.pop("prefilled_question", None)
                        st.session_state.pop("prefilled_answer", None)
                        st.success(f"✅ You said: {transcribed}")
                        st.rerun()
                    else:
                        st.warning("⚠️ No speech detected. Please try again and speak louder.")
                except Exception as e:
                    st.error(f"STT error: {e}")

    with col1:
        # Use STT result if available, otherwise use default
        current_input = st.session_state.get("stt_result", default_q)
        user_input = st.text_input("Type your question here:", value=current_input, key="user_input")

    # Auto-display prefilled FAQ answer
    if "prefilled_answer" in st.session_state and default_q and not st.session_state.get("stt_result"):
        st.subheader("💬 FAQ Answer")
        st.write(st.session_state["prefilled_answer"])
        if st.button("Play FAQ answer"):
            speak_text(st.session_state["prefilled_answer"])
        if st.button("Clear FAQ"):
            st.session_state.pop("prefilled_question", None)
            st.session_state.pop("prefilled_answer", None)
            st.rerun()
        return

    if user_input:
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
            st.subheader("💬 FAQ Match (Local)")
            st.write(local_answer)
            if st.button("Play FAQ Response"):
                speak_text(local_answer)
            return

        # retrieve context
        context = retrieve_context(user_input, index, texts, embed_model, k=3)
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer clearly and concisely."

        try:
            with st.spinner("🤔 Thinking..."):
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
