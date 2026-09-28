"""
streamlit_app.py
----------------
Streamlit web interface for HammadBot.
Masculine slate-blue theme with high contrast and readability.

Run with:
    python -m streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add src/ to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from chatbot import HammadAIChatbot


# ------------------------------------------
# Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="HammadBot - AI Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ------------------------------------------
# Masculine Slate-Blue Theme (READABLE)
# ------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Light slate background */
    .stApp {
        background: linear-gradient(135deg, #E2E8F0 0%, #CBD5E1 50%, #E2E8F0 100%);
        background-attachment: fixed;
    }

    /* Main white card */
    .main .block-container {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* Header */
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #0284C7 0%, #06B6D4 50%, #0284C7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        line-height: 1.2;
        letter-spacing: 2px;
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 1px;
    }

    .divider-line {
        text-align: center;
        color: #06B6D4;
        font-size: 1.1rem;
        margin: 1rem 0 2rem 0;
        letter-spacing: 0.8rem;
    }

    /* Chat bubbles - LIGHT with dark text */
    .stChatMessage {
        background: #F8FAFC !important;
        border-radius: 15px !important;
        border: 1.5px solid #06B6D4 !important;
        padding: 1rem 1.3rem !important;
        margin-bottom: 0.9rem !important;
        box-shadow: 0 2px 8px rgba(6, 182, 212, 0.1) !important;
        color: #0F172A !important;
        transition: all 0.3s ease;
    }

    .stChatMessage:hover {
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.25) !important;
        transform: translateY(-2px);
    }

    /* Force all text inside chat messages to be dark */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage li,
    .stChatMessage strong {
        color: #0F172A !important;
    }

    /* Sidebar - dark with white text */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 2px solid #06B6D4;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #22D3EE !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #E2E8F0 !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: #1E293B;
        color: #67E8F9 !important;
        border: 1.5px solid #06B6D4;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        text-align: left;
        font-size: 0.9rem;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(90deg, #06B6D4 0%, #3B82F6 100%);
        color: #FFFFFF !important;
        border-color: #22D3EE;
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(6, 182, 212, 0.5);
    }

    section[data-testid="stSidebar"] .stButton > button p {
        color: inherit !important;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 15px !important;
        border: 2px solid #06B6D4 !important;
        background: #FFFFFF !important;
        box-shadow: 0 5px 20px rgba(6, 182, 212, 0.2);
    }

    .stChatInput > div:focus-within {
        box-shadow: 0 5px 25px rgba(6, 182, 212, 0.5);
        border-color: #0284C7 !important;
    }

    .stChatInput textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: #0F172A !important;
    }

    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
        font-size: 0.85rem !important;
        text-align: center;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #06B6D4 !important;
    }

    hr {
        border-color: rgba(6, 182, 212, 0.4) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# Header
# ------------------------------------------
st.markdown('<div class="main-header">⚡ HammadBot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI ASSISTANT — Ask me about Muhammad Waqas</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="divider-line">◆ ─── ⚡ ─── ◆</div>', unsafe_allow_html=True)


# ------------------------------------------
# Sidebar
# ------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 About")
    st.write(
        "**HammadBot** is a RAG-based AI assistant that answers "
        "questions about Muhammad Waqas using his personal CV and documents."
    )
    st.divider()

    st.markdown("### 💡 Sample Queries")
    sample_questions = [
        "What is Muhammad Waqas's email address?",
        "What projects has he worked on?",
        "What are his technical skills?",
        "What is his education?",
        "Tell me about his experience.",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if "bot" in st.session_state:
            st.session_state.bot.reset()
        st.rerun()

    st.divider()
    st.caption("⚡ Powered by LangChain, FAISS, Groq & Streamlit")


# ------------------------------------------
# Session State
# ------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "bot" not in st.session_state:
    with st.spinner("Initializing HammadBot..."):
        st.session_state.bot = HammadAIChatbot()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ------------------------------------------
# Display Chat History
# ------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="🧑" if message["role"] == "user" else "⚡"
    ):
        st.markdown(message["content"])


# ------------------------------------------
# Handle Sample Question Click
# ------------------------------------------
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Processing..."):
            response = st.session_state.bot.respond(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


# ------------------------------------------
# Chat Input
# ------------------------------------------
user_input = st.chat_input("💬 Ask me anything about Muhammad Waqas...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Processing..."):
            response = st.session_state.bot.respond(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


# ------------------------------------------
# Footer
# ------------------------------------------
st.divider()
st.caption("⚡ HammadBot answers only from Muhammad Waqas's personal data. No hallucinations.")