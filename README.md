# ⚡ HammadBot — AI Personal Assistant

A **RAG-based (Retrieval-Augmented Generation)** chatbot that answers questions about **Muhammad Waqas** using his personal CV and documents. Built with modern NLP tools, deployed on Streamlit Cloud.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple?style=flat-square)

---

## 🌟 Features

- 🤖 **Personal AI Assistant** — Answers questions about Muhammad Waqas
- 📚 **RAG Pipeline** — Retrieves relevant info from personal CV
- 🧠 **LLM-powered** — Uses Groq (GPT-OSS-120B) for fast, accurate answers
- 💬 **Chat History** — Remembers conversation context
- 🎨 **Modern UI** — Streamlit with futuristic masculine theme
- ☁️ **Cloud Deployed** — Accessible anytime, anywhere
- 🔄 **Auto-Build Index** — FAISS index builds automatically on first run

---

## 🏗️ Architecture
User Query
↓
[Streamlit UI] → app/streamlit_app.py
↓
[HammadBot Chatbot] → src/chatbot.py
↓
[Retriever] → src/retriever.py
↓
[FAISS Vector DB] → vectorstore/faiss_index
↓
[Context + Query] → [Groq LLM]
↓
LLM Response → User

text

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **RAG Framework** | LangChain |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Vector Database** | FAISS |
| **LLM** | Groq (openai/gpt-oss-120b) |
| **Python Version** | 3.13 |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure
Hammad_AI/
│
├── .streamlit/
│ └── config.toml # Streamlit theme config
│
├── app/
│ └── streamlit_app.py # Web UI (main entry point)
│
├── data/
│ └── Muhammad Waqas.txt # Personal CV dataset
│
├── src/
│ ├── init.py
│ ├── chat_history.py # Conversation memory
│ ├── chatbot.py # Main chatbot logic
│ ├── document_loader.py # Load CV files
│ ├── embeddings.py # Generate embeddings
│ ├── llm_chain.py # RAG chain + LLM
│ ├── retriever.py # Similarity search
│ ├── text_splitter.py # Chunk documents
│ └── vector_db.py # FAISS vector store
│
├── vectorstore/ # Auto-generated FAISS index
│ └── faiss_index/
│
├── .env # API keys (NOT pushed)
├── .gitignore
├── build_index.py # Build FAISS index
├── requirements.txt
├── runtime.txt # Python version
└── README.md

text

---

## 🚀 Quick Start (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/f2023065318-dot/Hammad_AI.git
cd Hammad_AI
2. Create Virtual Environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Setup Environment Variables
Create a .env file in the root directory:

env
GROQ_API_KEY=your_groq_api_key_here
Get your free Groq API key from: https://console.groq.com/keys

5. Build the FAISS Index
bash
python build_index.py
6. Run the Chatbot
Web UI mode (recommended):

bash
python -m streamlit run app/streamlit_app.py
CLI mode:

bash
python src/chatbot.py
Open your browser at http://localhost:8501 🎉

☁️ Deployment (Streamlit Cloud)
1. Push to GitHub
bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Hammad_AI.git
git push -u origin main
2. Deploy on Streamlit Cloud
Go to https://share.streamlit.io

Click "Create app" → "Deploy a public app from GitHub"

Select your repository: f2023065318-dot/Hammad_AI

Branch: main

Main file path: app/streamlit_app.py

Click "Advanced settings" → Secrets and add:

toml
GROQ_API_KEY = "your_groq_api_key_here"
Click "Deploy!" 🚀

💡 Sample Questions
Try asking HammadBot:

"What is Muhammad Waqas's email address?"

"What projects has he worked on?"

"What are his technical skills?"

"What is his education?"

"Tell me about his experience."

"What blockchain experience does he have?"

🔒 Security Notes
⚠️ Never commit .env file — it contains your Groq API key

✅ .gitignore already excludes .env, venv/, and vectorstore/

✅ On Streamlit Cloud, use Secrets instead of .env

🐛 Troubleshooting
Issue	Solution
ModuleNotFoundError	Ensure (venv) is activated
GROQ_API_KEY not found	Check .env file or Streamlit Secrets
FAISS index not found	Run python build_index.py
Model not found	Verify openai/gpt-oss-120b is available on Groq
Slow first run	First embedding download takes 1-2 minutes
📊 How RAG Works Here
Loading — CV is loaded from data/ folder

Chunking — Text is split into 9 chunks of ~500 characters

Embedding — Each chunk is converted to a 384-dim vector

Indexing — Vectors stored in FAISS for fast search

Retrieval — Top 3-4 relevant chunks found for each query

Generation — Groq LLM generates answer using retrieved context

Memory — Chat history is preserved for contextual answers

👨‍💻 Author
Muhammad Waqas

🎓 BS Software Engineering @ UMT

📧 Email: malikhammad6445@gmail.com

📱 Phone: 03085315318

📍 Lahore, Pakistan

📄 License
This project is for educational purposes as part of the Natural Language Processing (CC438) course at UMT.

🙏 Acknowledgements
LangChain — RAG framework

Groq — Fast LLM inference

Streamlit — Web framework

HuggingFace — Embedding models

FAISS — Vector similarity search

<div align="center">
⚡ Built with passion at UMT ⚡

</div> ```