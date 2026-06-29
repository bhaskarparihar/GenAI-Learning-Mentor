# 🎓 GenAI Learning Mentor

> An AI-powered, adaptive learning coach that provides **personalized study plans**, **quiz generation**, **weak-area identification**, and **practice questions** — all grounded in your own course material via RAG.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Adaptive Tutor Chat** | Ask any question about your uploaded notes — answers are grounded in your material |
| 📝 **Quiz Generation** | AI generates MCQs from your course content with instant grading and explanations |
| 📅 **Personalized Study Plan** | Get a day-by-day study plan tailored to your weak areas and timeline |
| 🔍 **Weak Area Identification** | The coach analyzes your quiz results and chats to pinpoint knowledge gaps |
| ❓ **Practice Questions** | Generates questions at multiple cognitive levels (Recall → Analysis) with answers |

---

## 🏗️ Architecture

```
User (Streamlit UI)
       │
       ▼
  PDF / Text Upload
       │
       ▼
LangChain Document Loader + Text Splitter
       │
       ▼
  Embedding Model (Gemini / OpenAI)
       │
       ▼
 ChromaDB Vector Store  ◄──────────────────────┐
       │                                        │
       ▼                                        │
  MMR Retrieval                                 │
       │                                        │
       ▼                                        │
LangChain RetrievalQA / Agent (LearningAgent) ──┘
       │
       ▼
  LLM (Gemini 1.5 Flash / GPT-3.5 Turbo)
       │
       ▼
  Response (Study Plan / Quiz / Answer / Analysis)
       │
       ▼
 Streamlit Frontend (User)
```

---

## 🛠️ Tech Stack

### Mandatory
- **Python 3.10+**
- **LangChain** — RAG pipeline, prompt templates, RetrievalQA chain
- **Gemini API** (default) or **OpenAI API**

### Optional (All Used)
- **ChromaDB** — local persistent vector store
- **Streamlit** — frontend UI with dark glassmorphism design
- **FAISS** — available as alternative vector store

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/GenAI-Learning-Mentor.git
cd GenAI-Learning-Mentor
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your key:
# GOOGLE_API_KEY=your_gemini_api_key_here
```

> 🔑 Get a free Gemini API key at: https://aistudio.google.com/app/apikey

### 5. Run the application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📋 Usage Guide

1. **Configure** — Enter your API key in the sidebar and select your LLM provider
2. **Upload Notes** — Upload your PDF course notes (or paste text directly)
3. **Chat** — Ask questions in the Tutor Chat tab
4. **Quiz** — Generate and take an MCQ quiz to test yourself
5. **Study Plan** — Describe your goals and get a personalized plan
6. **Weak Areas** — After chatting and quizzing, analyze your performance
7. **Practice** — Generate deep-dive practice questions with model answers

---

## 📁 Project Structure

```
GenAI-Learning-Mentor/
├── app.py               # Streamlit frontend (UI + tabs)
├── rag_engine.py        # RAG pipeline (ingestion, ChromaDB, retrieval)
├── learning_agent.py    # LangChain agent (tutoring, quiz, study plan)
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── .gitignore
├── README.md
├── Architecture.md      # Detailed architecture diagram
└── Project_Report.md    # 3–5 page project report
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
