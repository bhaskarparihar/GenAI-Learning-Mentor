# Architecture — GenAI Learning Mentor

## Overview

The GenAI Learning Mentor uses a **Retrieval-Augmented Generation (RAG)** pipeline to ground all LLM responses in the student's actual course material. The system is composed of four main layers: the **User Interface**, the **RAG Engine**, the **Learning Agent**, and the **LLM**.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER (Student)                              │
│                  Streamlit Web Interface                        │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────┐ ┌────────┐  │
│  │ Tutor    │ │  Quiz    │ │  Study    │ │ Weak │ │Practice│  │
│  │  Chat    │ │Generator │ │  Plan     │ │ Area │ │   Qs   │  │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └──┬───┘ └───┬────┘  │
└───────┼────────────┼─────────────┼───────────┼─────────┼───────┘
        │            │             │           │         │
        └────────────┴─────────────┴───────────┴─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       LearningAgent         │
                    │  (learning_agent.py)        │
                    │  • Prompt Engineering       │
                    │  • RAG Chain (LangChain)    │
                    │  • Conversation Memory      │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
   ┌──────────▼──────────┐          ┌───────────▼──────────┐
   │     RAG Engine       │          │         LLM           │
   │   (rag_engine.py)    │          │  Gemini 1.5 Flash or  │
   │                      │          │  GPT-3.5 Turbo        │
   │  ┌────────────────┐  │◄────────►│                       │
   │  │ PDF / Text     │  │  Augm.   │  via LangChain        │
   │  │  Loader        │  │  Prompt  │  ChatGoogleGenAI /    │
   │  └───────┬────────┘  │          │  ChatOpenAI           │
   │          │           │          └───────────────────────┘
   │  ┌───────▼────────┐  │
   │  │ Text Splitter  │  │
   │  │ (Chunk: 1000t) │  │
   │  └───────┬────────┘  │
   │          │           │
   │  ┌───────▼────────┐  │
   │  │  Embeddings    │  │
   │  │ (Gemini/OpenAI)│  │
   │  └───────┬────────┘  │
   │          │           │
   │  ┌───────▼────────┐  │
   │  │  ChromaDB      │  │
   │  │  Vector Store  │  │
   │  │  (Persistent)  │  │
   │  └───────┬────────┘  │
   │          │ MMR       │
   │          │ Retrieval │
   └──────────┴───────────┘
```

---

## Component Breakdown

### 1. Streamlit Frontend (`app.py`)
- **Role**: User interface with 5 tabs (Tutor Chat, Quiz, Study Plan, Weak Areas, Practice Qs)
- **Key Features**: File upload, API key configuration, dark glassmorphism UI
- **Technology**: Streamlit 1.35, custom CSS, Google Fonts

### 2. RAG Engine (`rag_engine.py`)
- **Role**: Handles all document ingestion and retrieval
- **Document Loading**: `PyPDFLoader` for PDFs, direct text ingestion
- **Chunking**: `RecursiveCharacterTextSplitter` with 1000 token chunks and 150 token overlap
- **Embeddings**: `GoogleGenerativeAIEmbeddings` (models/embedding-001) or `OpenAIEmbeddings`
- **Vector Store**: `ChromaDB` with persistent storage (`./chroma_db`)
- **Retrieval Strategy**: **MMR (Maximal Marginal Relevance)** to ensure diverse, relevant results

### 3. Learning Agent (`learning_agent.py`)
- **Role**: Core AI coach — orchestrates all agent capabilities via carefully engineered prompts
- **Capabilities**:
  | Function | LangChain Component | Prompt Style |
  |---|---|---|
  | `ask_tutor()` | `RetrievalQA` with custom prompt | RAG-grounded, context-strict |
  | `generate_quiz()` | Direct LLM call | JSON-structured output |
  | `generate_study_plan()` | Direct LLM call | Structured planning prompt |
  | `identify_weak_areas()` | Direct LLM call | History + quiz analysis |
  | `generate_practice_questions()` | Direct LLM call | Bloom's taxonomy levels |

### 4. LLM Layer
- **Primary**: Gemini 1.5 Flash (`gemini-1.5-flash`) via `langchain-google-genai`
- **Alternate**: GPT-3.5 Turbo via `langchain-openai`
- **Temperature**: 0.4 (balanced between determinism and creativity)

---

## Data Flow

```
1. INGEST:   PDF → PyPDFLoader → Text Splitter → Embeddings → ChromaDB
2. QUERY:    User Question → LearningAgent → ChromaDB (MMR Retrieval)
3. AUGMENT:  Retrieved Chunks + Question → Prompt Template
4. GENERATE: Augmented Prompt → LLM → Response
5. DISPLAY:  Response → Streamlit UI → User
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **ChromaDB over FAISS** | ChromaDB offers persistent storage without serialization issues on Windows |
| **MMR Retrieval** | Avoids returning redundant chunks; improves answer quality |
| **Gemini as default** | Free tier available; fast inference; strong multilingual support |
| **Streamlit** | Fastest path to a functional, deployable UI for a hackathon setting |
| **Modular architecture** | `rag_engine.py` and `learning_agent.py` are fully decoupled for testability |
