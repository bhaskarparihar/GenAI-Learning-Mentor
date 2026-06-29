# Project Report: GenAI Learning Mentor
 
**Date:** June 2026  
**Track:** Generative AI Applications  

---

## 1. Problem Statement

### 1.1 Background

Modern education faces a critical challenge: one-size-fits-all learning. Traditional learning environments — online courses, textbooks, and recorded lectures — deliver the same content to every student, regardless of their prior knowledge, learning pace, or specific areas of confusion. Students are left to identify their own gaps, create their own study plans, and self-assess their understanding, often ineffectively.

### 1.2 Problem

Students need **personalized, adaptive learning guidance** that:
- Answers questions grounded in their *own* course material
- Identifies topics they struggle with (weak areas)
- Creates tailored study plans based on individual needs
- Assesses understanding through dynamic quizzing

Existing solutions (chatbots, generic tutors) hallucinate answers, lack knowledge of the student's specific curriculum, and provide no personalization.

### 1.3 Solution

The **GenAI Learning Mentor** is an AI-powered learning coach that ingests a student's actual course notes (PDFs), stores them in a vector database, and uses a RAG (Retrieval-Augmented Generation) pipeline to provide grounded, accurate, and personalized guidance across five core functions.

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a three-layer RAG architecture:

```
User → Streamlit UI → LearningAgent → [RAG: ChromaDB + Embeddings] → LLM → Response
```

### 2.2 Component Summary

| Component | Technology | Role |
|---|---|---|
| **Frontend** | Streamlit | User interface, file upload, chat, quiz display |
| **RAG Engine** | LangChain + ChromaDB | Document ingestion, chunking, embedding, retrieval |
| **Learning Agent** | LangChain Chains | Tutor, quiz generator, planner, analyzer |
| **LLM** | Gemini 1.5 Flash | Response generation |
| **Embeddings** | Gemini embedding-001 | Semantic search over course notes |
| **Vector Store** | ChromaDB (persistent) | Storage and MMR retrieval of document chunks |

### 2.3 RAG Pipeline Detail

1. **Ingestion**: PDFs are loaded via `PyPDFLoader`. Text is split into 1000-token chunks with 150-token overlap using `RecursiveCharacterTextSplitter`.
2. **Embedding**: Each chunk is embedded using `GoogleGenerativeAIEmbeddings` (`models/embedding-001`).
3. **Storage**: Embeddings are stored persistently in ChromaDB (`./chroma_db`).
4. **Retrieval**: At query time, the student's question is embedded and the top-k most relevant + diverse chunks are retrieved using **MMR (Maximal Marginal Relevance)**, which balances relevance and diversity to avoid returning redundant text blocks.
5. **Augmentation**: Retrieved chunks are injected into a prompt template alongside the user's question.
6. **Generation**: The augmented prompt is sent to the LLM, which generates a grounded response.

---

## 3. Prompt Design

Prompt engineering is central to the quality of every feature in this system.

### 3.1 Adaptive Tutor Prompt
**Strategy**: Strict context-grounded prompt. The LLM is explicitly instructed to use ONLY the retrieved context and to say "I don't know" if the answer is not present. This prevents hallucination.

```
You are an expert AI learning mentor...
Use ONLY the following retrieved context to answer...
If the answer is not in the context, say: "I don't have enough information..."
Context: {context}
Question: {question}
```

### 3.2 Study Plan Prompt
**Strategy**: Combines retrieved course context with the student's self-reported weak areas to produce a structured, actionable plan. Uses numbered headings to ensure structure.

```
You are an expert academic coach. Based on the course content and student's
weak areas, create a detailed, personalized study plan...
Generate: weekly schedule, specific topics, study techniques, daily goals, exercises.
```

### 3.3 Quiz Generation Prompt
**Strategy**: Forces JSON output for programmatic parsing. Specifies exactly `{num_questions}` questions in a strict schema. This enables automatic grading without any regex parsing heuristics.

```
Format: [{{"question": ..., "options": {"A":...,"B":...}, "answer": "B", "explanation": ...}}]
Output ONLY the JSON array, nothing else.
```

### 3.4 Weak Area Analysis Prompt
**Strategy**: Provides the full conversation history and quiz results to the LLM, which performs a meta-cognitive analysis. Uses structured output headers (Weak Areas, Strengths, Root Cause, Recommendations, Confidence Score) for consistency.

### 3.5 Practice Questions Prompt
**Strategy**: Explicitly references **Bloom's Taxonomy** cognitive levels (Recall, Comprehension, Application, Analysis) to ensure diverse, educationally-valid question types rather than simple recall questions.

---

## 4. Evaluation

### 4.1 Qualitative Evaluation

The system was evaluated qualitatively across several dimensions:

| Dimension | Method | Result |
|---|---|---|
| **Groundedness** | Manual check — does the answer exist in the uploaded PDF? | High — context-strict prompt prevents hallucination |
| **Quiz Quality** | Expert review of generated MCQs | Questions were relevant, options were plausible, distractors were appropriate |
| **Study Plan Quality** | Review against pedagogical best practices | Plans included specific techniques (spaced repetition, active recall) and realistic timelines |
| **Weak Area Detection** | Compare to known student performance | Accurately identified topics the student got wrong in quizzes |

### 4.2 RAG Retrieval Quality

- **MMR vs Similarity Search**: MMR was chosen over plain cosine similarity to avoid returning near-duplicate chunks, which degraded answer quality in initial tests.
- **Chunk Size**: A 1000-token chunk with 150-token overlap was found to balance context richness and retrieval precision.

### 4.3 Limitations

- Evaluation is primarily qualitative due to the hackathon scope; a RAGAS (RAG Assessment) framework evaluation is planned for future work.
- Quiz quality is highly dependent on the richness of uploaded notes.

---

## 5. Challenges

### 5.1 JSON Output Parsing
The quiz generator requires strict JSON output from the LLM. Early versions included markdown fences (```json ... ```) or preamble text, breaking `json.loads()`. This was solved by:
- Adding explicit instructions in the prompt: "Output ONLY the JSON array"
- Stripping markdown fences via regex before parsing
- Implementing a safe fallback when parsing fails

### 5.2 ChromaDB on Windows
Initial ChromaDB versions had SQLite path issues on Windows. Solved by using a relative path (`./chroma_db`) and ensuring the working directory is set correctly when running Streamlit.

### 5.3 Retriever Returning Empty Results
When the knowledge base is empty (no documents uploaded), retriever calls would throw errors. Solved by implementing `has_documents()` checks and returning user-friendly messages instead of crashing.

### 5.4 LLM Response Format Variability
Different LLM APIs (Gemini vs OpenAI) return responses in slightly different formats. Solved by normalizing with `.content` attribute access with a fallback to `str()`.

---

## 6. Future Enhancements

| Enhancement | Priority | Description |
|---|---|---|
| **Multi-modal Input** | High | Accept lecture video/audio via Whisper transcription |
| **RAGAS Evaluation** | High | Automated RAG pipeline evaluation using the RAGAS framework |
| **Spaced Repetition** | Medium | Track quiz performance over time and re-surface weak topics using SM-2 algorithm |
| **Student Profiles** | Medium | Persist student history, progress, and weak areas across sessions with SQLite |
| **Multi-document Comparison** | Medium | Answer questions that require synthesizing information from multiple sources |
| **Voice Interface** | Low | Add speech-to-text for a more natural tutoring experience |
| **Collaborative Learning** | Low | Allow multiple students to share a knowledge base for group study |
| **LLM Fine-tuning** | Low | Fine-tune a small model on educational Q&A pairs for offline use |

---

## 7. Conclusion

The GenAI Learning Mentor successfully demonstrates how RAG combined with carefully engineered prompts can transform a generic LLM into a personalized educational tool. By grounding all responses in the student's own course material, the system avoids hallucination while providing high-quality, context-aware tutoring, adaptive quizzing, personalized study plans, and actionable learning insights. The modular architecture (RAG Engine, Learning Agent, Streamlit UI) ensures the system is extensible for future enhancements.
