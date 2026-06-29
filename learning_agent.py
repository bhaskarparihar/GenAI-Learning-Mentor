# ============================================================
# learning_agent.py - LangChain Agent: Tutor, Planner, Quiz Master
# ============================================================

from __future__ import annotations

import json
import re
from typing import Optional

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from dotenv import load_dotenv

load_dotenv()


# ── LLM factory ─────────────────────────────────────────────
def get_llm(provider: str = "gemini", temperature: float = 0.4) -> BaseLanguageModel:
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=temperature,
            convert_system_message_to_human=True,
        )


# ── Prompt templates ─────────────────────────────────────────

TUTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert AI learning mentor helping a student understand their course material.
Use ONLY the following retrieved context to answer the student's question.
If the answer is not in the context, say: "I don't have enough information in the uploaded notes to answer that."

Context:
{context}

Student's Question:
{question}

Provide a clear, structured, and encouraging answer. Use bullet points where helpful.
""",
)

STUDY_PLAN_PROMPT = """You are an expert academic coach. Based on the following course content and the student's
described weak areas, create a detailed, personalized study plan.

Course Content Summary:
{context}

Student's Weak Areas / Goals:
{weak_areas}

Generate a structured study plan with:
1. A weekly schedule (Day-by-day breakdown for 1 week)
2. Specific topics to focus on
3. Recommended study techniques for each topic
4. Daily goals and milestones
5. Suggested practice exercises

Be encouraging, specific, and actionable.
"""

QUIZ_PROMPT = """You are an expert educator creating a quiz based on the following course material.

Course Material:
{context}

Topic Focus (if any): {topic}

Generate exactly {num_questions} multiple-choice questions. For each question provide:
- A clear question
- 4 answer options (A, B, C, D)
- The correct answer
- A brief explanation

Format your response as a valid JSON array like this:
[
  {{
    "question": "Question text here?",
    "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
    "answer": "B",
    "explanation": "Explanation of why B is correct."
  }}
]
Output ONLY the JSON array, nothing else.
"""

WEAK_AREA_PROMPT = """You are an expert AI tutor analyzing a student's learning session.

Based on the following conversation history and quiz performance, identify the student's weak areas.

Conversation / Quiz Results:
{history}

Provide:
1. **Identified Weak Areas** - List of specific topics the student struggles with
2. **Strength Areas** - Topics the student seems to understand well
3. **Root Cause Analysis** - Brief analysis of why these gaps might exist
4. **Targeted Recommendations** - Specific actions to improve each weak area
5. **Confidence Score** - Rate the student's overall understanding (1-10) with justification

Be constructive and encouraging in your response.
"""

PRACTICE_QUESTIONS_PROMPT = """You are a creative educator specializing in active learning.

Topic / Context:
{context}

Generate {num_questions} diverse practice questions covering different cognitive levels:
- Recall questions (remember facts)
- Comprehension questions (explain concepts)
- Application questions (apply knowledge)
- Analysis questions (break down information)

For each question, also provide a model answer.

Format:
**Question [N] ([Level]):**
[Question text]

**Model Answer:**
[Answer text]

---
"""


class LearningAgent:
    """The core AI learning coach orchestrating all agent capabilities."""

    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self.llm = get_llm(provider)
        self.conversation_history: list[dict] = []

    # ── Adaptive Tutoring ────────────────────────────────────
    def ask_tutor(self, question: str, retriever) -> str:
        """Answer a student question using RAG."""
        if retriever is None:
            return (
                "⚠️ No course materials uploaded yet. "
                "Please upload your PDF notes first so I can give you context-aware answers!"
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": TUTOR_PROMPT},
            return_source_documents=False,
        )
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", str(result))

        # Store in history for weak area analysis
        self.conversation_history.append({"role": "student", "content": question})
        self.conversation_history.append({"role": "mentor", "content": answer})
        return answer

    # ── Personalized Study Plan ──────────────────────────────
    def generate_study_plan(self, weak_areas: str, retriever) -> str:
        """Generate a personalized study plan using RAG context + weak areas."""
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(weak_areas)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = "No specific course notes provided. Generating a general study plan."

        prompt = STUDY_PLAN_PROMPT.format(context=context, weak_areas=weak_areas)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ── Quiz Generation ──────────────────────────────────────
    def generate_quiz(
        self, retriever, topic: str = "general", num_questions: int = 5
    ) -> list[dict]:
        """Generate MCQ quiz from course material. Returns list of question dicts."""
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(topic if topic != "general" else "key concepts")
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = f"Topic: {topic}"

        prompt = QUIZ_PROMPT.format(
            context=context, topic=topic, num_questions=num_questions
        )
        response = self.llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Extract JSON safely
        try:
            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            questions = json.loads(raw)
            return questions
        except json.JSONDecodeError:
            # Fallback: return raw as a single item
            return [{"question": raw, "options": {}, "answer": "", "explanation": ""}]

    # ── Weak Area Identification ─────────────────────────────
    def identify_weak_areas(self, quiz_results: Optional[str] = None) -> str:
        """Analyze conversation + quiz results to identify weak areas."""
        history_text = "\n".join(
            [f"{h['role'].capitalize()}: {h['content']}" for h in self.conversation_history[-20:]]
        )
        if quiz_results:
            history_text += f"\n\nQuiz Performance:\n{quiz_results}"

        if not history_text.strip():
            return "No conversation or quiz history yet. Start by asking questions or taking a quiz!"

        prompt = WEAK_AREA_PROMPT.format(history=history_text)
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ── Practice Questions ───────────────────────────────────
    def generate_practice_questions(
        self, topic: str, retriever, num_questions: int = 5
    ) -> str:
        """Generate diverse practice questions with model answers."""
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(topic)
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = f"Topic: {topic}"

        prompt = PRACTICE_QUESTIONS_PROMPT.format(
            context=context, num_questions=num_questions
        )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def clear_history(self):
        self.conversation_history = []
