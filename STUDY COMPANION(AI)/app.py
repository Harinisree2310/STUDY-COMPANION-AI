# main_app.py
from typing import List, Dict
import random
import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# Try GROQ import
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    st.error("Please install GROQ using `pip install groq`")
    GROQ_AVAILABLE = False


# ------------------------------
# Utility Classes
# ------------------------------
class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        except Exception as e:
            st.error(f"PDF Extraction Error: {e}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk.strip())
        return chunks


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def add_documents(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype("float32"))

    def similarity_search(self, query: str, k: int = 5):
        if not self.index:
            return []
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


class GroqMCQGenerator:
    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"

    def generate_mcqs(self, context, topic, num_questions, difficulty, question_types):
        prompt = f"""
Generate {num_questions} MCQs about \"{topic}\" with difficulty {difficulty}.
Types: {', '.join(question_types)}.
Use only the context below.

CONTEXT:
{context}

Format:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
  }}
]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional MCQ generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )

            raw = response.choices[0].message.content

            if not raw or not raw.strip():
                st.error("❌ Empty response from LLM.")
                return []

            json_start = raw.find("[")
            if json_start == -1:
                st.error("❌ JSON array '[' not found in response.")
                return []

            json_part = raw[json_start:].strip()
            if "```" in json_part:
                json_part = json_part.split("```")[0].strip()

            return json.loads(json_part)

        except Exception as e:
            st.error(f"❌ Failed to parse MCQs: {e}")
            return []


class GroqFlashcardGenerator:
    def __init__(self, api_key: str):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"  # Stable and supported model

    def generate_flashcards(self, context, topic, num_cards=5):
        prompt = f"""
Generate {num_cards} educational flashcards from the content below about "{topic}".
Each flashcard should have a question and a short, clear answer.

Format:
[
  {{
    "question": "...",
    "answer": "..."
  }}
]

CONTENT:
{context}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful flashcard generator for students."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )

            raw = response.choices[0].message.content

            # --- FIX: Extract only the JSON array from the output ---
            import re
            match = re.search(r"\[\s*{.*}\s*\]", raw, re.DOTALL)
            if not match:
                st.error("❌ No valid JSON array found in output.")
                return []

            json_part = match.group(0)

            import json
            return json.loads(json_part)

        except Exception as e:
            st.error(f"Flashcard generation failed: {str(e)}")
            return []


def generate_summary(api_key: str, context: str, topic: str, model: str = "llama3-8b-8192") -> str:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = f"""
You are a helpful academic assistant. Summarize the following content clearly and concisely.
Focus on the most important concepts relevant to the topic.

TOPIC: {topic}
CONTENT:
{context}

SUMMARY:
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You generate short, educational summaries for students."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=512,
            top_p=1.0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Summary generation failed: {str(e)}")
        return ""


def get_motivational_quote():
    quotes = [
        "Believe you can and you're halfway there.",
        "Push yourself, because no one else is going to do it for you.",
        "Dream it. Wish it. Do it.",
        "Success doesn’t just find you. You have to go out and get it.",
        "Great things never come from comfort zones.",
        "The harder you work for something, the greater you'll feel when you achieve it.",
    ]
    return random.choice(quotes)


# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.set_page_config(page_title="📚 AI Study Companion", layout="wide")
    st.title("📚 AI Study Companion")
    st.markdown("Upload PDFs, generate summaries, MCQs, and flashcards with GROQ!")

    if not GROQ_AVAILABLE:
        st.stop()

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'quiz_mode' not in st.session_state:
        st.session_state.quiz_mode = False
    if 'mcqs' not in st.session_state:
        st.session_state.mcqs = []
    if 'score' not in st.session_state:
        st.session_state.score = 0

    with st.sidebar:
        st.header("AI STUDY COMPANION")
        mcq_api_key = "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"
        summary_api_key = "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"
        flashcard_api_key = "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"

        st.divider()
        if "quote" not in st.session_state:
            st.session_state["quote"] = get_motivational_quote()
        st.subheader("💡 Motivational Quote")
        st.info(st.session_state["quote"])
        if st.button("🔄 New Quote"):
            st.session_state["quote"] = get_motivational_quote()
            st.rerun()
        
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process PDFs"):
        all_chunks = []
        for file in uploaded_files:
            text = PDFProcessor.extract_text_from_pdf(file)
            chunks = PDFProcessor.chunk_text(text)
            all_chunks.extend(chunks)
            st.success(f"Processed {file.name}")
        st.session_state.vector_store.add_documents(all_chunks)
        st.session_state.documents_processed = True

    if st.session_state.get("documents_processed", False):
        st.success("✅ Documents processed and ready!")

        topic = st.text_input("Topic (for summary, MCQs, and flashcards)")
        num_questions = st.slider("Number of MCQs", 1, 20, 5)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        question_types = st.multiselect("Question Types", ["Factual", "Conceptual", "Application"], default=["Factual"])

        if topic and st.button("📄 Generate Summary"):
            chunks = st.session_state.vector_store.similarity_search(topic, k=5)
            context = "\n\n".join(chunks)[:3000]
            summary = generate_summary(summary_api_key, context, topic)
            if summary:
                st.markdown("### 📝 Summary")
                st.info(summary)

        if topic and st.button("🧠 Generate MCQs"):
            chunks = st.session_state.vector_store.similarity_search(topic, k=8)
            context = "\n\n".join(chunks)[:4000]
            generator = GroqMCQGenerator(api_key=mcq_api_key)
            mcqs = generator.generate_mcqs(context, topic, num_questions, difficulty, question_types)

            if mcqs:
                st.session_state.mcqs = mcqs
                st.markdown("### ✅ MCQs Generated")

                for i, mcq in enumerate(mcqs):
                    with st.expander(f"Q{i+1}: {mcq['question']}"):
                        for opt_label in ["A", "B", "C", "D"]:
                            option_text = mcq["options"].get(opt_label, "")
                            if opt_label == mcq["correct_answer"]:
                                st.success(f"**{opt_label}.** {option_text}")
                            else:
                                st.write(f"**{opt_label}.** {option_text}")
                        st.caption(f"Explanation: {mcq['explanation']}")

                mcq_df = pd.DataFrame([
                    {
                        "Question": mcq["question"],
                        "Option A": mcq["options"].get("A", ""),
                        "Option B": mcq["options"].get("B", ""),
                        "Option C": mcq["options"].get("C", ""),
                        "Option D": mcq["options"].get("D", ""),
                        "Correct Answer": mcq["correct_answer"],
                        "Explanation": mcq["explanation"]
                    }
                    for mcq in mcqs
                ])

                st.download_button("📥 Download as JSON", json.dumps(mcqs, indent=2), "mcqs.json", "application/json")
                st.download_button("📄 Download as CSV", mcq_df.to_csv(index=False), "mcqs.csv", "text/csv")

                if st.button("🧹 Clear MCQs"):
                    st.session_state.mcqs = []
                    st.session_state.score = 0
                    st.rerun()

        if topic and st.button("📇 Generate Flashcards"):
            chunks = st.session_state.vector_store.similarity_search(topic, k=5)
            context = "\n\n".join(chunks)[:3000]
            flashcard_gen = GroqFlashcardGenerator(api_key=flashcard_api_key)
            flashcards = flashcard_gen.generate_flashcards(context, topic, num_cards=5)

            if flashcards:
                st.markdown("### 📇 Flashcards")
                for i, fc in enumerate(flashcards):
                    with st.expander(f"Card {i+1}: {fc['question']}"):
                        st.success(fc["answer"])

                st.download_button(
                    "📥 Download Flashcards JSON",
                    json.dumps(flashcards, indent=2),
                    "flashcards.json",
                    "application/json"
                )

        # Quiz Mode Section
        if st.session_state.mcqs and st.button("🚀 Start Quiz Mode"):
            st.session_state.quiz_mode = True
            st.session_state.score = 0
            st.session_state.quiz_index = 0
            st.rerun()

        if st.session_state.get("quiz_mode") and st.session_state.mcqs:
            idx = st.session_state.get("quiz_index", 0)
            mcqs = st.session_state.mcqs

            if idx < len(mcqs):
                current = mcqs[idx]
                st.markdown(f"### ❓ Q{idx + 1}: {current['question']}")
                selected = st.radio(
                    "Choose an option:",
                    options=["A", "B", "C", "D"],
                    key=f"quiz_q_{idx}",
                    format_func=lambda x: f"{x}. {current['options'].get(x, '[Missing Option]')}"
                )

                if st.button("Submit Answer"):
                    if selected == current['correct_answer']:
                        st.success("✅ Correct!")
                        st.session_state.score += 1
                    else:
                        st.error(f"❌ Incorrect. Correct answer is {current['correct_answer']}")
                        st.info(f"Explanation: {current['explanation']}")

                    st.session_state.quiz_index += 1
                    time.sleep(1)
                    st.rerun()
            else:
                st.balloons()
                st.success(f"🎉 Quiz Completed! You got {st.session_state.score}/{len(mcqs)} correct.")
                if st.button("🔄 Restart Quiz"):
                    st.session_state.quiz_mode = False
                    st.session_state.quiz_index = 0
                    st.session_state.score = 0
                    st.rerun()


if __name__ == "__main__":
    main()
