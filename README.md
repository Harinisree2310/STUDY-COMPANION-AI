
# 📚 AI Study Companion

**AI Study Companion** is an intelligent learning assistant that helps students study smarter by extracting content from PDFs and generating educational content like summaries, multiple-choice questions (MCQs), and quizzes using advanced AI models. Built with a Streamlit interface and powered by GROQ's LLaMA-3 models, sentence-transformers, and FAISS for contextual understanding and vector search.

---

## ✨ Features

- 📄 Upload one or more PDF files
- 🔍 Extract text content from documents
- ✂️ Chunk text into manageable sections for AI processing
- 📝 Generate concise, topic-based summaries
- 🧠 Generate MCQs based on difficulty and question types (factual, conceptual, application)
- 🚀 Interactive quiz mode with scoring and feedback
- 📥 Download MCQs as JSON or CSV
- 💡 Displays daily motivational quotes for learners

---

## 🧰 Tech Stack

| Layer        | Tools / Libraries                                |
|--------------|--------------------------------------------------|
| **Frontend** | Streamlit                                        |
| **PDF Parsing** | PyPDF2                                       |
| **NLP Embedding** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Similarity Search** | FAISS                                  |
| **LLM Integration** | GROQ API (`llama3-70b-8192`, `llama3-8b-8192`) |
| **Data Processing** | pandas, json, numpy                      |

---

## 🛠️ Installation

###  Clone the repository

```bash
git clone https://github.com/your-username/AI-study-companion.git
cd AI-study-companion
```
### Create a virtual environment
```bash
python -m venv venv
```
#### Activate virtual environment(Windows)
```bash
venv\Scripts\activate
```
### Activate virtual environment(macOS/Linux)
```bash
source venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Running the app
```bash
pip install -r requirements.txt
```

### 📂 File Structure
AI-study-companion/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation



### 📋 Example Use Case
* Upload your class notes or textbook in PDF format
* Enter a topic you'd like to review
* Click "Generate Summary" to get a short, focused explanation
* Click "Generate MCQs" to create practice questions
* Use Quiz Mode to test your knowledge
* Export the questions for offline practice!


### ✅ To-Do / Improvements
 * Flashcard generation
 * OCR support for scanned handwritten notes
 * Login/authentication for personalized user data
 * Save question sets for later review
 * Hugging Face fallback for offline or alternate LLMs

### 🧠 Credits
Streamlit
GROQ
Sentence Transformers
FAISS
PyPDF2


### 📄 License
This project is licensed under the [MIT License](LICENSE).


### 🙋‍♀️ Contributions
Contributions, bug reports, and feature requests are welcome!
Feel free to fork the repo and open a pull request.
