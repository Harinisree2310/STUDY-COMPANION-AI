# 📚 AI Study Companion

An interactive **Streamlit app** that helps students learn smarter by generating **summaries, MCQs, and flashcards** from PDF documents using **GROQ LLMs + embeddings (FAISS + Sentence Transformers)**.  

🚀 Features:
- 📄 Upload and process multiple PDFs  
- 📝 Generate clear, concise summaries  
- 🧠 Auto-generate MCQs with difficulty & type control  
- 📇 Flashcards for active recall  
- 🚀 Quiz Mode to test your knowledge interactively  
- 💡 Motivational quotes to keep you inspired  

---

## 📦 Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/your-username/ai-study-companion.git
cd ai-study-companion
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
Create a **`.env`** file in the project root with your GROQ API keys:
```env
GROQ_API_KEY_MCQ=your_mcq_api_key_here
GROQ_API_KEY_SUMMARY=your_summary_api_key_here
GROQ_API_KEY_FLASHCARD=your_flashcard_api_key_here
```
⚠️ Never commit your real `.env` file. An `.env.example` file is provided for reference.

### 5. Run the app
```bash
streamlit run main_app.py
```

---

## 🖼️ Project Structure
```
ai-study-companion/
│── main_app.py           # Streamlit app
│── requirements.txt      # Python dependencies
│── .env.example          # Example env file (no real keys)
│── .gitignore            # Ignore secrets, cache, venv, etc.
│── README.md             # Documentation
│
├── data/                 # (Optional) sample PDFs
├── assets/               # (Optional) images, logos
```

---

## ✨ How It Works
1. **PDF Upload** → Extracts text & splits into chunks  
2. **Embeddings + FAISS** → Enables semantic search for relevant chunks  
3. **GROQ LLMs** →  
   - 📄 Summaries with `llama3-8b-8192`  
   - 🧠 MCQs with `llama3-70b-8192`  
   - 📇 Flashcards with `llama3-8b-8192`  
4. **Interactive UI** → View, download, or quiz yourself  

---

## ⚡ Example Use Cases
- Study NCERT or research papers  
- Generate quizzes for revision  
- Create flashcards for active recall  
- Practice exam prep with interactive mode  

---

## 💡 Motivational Quotes
Every session gives you a fresh quote to keep you motivated while studying.  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Fork the repo, make your changes, and submit a PR.  

---

## 📜 License
This project is licensed under the MIT License.  

---

### 🎉 Happy Learning with AI Study Companion!
