# 🧭 AI-Powered Database Search

This is a microservicesemantic search engine that helps users search a database file tailored to a user profile. It uses vector embeddings and keyword matching to deliver personalized, high-quality recommendations through a fast, responsive UI.

---

## 🌟 Features

- 🔍 **Natural Language Search**: Just describe what you're looking for.
- 🤖 **AI-Powered Matching**: Uses Sentence Transformers for semantic similarity.
- 👥 **User Profiles**: Choose from multiple preset personas (teen volunteer, hedge fund manager, etc.).
- 💡 **Contextual Suggestions**: Get relevant results even without a search query.
- 🌑 **Dark Mode UI**: Built with React + CSS for a clean, accessible experience.
- ⚡ **FastAPI Backend**: Efficient API with pre-computed vector caching.

---

## 🧱 Tech Stack

| Frontend        | Backend         | ML/NLP        | Other               |
|----------------|-----------------|---------------|---------------------|
| React + Vite   | FastAPI + Uvicorn | `sentence-transformers` (`all-MiniLM-L6-v2`) | CSS, JSON, NumPy, Pickle |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/volunteergo.git
cd volunteergo
```

### 2. Backend Setup (FastAPI)

#### Install dependencies:

```bash
cd search-backend
pip install -r requirements.txt
```

Make sure `final_opps.json` and `profiles.json` are in the backend directory.

#### Run the API server:

```bash
uvicorn main:app --reload
```

By default, this runs on [http://localhost:8000](http://localhost:8000)

---

### 3. Frontend Setup (React)

```bash
cd ../search-ui
npm install
npm run dev
```

This runs the UI on [http://localhost:5173](http://localhost:5173)

---

---

## 🧪 Example Search Prompts

- `"I want to help children in after-school programs"`
- `"Looking for weekend nature clean-up activities"`
- `"Corporate team volunteering opportunities"`
- `"I’m a student good at math and want to tutor"`

---

## 📌 Notes

- Opportunities are vectorized using SentenceTransformer (`all-MiniLM-L6-v2`).
- Descriptions are truncated for concise display.
- Scores are based on cosine similarity and optional keyword relevance.

---

## 🙌 Credits

Built by Jasmine Andresol.

---
