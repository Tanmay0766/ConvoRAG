# ConvoRAG

Conversation intelligence chatbot powered by local-first RAG. Extract topics, detect personality, and chat with conversation history — no external APIs.

**Live Demo:** https://web-production-fbb03.up.railway.app/

## Features

- **Topic Detection** — Automatic semantic topic segmentation using TF-IDF
- **Persona Extraction** — Habits, personality traits, communication style from message patterns
- **Dual-Layer Retrieval** — Topic context + individual message evidence
- **No External APIs** — Runs entirely locally with scikit-learn

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Run
python app.py

# 3. Open browser
http://localhost:5000
```

Upload a CSV with columns: `timestamp`, `sender`, `message` (names flexible).

## CSV Format

```
timestamp,sender,message
2024-01-15 08:05:00,Alice,Good morning!
2024-01-15 09:30:00,Bob,Hey Alice!
```

Column names auto-detected: `date/time`, `from/author`, `text/content/body`, etc.

## Tech Stack

- **Backend:** Flask + Python
- **NLP:** scikit-learn (TF-IDF, cosine similarity)
- **Frontend:** Vanilla HTML/CSS/JS
- **Deployment:** Gunicorn on Railway/Render

## Project Structure

```
src/
├── processor.py   # CSV loading, topic detection, checkpoints
├── persona.py     # Persona extraction from message patterns
├── rag.py         # Retrieval and answer synthesis
app.py             # Flask API
templates/
└── index.html     # Chat UI
```

## Deployment

Deployed on Railway. Auto-deploys on GitHub push.

To self-host:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```
├── Procfile                  # Heroku/Railway
└── render.yaml               # Render.com
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process` | Upload CSV (multipart) or `{"csv":"path"}` |
| `POST` | `/api/query` | `{"query":"..."}` → RAG answer |
| `GET` | `/api/topics` | All topic checkpoints |
| `GET` | `/api/checkpoints` | All 100-msg checkpoints |
| `GET` | `/api/persona` | Persona JSON |
| `GET` | `/api/status` | Processing status |
