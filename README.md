# ConvoRAG — Conversation Intelligence Chatbot

A **local-first RAG system** that processes chronological conversation CSVs to detect topic shifts, create message checkpoints, extract user persona, and power an intelligent chatbot — **no external AI APIs required**.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Process your CSV + start server

```bash
# Process a CSV and launch
python app.py --process sample_conversations.csv

# Or just start the server (use the UI to upload)
python app.py
```

### 3. Open browser

```
http://localhost:5000
```

Upload a CSV (or click **Load Demo Data**), then ask questions in the chat.

---

## CSV Format

Auto-detected — any CSV with columns named like:

| timestamp | sender | message |
|-----------|--------|---------|
| 2024-01-15 08:05:00 | Alice | Good morning! |

Column names are flexible: `date/time`, `from/author`, `text/content/body` all work.

---

## Architecture & Design Decisions

### Part 1 — RAG System with Checkpoints

#### Topic Change Detection

**Algorithm: Sliding-Window Centroid Drift on TF-IDF Vectors**

```
For each message (chronological):
  1. Compute TF-IDF vector
  2. Compare to running centroid of current segment (cosine similarity)
  3. If similarity < threshold for DRIFT_WINDOW consecutive messages → seal topic
  4. Start new topic segment, reset centroid
```

**Why this approach:**
- **TF-IDF** (not embeddings): No GPU, no API, runs on any machine in <1s. Interpretable — you can inspect which words drive the topic.
- **Centroid drift** (not fixed windows): Topics have natural, variable lengths. A fixed 20-message window would split mid-topic. Centroid drift detects *semantic* shifts.
- **Sliding window** (`DRIFT_WINDOW=3`): One off-topic message (noise) won't fragment topics. 3 consecutive low-similarity messages signal a genuine change.
- **Topic titles**: Extracted from top TF-IDF terms of the segment — cheap, readable, no LLM needed.

**Output:**
```
Topic 1 → messages 1–25   → title: "Work, Project, Deadline"   → summary
Topic 2 → messages 26–40  → title: "Food, Pizza, Lunch"        → summary
Topic 3 → messages 41–90  → title: "Travel, Bali, Vacation"    → summary
```

#### 100-Message Checkpoints

Independent of topic boundaries. Every 100 chronological messages gets an extractive summary (top IDF-scored sentences). These serve as a "chapter index" over the full conversation history.

#### Query Handling (Dual-Layer Retrieval)

```
Query
  │
  ├─ Layer 1: Topic retrieval
  │   Transform query → TF-IDF vector
  │   Cosine similarity vs. all topic summaries
  │   Return top-K topic contexts (breadth)
  │
  └─ Layer 2: Message retrieval
      Cosine similarity vs. all individual messages
      Return top-K message hits (specificity/evidence)

Answer synthesis:
  Persona data + topic summaries + message quotes → coherent response
```

**Why two layers?** Topic summaries give *context* (what was being discussed). Individual messages give *evidence* (exact quotes). Combining both yields answers that are both informed and grounded.

---

### Part 2 — User Persona Extraction

**Design principle: every field must be derived from a measurable signal, never guessed.**

| Persona Field | Signal Source |
|--------------|---------------|
| **Habits** | Timestamp hours (0–5 AM = late night), food/work/travel keyword frequency, peak active hour |
| **Personal facts** | Regex patterns: `married\|wife\|husband`, `kid\|children`, `moved\|relocat`, `gym\|workout` |
| **Personality traits** | Laugh expressions (`lol/haha/lmao`), emoji Unicode ranges, `!` count, positive/negative word ratio |
| **Communication style** | Avg/median words per message, emoji ratio, question rate, exclamation rate |
| **Emotional profile** | Positive vs negative word counts against curated word lists |
| **Top topics** | Most frequent non-stop tokens from full conversation |

All outputs include **`signal_counts`** — raw numbers showing *why* each conclusion was drawn. Transparent, auditable, not a black box.

**Output (JSON):**
```json
{
  "habits": ["coffee drinker", "late-night active", "food-conscious"],
  "personal_facts": ["married / has a spouse", "has or mentions pets"],
  "personality_traits": ["has a great sense of humor", "generally positive"],
  "communication_style": {
    "style_label": "concise (casual texting style)",
    "avg_words_per_message": 8.4,
    "emoji_ratio": 0.23
  },
  "emotional_profile": { "overall_sentiment": "positive" },
  "top_topics": ["work", "food", "sleep", "travel"],
  "signal_counts": { "total_messages": 120, "emoji_total": 28, ... }
}
```

---

### Part 3 — Chatbot

**Stack:** Flask (server) + Vanilla JS (client) — no React/Vue overhead.

**Answer synthesis logic:**
1. Detect query *intent* from keywords (persona query? habit query? style query?)
2. Run dual-layer retrieval (topic + message)
3. Fill a structured template with retrieved data + persona JSON
4. Return formatted markdown answer

**Special queries handled:**
- `"What kind of person is this user?"` → personality traits + facts + sentiment
- `"What are their habits?"` → habits list from persona
- `"How do they communicate?"` → style metrics + examples
- General queries → relevant topic summaries + quoted messages

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Web server | Flask | Lightweight, zero config, easy to deploy |
| Vectorisation | scikit-learn TF-IDF | No GPU, fast, interpretable |
| Similarity | Cosine similarity | Standard for sparse vectors |
| Index storage | Pickle | Simple, no DB dependency |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, instant load |
| Deployment | Gunicorn + Render/Railway | Free tier, Python-native |

---

## Deployment (Render.com)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect repo
3. It auto-detects `render.yaml` — click Deploy
4. Your app is live at `https://convorag.onrender.com`

Or manually:
```bash
gunicorn app:app --bind 0.0.0.0:8000 --timeout 120
```

---

## Project Structure

```
.
├── app.py                    # Flask app + API routes
├── src/
│   ├── processor.py          # CSV loading, topic detection, checkpoints
│   ├── persona.py            # Persona extraction engine
│   └── rag.py                # Retrieval + answer synthesis
├── templates/
│   └── index.html            # Premium chatbot UI
├── outputs/                  # Generated after processing (gitignored)
│   ├── topics.json
│   ├── checkpoints_100.json
│   ├── persona.json
│   └── index.pkl
├── sample_conversations.csv  # Demo dataset
├── requirements.txt
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
