"""
app.py — Flask entrypoint for the Conversation RAG Chatbot.

Routes:
  GET  /                  → Chatbot UI
  POST /api/process       → Load CSV, detect topics, extract persona, build index
  POST /api/query         → RAG query (topic + message retrieval + synthesis)
  GET  /api/status        → Check if data has been processed
  GET  /api/topics        → Return all topic checkpoints
  GET  /api/checkpoints   → Return all 100-msg checkpoints
  GET  /api/persona       → Return extracted persona JSON
"""

import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from flask import Flask, request, render_template, jsonify

from src.processor import load_csv, process_topics, save_outputs
from src.persona   import extract_persona
from src.rag       import (
    build_message_index, load_index,
    retrieve, retrieve_topics, synthesize_answer,
)

app      = Flask(__name__)
DATA_DIR = 'outputs'


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load_persona() -> dict:
    p = os.path.join(DATA_DIR, 'persona.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf8') as f:
            return json.load(f)
    return {}


def _load_topics() -> list:
    p = os.path.join(DATA_DIR, 'topics.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf8') as f:
            return json.load(f).get('topics', [])
    return []


def _load_checkpoints() -> list:
    p = os.path.join(DATA_DIR, 'checkpoints_100.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf8') as f:
            return json.load(f).get('checkpoints_100', [])
    return []


def _is_processed() -> bool:
    # Consider data "processed" only when an index exists and a file was uploaded via the UI
    idx = os.path.join(DATA_DIR, 'index.pkl')
    uploaded = os.path.join(DATA_DIR, 'uploaded.csv')
    return os.path.exists(idx) and os.path.exists(uploaded)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    processed = _is_processed()
    topics     = _load_topics()     if processed else []
    checkpoints = _load_checkpoints() if processed else []
    return jsonify({
        'processed':       processed,
        'topic_count':     len(topics),
        'checkpoint_count': len(checkpoints),
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """
    Process a conversation CSV.
    Accepts JSON body: { "csv": "<path or filename>" }
    Also accepts multipart file upload via 'file' field.
    """
    # ── File upload path ──────────────────────────────────────────────────────
    if 'file' in request.files:
        f = request.files['file']
        if not f or f.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, 'uploaded.csv')
        f.save(csv_path)
    else:
        # Do not allow implicit demo/sample processing from the frontend.
        # Require either a multipart upload (handled above) or an explicit existing path.
        data = request.get_json(silent=True) or {}
        csv_path = data.get('csv', '')
        if not csv_path or not os.path.exists(csv_path):
            return jsonify({'error': 'CSV path missing or not found. Upload a file via the UI or pass {"csv":"<existing-path>"}'}), 400

    try:
        print(f"[DEBUG] Loading CSV from: {csv_path}")
        df = load_csv(csv_path)
        print(f"[DEBUG] CSV loaded successfully: {len(df)} rows, columns: {list(df.columns)}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load CSV: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to read CSV: {str(e)}. Make sure it has message/text/content column.'}), 400

    try:
        res = process_topics(df)
        save_outputs(DATA_DIR, res, df)
        build_message_index(res['vectorizer'], res['X'], res['messages'], DATA_DIR)

        persona = extract_persona(df)
        with open(os.path.join(DATA_DIR, 'persona.json'), 'w', encoding='utf8') as fp:
            json.dump(persona, fp, indent=2, ensure_ascii=False)
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    return jsonify({
        'status':          'processed',
        'total_messages':  len(df),
        'topics_detected': len(res['topics']),
        'checkpoints':     len(res['checkpoints_100']),
        'persona_habits':  len(persona.get('habits', [])),
    })


@app.route('/api/query', methods=['POST'])
def api_query():
    """
    RAG query endpoint.
    Body: { "query": "<question>" }
    """
    if not _is_processed():
        return jsonify({'error': 'Data not processed yet. Call /api/process first.'}), 400

    data  = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'query field is required'}), 400

    idx      = load_index(DATA_DIR)
    topics   = _load_topics()
    persona  = _load_persona()

    # Dual-layer retrieval
    msg_hits   = retrieve(query, idx, top_k=5)
    topic_hits = retrieve_topics(query, topics, idx, top_k=3)

    answer = synthesize_answer(query, topic_hits, msg_hits, persona)

    return jsonify({
        'answer':      answer,
        'topic_hits':  topic_hits,
        'msg_hits':    msg_hits,
    })


@app.route('/api/topics')
def api_topics():
    if not _is_processed():
        return jsonify({'error': 'Not processed yet'}), 400
    return jsonify({'topics': _load_topics()})


@app.route('/api/checkpoints')
def api_checkpoints():
    if not _is_processed():
        return jsonify({'error': 'Not processed yet'}), 400
    return jsonify({'checkpoints_100': _load_checkpoints()})


@app.route('/api/persona')
def api_persona():
    if not _is_processed():
        return jsonify({'error': 'Not processed yet'}), 400
    return jsonify(_load_persona())


# ─── CLI helper ───────────────────────────────────────────────────────────────
def _cli_process(csv_path: str):
    print(f"Processing: {csv_path}")
    try:
        df = load_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    try:
        res = process_topics(df)
        save_outputs(DATA_DIR, res, df)
        build_message_index(res['vectorizer'], res['X'], res['messages'], DATA_DIR)
        persona = extract_persona(df)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, 'persona.json'), 'w', encoding='utf8') as fp:
            json.dump(persona, fp, indent=2, ensure_ascii=False)
        print(f"Success: {len(df)} messages | {len(res['topics'])} topics | "
              f"{len(res['checkpoints_100'])} 100-msg checkpoints")
        print(f"Success: Persona saved -> {DATA_DIR}/persona.json")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conversation RAG Chatbot')
    parser.add_argument('--process', metavar='CSV', help='Process a CSV before starting server')
    parser.add_argument('--host',    default='0.0.0.0')
    parser.add_argument('--port',    type=int, default=5000)
    parser.add_argument('--no-debug', action='store_true')
    args = parser.parse_args()

    if args.process:
        _cli_process(args.process)

    app.run(host=args.host, port=args.port, debug=not args.no_debug)
