import os
import json
import re
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Constants
SIM_THRESHOLD = 0.20       # cosine similarity below this → topic drift
DRIFT_WINDOW  = 3          # messages in a row below threshold to trigger split
MIN_TOPIC_LEN = 5          # minimum messages to form a topic segment
CHECKPOINT_N  = 100        # fixed-length checkpoint size


# CSV Loader
def load_csv(path: str) -> pd.DataFrame:
    """Load conversation CSV. Auto-detects timestamp/sender/message columns."""
    df = pd.read_csv(path)
    mapping: Dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if 'time' in lc or 'date' in lc:
            mapping.setdefault('timestamp', c)
        elif any(k in lc for k in ('sender', 'from', 'author', 'user', 'name')):
            mapping.setdefault('sender', c)
        elif any(k in lc for k in ('message', 'text', 'content', 'body', 'msg')):
            mapping.setdefault('message', c)

    if 'message' not in mapping:
        mapping['message'] = df.columns[-1]          # fallback: last column
    if 'timestamp' not in mapping:
        df['__ts'] = pd.RangeIndex(len(df))
        mapping['timestamp'] = '__ts'
    if 'sender' not in mapping:
        df['__sender'] = 'unknown'
        mapping['sender'] = '__sender'

    # Build rename dict and apply it
    rename_dict = {v: k for k, v in mapping.items()}
    df = df.rename(columns=rename_dict)
    
    # Ensure all three columns exist
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.RangeIndex(len(df))
    if 'sender' not in df.columns:
        df['sender'] = 'unknown'
    if 'message' not in df.columns:
        df['message'] = ''
    
    df = df[['timestamp', 'sender', 'message']].copy()
    df['message'] = df['message'].fillna('').astype(str)
    # Sort chronologically if timestamp is parseable
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception:
        pass
    return df


# Summarizer
def summarize_text(messages: List[str], idf_lookup: Dict[str, float], top_n: int = 4) -> str:
    """Extractive summariser: score sentences by IDF, pick top-N."""
    text = ' '.join(messages)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= top_n:
        return text.strip()[:800]
    scored = []
    for s in sentences:
        words = re.findall(r'\w+', s.lower())
        score = sum(idf_lookup.get(w, 0.0) for w in words)
        scored.append((score, s))
    scored.sort(reverse=True)
    return ' '.join(s for _, s in scored[:top_n]).strip()


def _topic_title(messages: List[str], vectorizer: TfidfVectorizer, top_n: int = 3) -> str:
    """Generate a short label from top TF-IDF terms of the segment."""
    if not messages:
        return 'General'
    try:
        vecs = vectorizer.transform(messages)
        mean = np.asarray(vecs.mean(axis=0)).flatten()
        feat_names = vectorizer.get_feature_names_out()
        top_indices = mean.argsort()[::-1][:top_n]
        terms = [feat_names[i] for i in top_indices if mean[i] > 0]
        return ', '.join(terms).title() if terms else 'General'
    except Exception:
        return 'General'


# Topic Detection
def process_topics(df: pd.DataFrame,
                   sim_threshold: float = SIM_THRESHOLD,
                   drift_window: int = DRIFT_WINDOW) -> Dict[str, Any]:
    """
    Detect topic segments chronologically using sliding-window centroid drift.

    Algorithm:
      1. Fit TF-IDF over all messages once (global vocabulary).
      2. Iterate messages in order, maintaining a running centroid for the
         current segment.
      3. If cosine(current_message, centroid) < threshold for `drift_window`
         consecutive messages, seal the current topic and start a new one.
      4. The sealed topic gets an extractive summary and a keyword-derived title.
    """
    messages = df['message'].tolist()
    n = len(messages)

    vectorizer = TfidfVectorizer(max_features=8000, stop_words='english',
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(messages)

    idf_lookup = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    topics: List[Dict[str, Any]] = []
    current_idxs: List[int] = []
    centroid = None
    low_sim_streak = 0       # consecutive low-similarity messages

    def seal_topic(idxs: List[int]):
        # Format for LLM
        messages_data = []
        for j in idxs:
            messages_data.append({
                'idx': j + 1,
                'sender': df.loc[j, 'sender'],
                'message': messages[j]
            })
            
        from src.llm import summarize_topic_segment
        llm_data = summarize_topic_segment(messages_data)
        
        # Override basic titles if LLM is not active or returned generic
        title = llm_data.get('title')
        if not title or title.startswith("Segment "):
            seg_msgs = [messages[j] for j in idxs]
            title = _topic_title(seg_msgs, vectorizer)
            
        topics.append({
            'topic_number': len(topics) + 1,
            'title':        title,
            'start_idx':    idxs[0] + 1,   # 1-based
            'end_idx':      idxs[-1] + 1,
            'message_count': len(idxs),
            'summary':      llm_data.get('summary', ''),
            'key_points':   llm_data.get('key_points', []),
            'participants': llm_data.get('participants', []),
            'sentiment':    llm_data.get('sentiment', 'neutral'),
            'key_quotes':   llm_data.get('key_quotes', [])
        })

    for i in range(n):
        v = X[i]
        if centroid is None:
            current_idxs = [i]
            centroid = v.copy().astype(float)
            low_sim_streak = 0
            continue

        sim = float(cosine_similarity(v, centroid)[0, 0])

        if sim < sim_threshold:
            low_sim_streak += 1
        else:
            low_sim_streak = 0
            current_idxs.append(i)
            centroid = centroid + v

        if low_sim_streak >= drift_window:
            if len(current_idxs) >= MIN_TOPIC_LEN:
                seal_topic(current_idxs)
            # Start new topic from the drift window's start
            drift_start = i - low_sim_streak + 1
            current_idxs = list(range(drift_start, i + 1))
            centroid = X[drift_start].copy().astype(float)
            for j in range(drift_start + 1, i + 1):
                centroid = centroid + X[j]
            low_sim_streak = 0

    # seal last segment
    if current_idxs:
        seal_topic(current_idxs)

    # 100-message checkpoints (independent of topics)
    checkpoints: List[Dict[str, Any]] = []
    from src.llm import summarize_checkpoint
    for start in range(0, n, CHECKPOINT_N):
        end = min(start + CHECKPOINT_N, n)
        seg_msgs = messages[start:end]
        fallback_summary = summarize_text(seg_msgs, idf_lookup)
        summary = summarize_checkpoint(seg_msgs, fallback_summary)
        checkpoints.append({
            'checkpoint_number': len(checkpoints) + 1,
            'start_idx':  start + 1,
            'end_idx':    end,
            'message_count': end - start,
            'summary':    summary,
        })

    return {
        'topics':          topics,
        'checkpoints_100': checkpoints,
        'vectorizer':      vectorizer,
        'X':               X,
        'idf_lookup':      idf_lookup,
        'messages':        messages,
    }


# Persistence
def save_outputs(out_dir: str, results: Dict[str, Any], df: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'topics.json'), 'w', encoding='utf8') as f:
        json.dump({'topics': results['topics']}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'checkpoints_100.json'), 'w', encoding='utf8') as f:
        json.dump({'checkpoints_100': results['checkpoints_100']}, f,
                  indent=2, ensure_ascii=False)
    df.to_csv(os.path.join(out_dir, 'messages.csv'), index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    df = load_csv(args.csv)
    res = process_topics(df)
    save_outputs(args.out, res, df)
    print(f"Detected {len(res['topics'])} topics, "
          f"{len(res['checkpoints_100'])} 100-msg checkpoints.")
