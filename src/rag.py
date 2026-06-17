import os
import pickle
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Index I/O
def build_message_index(vectorizer, X, messages: List[str], out_dir: str):
    """Persist the TF-IDF vectoriser + message matrix to disk."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'index.pkl'), 'wb') as f:
        pickle.dump({'messages': messages, 'X': X, 'vectorizer': vectorizer}, f)


def load_index(out_dir: str) -> Dict[str, Any]:
    with open(os.path.join(out_dir, 'index.pkl'), 'rb') as f:
        return pickle.load(f)


# Retrieval 
def retrieve(query: str, index: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """Return top-K messages ranked by TF-IDF cosine similarity to query."""
    vec  = index['vectorizer'].transform([query])
    sims = cosine_similarity(vec, index['X']).flatten()
    idxs = list(reversed(sims.argsort()))[:top_k]
    return [
        {
            'idx':     int(i),
            'score':   float(sims[i]),
            'message': index['messages'][i],
        }
        for i in idxs
        if sims[i] > 0
    ]


def retrieve_topics(query: str, topics: List[Dict[str, Any]],
                    index: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Return top-K topic summaries ranked by cosine similarity to query."""
    summaries = [t.get('summary', '') for t in topics]
    if not summaries:
        return []
    vecs = index['vectorizer'].transform(summaries)
    qvec = index['vectorizer'].transform([query])
    sims = cosine_similarity(qvec, vecs).flatten()
    idxs = list(reversed(sims.argsort()))[:top_k]
    results = []
    for i in idxs:
        if sims[i] > 0:
            t = dict(topics[i])
            t['score'] = float(sims[i])
            results.append(t)
    return results


# Answer Synthesis
def synthesize_answer(query: str,
                      topic_hits: List[Dict[str, Any]],
                      msg_hits: List[Dict[str, Any]],
                      persona: Dict[str, Any]) -> str:
    """
    Combine retrieved context into a coherent natural-language answer using LLM
    or fallback to the rule-based synthesizer.
    """
    from src.llm import is_llm_active, generate_answer

    # If LLM is not active, run the fallback directly
    if not is_llm_active():
        return _synthesize_answer_fallback(query, topic_hits, msg_hits, persona)

    # Format topic context
    topic_context_str = ""
    for t in topic_hits:
        key_pts = ", ".join(t.get('key_points', []))
        parts_str = ", ".join(t.get('participants', []))
        topic_context_str += (
            f"- Topic {t.get('topic_number')}: {t.get('title')} "
            f"(Messages {t.get('start_idx')}–{t.get('end_idx')})\n"
            f"  Summary: {t.get('summary')}\n"
            f"  Participants: {parts_str}\n"
            f"  Sentiment: {t.get('sentiment', 'neutral')}\n"
            f"  Key Points: {key_pts}\n\n"
        )

    msg_context_str = ""
    for h in msg_hits:
        msg_context_str += f"- [msg #{h['idx'] + 1}] \"{h['message']}\"\n"

    # Compute fallback answer to pass to the generator as fallback
    fallback_ans = _synthesize_answer_fallback(query, topic_hits, msg_hits, persona)

    return generate_answer(query, topic_context_str, msg_context_str, persona, fallback_ans)


def _synthesize_answer_fallback(query: str,
                                topic_hits: List[Dict[str, Any]],
                                msg_hits: List[Dict[str, Any]],
                                persona: Dict[str, Any]) -> str:

    q_lower = query.lower()
    parts: List[str] = []

    # Persona queries
    persona_triggers = [
        'what kind of person', 'who is', 'describe the user', 'tell me about',
        'personality', 'character', 'person',
    ]
    habit_triggers = ['habit', 'routine', 'daily', 'sleep', 'coffee', 'food', 'eat', 'work']
    style_triggers = ['talk', 'communicate', 'style', 'write', 'message', 'express', 'tone', 'emoji']

    is_persona_q = any(t in q_lower for t in persona_triggers)
    is_habit_q   = any(t in q_lower for t in habit_triggers)
    is_style_q   = any(t in q_lower for t in style_triggers)

    if is_persona_q and persona:
        traits = persona.get('personality_traits', [])
        facts  = persona.get('personal_facts', [])
        top_topics = persona.get('top_topics', [])[:6]
        sentiment = persona.get('emotional_profile', {}).get('overall_sentiment', 'unknown')
        s = "**About this user:**\n"
        if traits:
            s += f"• Personality: {'; '.join(traits[:4])}\n"
        if facts:
            s += f"• Personal facts: {'; '.join(facts[:4])}\n"
        if top_topics:
            s += f"• Often talks about: {', '.join(top_topics)}\n"
        s += f"• Overall emotional tone: {sentiment}\n"
        parts.append(s)

    if is_habit_q and persona:
        habits = persona.get('habits', [])
        s = "**Habits & Routines:**\n"
        if habits:
            for h in habits[:6]:
                s += f"• {h}\n"
        else:
            s += "• No clear habit patterns detected in the conversation.\n"
        parts.append(s)

    if is_style_q and persona:
        cs = persona.get('communication_style', {})
        s = "**Communication Style:**\n"
        s += f"• Style: {cs.get('style_label', 'N/A')}\n"
        s += f"• Avg words/message: {cs.get('avg_words_per_message', 'N/A')}\n"
        s += f"• Emoji usage: {cs.get('emoji_ratio', 0):.0%} of messages contain emojis\n"
        s += f"• Question rate: {cs.get('question_rate', 0):.0%}\n"
        ex = cs.get('examples', {})
        if ex.get('humor_message'):
            s += f"• Humor example: \"{ex['humor_message']}\"\n"
        if ex.get('emoji_message'):
            s += f"• Emoji example: \"{ex['emoji_message']}\"\n"
        parts.append(s)

    # Topic context
    if topic_hits:
        s = "**Relevant conversation topics:**\n"
        for t in topic_hits[:2]:
            s += (f"• **Topic {t.get('topic_number','?')}: {t.get('title','...')}** "
                  f"(messages {t.get('start_idx','?')}–{t.get('end_idx','?')})\n"
                  f"  Summary: {t.get('summary','')}\n")
        parts.append(s)

    # Message evidence
    if msg_hits:
        s = "**Relevant messages from the conversation:**\n"
        for h in msg_hits[:4]:
            s += f"• [msg #{h['idx']+1}] \"{h['message'][:200]}\"\n"
        parts.append(s)

    if not parts:
        parts.append("I couldn't find specific information about that in the conversation data. "
                     "Try processing a CSV first, or rephrase the question.")

    return '\n'.join(parts)
