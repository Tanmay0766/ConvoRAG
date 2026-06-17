import os
import json
import re
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

# Read configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower().strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip()

# Set defaults if not provided
if LLM_PROVIDER == "gemini" and not LLM_MODEL:
    LLM_MODEL = "gemini-2.5-flash"
elif LLM_PROVIDER == "openai" and not LLM_MODEL:
    LLM_MODEL = "gpt-4o-mini"
elif LLM_PROVIDER == "ollama":
    if not LLM_MODEL:
        LLM_MODEL = "gemma2:2b"
    if not LLM_API_BASE:
        LLM_API_BASE = "http://localhost:11434/v1"


def is_llm_active() -> bool:
    return LLM_PROVIDER in ("gemini", "openai", "ollama")


def _call_llm(prompt: str, system_instruction: str = "", json_mode: bool = False) -> str:
    if not is_llm_active():
        raise ValueError("No LLM provider is active.")

    # GEMINI API
    if LLM_PROVIDER == "gemini":
        if not LLM_API_KEY:
            raise ValueError("LLM_API_KEY is required for Gemini provider.")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent?key={LLM_API_KEY}"
        
        headers = {"Content-Type": "application/json"}
        
        # Build contents structure
        parts = []
        if system_instruction:
            parts.append({"text": f"System Instructions:\n{system_instruction}\n\n"})
        parts.append({"text": prompt})
        
        payload = {
            "contents": [{
                "parts": parts
            }]
        }
        
        if json_mode:
            payload["generationConfig"] = {
                "responseMimeType": "application/json"
            }
            
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        
        try:
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from Gemini API: {res_json}") from e

    # OPENAI OR OLLAMA
    else:
        # Both use OpenAI Chat Completion format
        if LLM_PROVIDER == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            if not LLM_API_KEY:
                raise ValueError("LLM_API_KEY is required for OpenAI provider.")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_KEY}"
            }
        else: # ollama
            url = f"{LLM_API_BASE}/chat/completions"
            headers = {
                "Content-Type": "application/json"
            }
            
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.3
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
            
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        res_json = response.json()
        
        try:
            return res_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format from {LLM_PROVIDER} API: {res_json}") from e


# Topic Segment Summarization
def summarize_topic_segment(messages_data: List[Dict[str, Any]]) -> Dict[str, Any]:

    # 1. Format the conversation transcript for LLM input
    formatted_chat = []
    for m in messages_data:
        formatted_chat.append(f"[msg #{m['idx']}] {m['sender']}: {m['message']}")
    chat_text = "\n".join(formatted_chat)
    
    # 2. Extract basic local metadata for fallback/enrichment
    senders = sorted(list(set(m['sender'] for m in messages_data)))
    start_idx = messages_data[0]['idx']
    end_idx = messages_data[-1]['idx']
    msg_count = len(messages_data)

    if not is_llm_active():
        return _local_topic_fallback(messages_data, chat_text, senders, start_idx, end_idx, msg_count)

    system_instr = (
        "You are an expert conversation analyst. You summarize conversation segments. "
        "You MUST respond ONLY with a valid JSON object matching the requested schema. "
        "Do not wrap the response in markdown code blocks like ```json."
    )
    
    prompt = f"""Analyze this chronological segment of a conversation and output a JSON object.
The conversation segment:
\"\"\"
{chat_text}
\"\"\"

The JSON object MUST contain the following fields:
- "title": (string) A short, specific title (3-6 words) describing the core topic. Avoid generic titles like "General chat".
- "summary": (string) A detailed 2-3 sentence summary explaining what was discussed, what was decided, or what occurred.
- "key_points": (list of strings) 3-5 bullet points capturing the main details, decisions, or questions raised.
- "participants": (list of strings) Senders active in this segment.
- "sentiment": (string) Overall emotional tone (e.g., "collaborative", "stressful", "enthusiastic", "casual", "frustrated").
- "key_quotes": (list of strings) 1-2 important raw quotes verbatim from the messages, including the sender name (e.g., "Alice: 'I love sushi!'").

Format your output exactly as this JSON schema (do not output any other text or markers):
{{
  "title": "...",
  "summary": "...",
  "key_points": [
    "...",
    "..."
  ],
  "participants": [
    "..."
  ],
  "sentiment": "...",
  "key_quotes": [
    "..."
  ]
}}
"""
    try:
        raw_res = _call_llm(prompt, system_instruction=system_instr, json_mode=True)
        # Strip codeblock wrappers if LLM still returned them
        raw_res = re.sub(r"^```json\s*", "", raw_res, flags=re.IGNORECASE)
        raw_res = re.sub(r"\s*```$", "", raw_res, flags=re.IGNORECASE).strip()
        data = json.loads(raw_res)
        
        # Enforce boundary indices
        data['start_idx'] = start_idx
        data['end_idx'] = end_idx
        data['message_count'] = msg_count
        return data
    except Exception as e:
        print(f"LLM topic summarization failed: {e}. Falling back to rule-based.")
        return _local_topic_fallback(messages_data, chat_text, senders, start_idx, end_idx, msg_count)


def _local_topic_fallback(messages_data: List[Dict[str, Any]], chat_text: str, 
                          senders: List[str], start_idx: int, end_idx: int, 
                          msg_count: int) -> Dict[str, Any]:
    """Generates detailed topic metadata using local rule-based heuristics."""
    # Build a simple extractive summary (first few sentences/messages)
    text_content = [m['message'] for m in messages_data if m['message'].strip()]
    summary_text = " ".join(text_content[:3])
    if len(summary_text) > 200:
        summary_text = summary_text[:197] + "..."
        
    # Build key points (simple list of first few messages)
    key_points = [m['message'][:80] + ("..." if len(m['message']) > 80 else "") 
                  for m in messages_data[:4] if m['message'].strip()]
    
    # Determine sentiment based on simple lexicons
    pos_words = {"love", "good", "great", "awesome", "perfect", "thanks", "excited", "happy"}
    neg_words = {"sad", "bad", "terrible", "awful", "tired", "stressed", "no", "never"}
    words = re.findall(r"\w+", chat_text.lower())
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    sentiment = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
    
    # Generate a basic title using TF-IDF keywords
    title = f"Segment {start_idx} to {end_idx}"
    
    # Key quotes
    quotes = []
    if messages_data:
        quotes.append(f"{messages_data[0]['sender']}: '{messages_data[0]['message'][:100]}'")
        if len(messages_data) > 1:
            quotes.append(f"{messages_data[-1]['sender']}: '{messages_data[-1]['message'][:100]}'")

    return {
        "topic_number": 0,  # Set by processor
        "title": title,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "message_count": msg_count,
        "summary": summary_text or "No text content.",
        "key_points": key_points or ["No key points extracted."],
        "participants": senders,
        "sentiment": sentiment,
        "key_quotes": quotes
    }


# 100-Message Checkpoint Summarization
def summarize_checkpoint(messages: List[str], fallback_summary: str) -> str:
    """Summarize a 100-message block. Falls back to precomputed summary if LLM fails."""
    if not is_llm_active():
        return fallback_summary

    prompt = f"""Provide a concise, single-paragraph summary of the following 100-message conversation segment:
\"\"\"
{" ".join(messages)}
\"\"\"
Summary:"""
    try:
        return _call_llm(prompt, system_instruction="Write a concise paragraph summary of the conversation segment.").strip()
    except Exception as e:
        print(f"LLM checkpoint summarization failed: {e}. Using fallback.")
        return fallback_summary


# Persona Synthesis
def generate_persona(messages_summary: str, raw_facts: List[str], 
                     style_metrics: Dict[str, Any], fallback_persona: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize a rich persona profile from heuristic inputs.
    """
    if not is_llm_active():
        return fallback_persona

    system_instr = (
        "You are an expert character analyst. Synthesize a user persona. "
        "You MUST respond ONLY with a valid JSON object matching the requested schema. "
        "Do not wrap the response in markdown code blocks like ```json."
    )
    
    prompt = f"""Synthesize a structured user persona based on the following conversation statistics and facts.
- **Top topics & summary of chats**: "{messages_summary}"
- **Detected personal facts**: {json.dumps(raw_facts)}
- **Style Metrics**: {json.dumps(style_metrics)}

Output a JSON object conforming exactly to this schema:
{{
  "personality_traits": [
    "Trait 1 (with detail, e.g., 'generally positive and upbeat in tone')",
    "Trait 2..."
  ],
  "habits": [
    "Habit 1 (e.g., 'active late at night, often between midnight and 5am')",
    "Habit 2..."
  ],
  "communication_style": {{
    "style_label": "e.g., concise, verbose, casual texting",
    "avg_words_per_message": {style_metrics.get('avg_words_per_message', 0.0)},
    "emoji_ratio": {style_metrics.get('emoji_ratio', 0.0)},
    "question_rate": {style_metrics.get('question_rate', 0.0)},
    "exclamation_rate": {style_metrics.get('exclamation_rate', 0.0)}
  }},
  "emotional_profile": {{
    "overall_sentiment": "positive/negative/neutral",
    "tone_summary": "A 1-sentence summary of their general tone."
  }},
  "personal_facts": [
    "Fact 1 (e.g., 'has a spouse')",
    "Fact 2..."
  ],
  "top_topics": [
    "topic1", "topic2"
  ]
}}
"""
    try:
        raw_res = _call_llm(prompt, system_instruction=system_instr, json_mode=True)
        raw_res = re.sub(r"^```json\s*", "", raw_res, flags=re.IGNORECASE)
        raw_res = re.sub(r"\s*```$", "", raw_res, flags=re.IGNORECASE).strip()
        data = json.loads(raw_res)
        
        # Merge signal counts for transparency
        data['signal_counts'] = fallback_persona.get('signal_counts', {})
        return data
    except Exception as e:
        print(f"LLM persona extraction failed: {e}. Using fallback.")
        return fallback_persona


# RAG Answer Synthesis
def generate_answer(query: str, topic_context: str, messages_context: str, 
                    persona: Dict[str, Any], fallback_answer: str) -> str:

    if not is_llm_active():
        return fallback_answer

    prompt = f"""Answer the user's query about a chat conversation.
    
**User Persona Profile:**
{json.dumps(persona, indent=2)}

**Retrieved Topic Context (High Level):**
{topic_context}

**Retrieved Raw Messages Context (Low Level):**
{messages_context}

**User Query:** "{query}"

Instructions:
1. Provide a direct, natural-sounding, and engaging answer.
2. Ground all facts in the retrieved context and persona. Do not invent details.
3. Reference specific messages using "[msg #INDEX]" tags when quoting them.
4. Format the output in clean, readable markdown (use bullets, bold text, etc.).
"""
    try:
        return _call_llm(prompt, system_instruction="Answer queries about a chat history using retrieved context and persona details.").strip()
    except Exception as e:
        print(f"LLM answer synthesis failed: {e}. Using fallback.")
        return fallback_answer
