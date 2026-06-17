import os
import re
import json
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
import numpy as np


# Word lists 
POSITIVE_WORDS = {
    'love', 'happy', 'great', 'awesome', 'wonderful', 'fantastic', 'good',
    'excellent', 'amazing', 'best', 'beautiful', 'thankful', 'grateful',
    'enjoy', 'fun', 'excited', 'brilliant', 'perfect', 'nice', 'glad',
    'pleased', 'cool', 'super', 'yay', 'woohoo', 'yes', 'win',
}
NEGATIVE_WORDS = {
    'sad', 'hate', 'bad', 'terrible', 'awful', 'worst', 'horrible', 'miss',
    'cry', 'hurt', 'upset', 'angry', 'mad', 'disappointed', 'tired',
    'exhausted', 'sick', 'stressed', 'annoyed', 'frustrated', 'lonely',
    'bored', 'ugly', 'fail', 'no', 'never', 'ugh',
}
FOOD_WORDS = {
    'breakfast', 'lunch', 'dinner', 'coffee', 'tea', 'pizza', 'burger',
    'salad', 'food', 'eat', 'eating', 'restaurant', 'cook', 'cooking',
    'meal', 'hungry', 'snack', 'drink', 'beer', 'wine', 'vegan', 'veggie',
    'cake', 'chocolate', 'sushi', 'ramen', 'rice',
}
SLEEP_WORDS = {'sleep', 'asleep', 'bed', 'nap', 'tired', 'wake', 'woke', 'insomnia'}
WORK_WORDS  = {'work', 'office', 'meeting', 'boss', 'job', 'project', 'deadline',
               'client', 'salary', 'interview', 'presentation', 'team', 'colleague'}
TRAVEL_WORDS = {'travel', 'trip', 'flight', 'hotel', 'airport', 'vacation',
                'holiday', 'visit', 'abroad', 'country', 'city', 'map', 'tour'}


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


def _emoji_count(text: str) -> int:
    """Count Unicode emoji characters (codepoint > 0x1F300)."""
    return sum(1 for ch in text if ord(ch) > 0x1F000)


# Main extractor
def extract_persona(df: pd.DataFrame) -> Dict[str, Any]:

    messages: List[str] = df['message'].fillna('').astype(str).tolist()
    full_text = ' '.join(messages).lower()
    all_tokens = _tokenize(full_text)
    token_freq = Counter(all_tokens)

    # Temporal analysis
    late_night_msgs = 0
    early_morning_msgs = 0
    hour_dist: Counter = Counter()
    try:
        ts = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='coerce')
        for t in ts.dropna():
            h = t.hour
            hour_dist[h] += 1
            if 0 <= h < 5:
                late_night_msgs += 1
            if 5 <= h < 8:
                early_morning_msgs += 1
        peak_hour = hour_dist.most_common(1)[0][0] if hour_dist else None
        active_hours = sorted([h for h, _ in hour_dist.most_common(5)])
    except Exception:
        peak_hour = None
        active_hours = []

    # Habits
    habits: List[str] = []

    if late_night_msgs >= 3:
        habits.append(f'late-night active (sends messages between midnight and 5 AM, ~{late_night_msgs} instances)')
    if early_morning_msgs >= 3:
        habits.append(f'early riser (active 5–8 AM, ~{early_morning_msgs} messages)')

    food_hits = [w for w in FOOD_WORDS if token_freq[w] > 0]
    if food_hits:
        habits.append(f"food-conscious — frequently mentions: {', '.join(sorted(food_hits)[:5])}")

    if token_freq['coffee'] >= 2:
        habits.append('coffee drinker (mentions coffee frequently)')
    if token_freq['tea'] >= 2:
        habits.append('tea drinker')

    sleep_hits = [w for w in SLEEP_WORDS if token_freq[w] > 0]
    if sleep_hits and late_night_msgs < 3:
        habits.append(f"mentions sleep-related topics ({', '.join(sleep_hits)})")

    work_hits = [w for w in WORK_WORDS if token_freq[w] > 1]
    if len(work_hits) >= 3:
        habits.append(f"work-focused — frequent work references: {', '.join(sorted(work_hits)[:5])}")

    travel_hits = [w for w in TRAVEL_WORDS if token_freq[w] > 0]
    if len(travel_hits) >= 2:
        habits.append(f"travel enthusiast — mentions: {', '.join(sorted(travel_hits)[:4])}")

    if peak_hour is not None:
        period = 'morning' if 5 <= peak_hour < 12 else \
                 'afternoon' if 12 <= peak_hour < 17 else \
                 'evening' if 17 <= peak_hour < 21 else 'night'
        habits.append(f'most active in the {period} (peak hour: {peak_hour:02d}:00)')

    # Personal Facts
    facts: List[str] = []

    if re.search(r'\b(married|wife|husband|spouse)\b', full_text):
        facts.append('married / has a spouse')
    if re.search(r'\b(girlfriend|boyfriend|partner|dating)\b', full_text):
        facts.append('in a relationship or mentions dating')
    if re.search(r'\b(kid|kids|child|children|son|daughter|baby)\b', full_text):
        facts.append('has or mentions children')
    if re.search(r'\b(mom|mother|dad|father|parents|family)\b', full_text):
        facts.append('family-oriented — frequently mentions family members')
    if re.search(r'\b(birthday|turned \d+|years old)\b', full_text):
        facts.append('mentions birthday or age milestones')
    if re.search(r'\b(moved|moving|relocat|new city|new apartment|new house)\b', full_text):
        facts.append('mentions moving / relocation')
    if re.search(r'\b(college|university|student|studying|degree|class|professor)\b', full_text):
        facts.append('student or in academic setting')
    if re.search(r'\b(gym|workout|running|fitness|exercise|yoga|weights)\b', full_text):
        facts.append('mentions fitness / exercise')
    if re.search(r'\b(dog|cat|pet|puppy|kitten)\b', full_text):
        facts.append('has or mentions pets')

    # Personality Traits
    total = max(1, len(messages))
    emoji_total = sum(_emoji_count(m) for m in messages)
    laugh_count = sum(1 for m in messages
                      if re.search(r'\b(lol|lmao|haha|hehe|rofl|😂|😆)\b', m.lower()))
    exclaim_count = sum(m.count('!') for m in messages)
    question_count = sum(m.count('?') for m in messages)
    caps_msgs = sum(1 for m in messages if sum(1 for c in m if c.isupper()) > len(m) * 0.4 and len(m) > 3)

    pos_count = sum(token_freq[w] for w in POSITIVE_WORDS)
    neg_count = sum(token_freq[w] for w in NEGATIVE_WORDS)

    traits: List[str] = []
    if laugh_count / total > 0.05:
        traits.append(f'has a great sense of humor (laughing expressions in {laugh_count} messages)')
    if emoji_total / total > 0.5:
        traits.append(f'very expressive with emojis (~{emoji_total} emojis across conversation)')
    elif emoji_total > 0:
        traits.append(f'occasionally uses emojis ({emoji_total} total)')
    if exclaim_count / total > 0.3:
        traits.append('enthusiastic and expressive (high exclamation mark usage)')
    if question_count / total > 0.4:
        traits.append('curious and inquisitive (asks many questions)')
    if caps_msgs / total > 0.1:
        traits.append('emphasizes points with ALL CAPS')
    if pos_count > neg_count * 2:
        traits.append('generally positive and upbeat in tone')
    elif neg_count > pos_count * 2:
        traits.append('tends toward emotional / venting in conversations')
    else:
        traits.append('balanced emotional tone (mix of positive and venting)')
    if re.search(r'\b(sorry|my bad|apologize|apologies)\b', full_text):
        traits.append('apologetic / considerate of others')
    if re.search(r'\b(hm+|hmm+|idk|i guess|maybe|perhaps|not sure)\b', full_text):
        traits.append('sometimes indecisive or reflective')

    # Communication Style
    lengths = [len(m.split()) for m in messages if m.strip()]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0

    # Examples
    emoji_example = next((m for m in messages if _emoji_count(m) > 0), '')[:200]
    humor_example = next((m for m in messages
                          if re.search(r'\b(lol|lmao|haha|😂)\b', m.lower())), '')[:200]
    long_example  = max(messages, key=lambda m: len(m.split()), default='')[:200]

    if avg_len < 5:
        style_label = 'very brief/terse (short bursts)'
    elif avg_len < 12:
        style_label = 'concise (casual texting style)'
    elif avg_len < 25:
        style_label = 'moderate-length messages'
    else:
        style_label = 'verbose (writes long messages)'

    comm_style: Dict[str, Any] = {
        'style_label':           style_label,
        'avg_words_per_message': round(avg_len, 2),
        'median_words':          round(median_len, 2),
        'emoji_ratio':           round(emoji_total / total, 3),
        'question_rate':         round(question_count / total, 3),
        'exclamation_rate':      round(exclaim_count / total, 3),
        'humor_rate':            round(laugh_count / total, 3),
        'examples': {
            'emoji_message':    emoji_example,
            'humor_message':    humor_example,
            'longest_message':  long_example,
        },
    }

    # Emotional profile
    emotional_profile: Dict[str, Any] = {
        'positive_word_count': pos_count,
        'negative_word_count': neg_count,
        'overall_sentiment':   'positive' if pos_count > neg_count else
                               'negative' if neg_count > pos_count else 'neutral',
        'top_positive_words':  [w for w in POSITIVE_WORDS if token_freq[w] > 0][:10],
        'top_negative_words':  [w for w in NEGATIVE_WORDS if token_freq[w] > 0][:10],
    }

    # Top Topics
    stop_words = {
        'i', 'me', 'my', 'you', 'your', 'the', 'a', 'an', 'is', 'it',
        'in', 'on', 'at', 'to', 'and', 'or', 'but', 'so', 'do', 'did',
        'was', 'be', 'are', 'have', 'has', 'just', 'like', 'that', 'this',
        'with', 'for', 'of', 'we', 'he', 'she', 'they', 'its', 'not',
        'no', 'ok', 'okay', 'yeah', 'yes', 'lol', 'haha', 'oh', 'hi',
        'hey', 'um', 'uh', 'im', 've', 'll', 're', 'don', 't', 's',
    }
    top_topics = [w for w, _ in token_freq.most_common(50)
                  if w not in stop_words and len(w) > 3][:15]

    # Signal Counts (transparency)
    signal_counts: Dict[str, int] = {
        'total_messages':    total,
        'emoji_total':       emoji_total,
        'laugh_expressions': laugh_count,
        'exclamations':      exclaim_count,
        'questions':         question_count,
        'late_night_msgs':   late_night_msgs,
        'positive_words':    pos_count,
        'negative_words':    neg_count,
    }

    fallback_persona = {
        'habits':              habits,
        'personal_facts':      facts,
        'personality_traits':  traits,
        'communication_style': comm_style,
        'emotional_profile':   emotional_profile,
        'top_topics':          top_topics,
        'signal_counts':       signal_counts,
    }

    from src.llm import is_llm_active, generate_persona
    if not is_llm_active():
        return fallback_persona

    summary_for_llm = f"Top topics: {', '.join(top_topics)}. Peak active hours include: {[f'{h}:00' for h in active_hours]}"
    return generate_persona(summary_for_llm, facts, comm_style, fallback_persona)


if __name__ == '__main__':
    import argparse
    from src.processor import load_csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    df = load_csv(args.csv)
    persona = extract_persona(df)
    os.makedirs(args.out, exist_ok=True)
    with open(f'{args.out}/persona.json', 'w', encoding='utf8') as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(json.dumps(persona, indent=2))