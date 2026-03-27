from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import re

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])

# ── Simple rule-based VADER-style lexicon (no NLTK download needed) ──────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","outstanding",
    "superb","brilliant","love","happy","best","perfect","awesome","positive",
    "nice","beautiful","enjoyable","pleased","impressive","recommend","helpful",
    "satisfied","delighted","thankful","grateful","thrilled","joy","success",
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","poor","worst","hate","sad","angry",
    "disappointing","disappointing","useless","failure","broken","slow","ugly",
    "frustrating","annoying","waste","regret","unhappy","dissatisfied","dreadful",
    "mediocre","inferior","defective","painful","difficult","problematic","wrong",
}
INTENSIFIERS = {"very","extremely","really","absolutely","totally","incredibly","quite"}
NEGATORS    = {"not","never","no","neither","nor","cannot","can't","won't","don't"}

def _score_text(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    score = 0.0
    i = 0
    while i < len(words):
        w = words[i]
        multiplier = 1.0
        if i > 0 and words[i-1] in INTENSIFIERS:
            multiplier = 1.5
        if i > 0 and words[i-1] in NEGATORS:
            multiplier = -1.0
        if i > 1 and words[i-2] in NEGATORS:
            multiplier = -0.8
        if w in POSITIVE_WORDS:
            score += 1.0 * multiplier
        elif w in NEGATIVE_WORDS:
            score -= 1.0 * abs(multiplier)
        i += 1
    return score

def _classify(score: float, text: str):
    # Emoji / punctuation boosts
    if "!" in text:
        score += 0.2
    if "?" in text:
        score -= 0.1
    if score > 0.3:
        label = "Positive"
        emoji = "😊"
    elif score < -0.3:
        label = "Negative"
        emoji = "😞"
    else:
        label = "Neutral"
        emoji = "😐"
    # Confidence mapped to 0-1
    confidence = min(abs(score) / 3.0, 1.0)
    return label, emoji, round(confidence, 2)


class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResult(BaseModel):
    text: str
    label: str
    emoji: str
    confidence: float
    score: float

@router.post("/analyze", response_model=List[SentimentResult])
def analyze_sentiment(req: SentimentRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided.")
    if len(req.texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per request.")

    results = []
    for text in req.texts:
        if not text.strip():
            continue
        score = _score_text(text)
        label, emoji, confidence = _classify(score, text)
        results.append(SentimentResult(
            text=text[:120],
            label=label,
            emoji=emoji,
            confidence=confidence,
            score=round(score, 3),
        ))
    return results


@router.get("/summary")
def sentiment_summary(texts: str):
    """Quick GET endpoint — pass texts as comma-separated query param."""
    text_list = [t.strip() for t in texts.split("||") if t.strip()]
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for text in text_list:
        score = _score_text(text)
        label, _, _ = _classify(score, text)
        counts[label] += 1
    total = len(text_list) or 1
    return {
        "total": len(text_list),
        "positive": counts["Positive"],
        "neutral":  counts["Neutral"],
        "negative": counts["Negative"],
        "positive_pct": round(counts["Positive"] / total * 100, 1),
        "negative_pct": round(counts["Negative"] / total * 100, 1),
    }
