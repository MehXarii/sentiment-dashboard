from transformers import pipeline


NEUTRAL_THRESHOLD = 0.65  # confidence below this → NEUTRAL


def load_model():
    """
    Load the DistilBERT sentiment pipeline.
    Returns a HuggingFace text-classification pipeline.
    Cached by Streamlit's @st.cache_resource so it loads only once.
    """
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )


def analyze_sentiment(nlp_pipeline, reviews: list[str]) -> list[dict]:
    """
    Run sentiment analysis on a list of review strings.

    Args:
        nlp_pipeline: HuggingFace pipeline from load_model()
        reviews: list of raw review strings

    Returns:
        list of dicts with keys: review, label, confidence
        label is one of: POSITIVE, NEGATIVE, NEUTRAL
    """
    results = []

    for review in reviews:
        if not review.strip():
            continue

        raw = nlp_pipeline(review)[0]
        label      = raw["label"]       # "POSITIVE" or "NEGATIVE"
        confidence = raw["score"]       # float between 0 and 1

        # If the model isn't confident enough, call it NEUTRAL
        if confidence < NEUTRAL_THRESHOLD:
            label = "NEUTRAL"

        results.append({
            "review":     review,
            "label":      label,
            "confidence": confidence,
        })

    return results
