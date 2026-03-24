"""
ML Scoring Module
Combines CV and NLP features to produce final interview scores using a trained model.
"""

import numpy as np
import logging
import os
import joblib
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "interview_scorer.joblib"


@dataclass
class InterviewScores:
    confidence_score: float = 0.0
    communication_score: float = 0.0
    overall_score: float = 0.0
    cv_contribution: float = 0.0
    nlp_contribution: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    performance_tier: str = ""


def _build_feature_vector(cv_result, nlp_result, audio_result) -> np.ndarray:
    """
    Construct a flat feature vector from all module outputs.
    Feature order must remain consistent between training and inference.
    """
    features = [
        # CV features
        getattr(cv_result, "eye_contact_percentage", 0),
        getattr(cv_result, "nervous_behavior_score", 0),
        getattr(cv_result, "emotion_stability_score", 0),
        getattr(cv_result, "head_movement_variance", 0),
        getattr(cv_result, "confidence_score", 0),
        cv_result.emotion_counts.get("happy", 0) if hasattr(cv_result, "emotion_counts") else 0,
        cv_result.emotion_counts.get("neutral", 0) if hasattr(cv_result, "emotion_counts") else 0,
        cv_result.emotion_counts.get("nervous", 0) if hasattr(cv_result, "emotion_counts") else 0,

        # Audio features
        getattr(audio_result, "words_per_minute", 0),
        getattr(audio_result, "silence_ratio", 0),

        # NLP features
        getattr(nlp_result, "communication_clarity_score", 0),
        getattr(nlp_result, "filler_word_rate", 0),
        getattr(nlp_result, "vocabulary_richness", 0),
        getattr(nlp_result, "grammar_score", 0),
        getattr(nlp_result, "sentence_structure_score", 0),
        getattr(nlp_result, "avg_sentence_length", 0),
        getattr(nlp_result, "readability_score", 0),
        getattr(nlp_result, "transition_word_count", 0),
    ]
    return np.array(features, dtype=float)


def _generate_synthetic_training_data(n_samples: int = 500):
    """
    Generate synthetic labeled training data for demonstration.
    In production, replace with real labeled interview data.
    """
    rng = np.random.RandomState(42)

    X = []
    y_confidence = []
    y_communication = []

    for _ in range(n_samples):
        # High-performer profile
        if rng.random() < 0.33:
            eye_contact = rng.uniform(65, 95)
            nervous = rng.uniform(5, 20)
            stability = rng.uniform(70, 95)
            head_var = rng.uniform(0, 200)
            cv_conf = rng.uniform(70, 95)
            happy = rng.randint(10, 40)
            neutral = rng.randint(30, 60)
            nervous_cnt = rng.randint(0, 10)
            wpm = rng.uniform(120, 160)
            silence = rng.uniform(0.05, 0.2)
            clarity = rng.uniform(70, 90)
            filler_rate = rng.uniform(0.5, 3)
            vocab = rng.uniform(65, 90)
            grammar = rng.uniform(75, 95)
            structure = rng.uniform(70, 90)
            avg_len = rng.uniform(14, 20)
            readability = rng.uniform(65, 85)
            transitions = rng.randint(5, 12)
            conf_label = rng.uniform(75, 95)
            comm_label = rng.uniform(75, 92)

        # Average-performer profile
        elif rng.random() < 0.5:
            eye_contact = rng.uniform(40, 70)
            nervous = rng.uniform(20, 40)
            stability = rng.uniform(50, 70)
            head_var = rng.uniform(200, 600)
            cv_conf = rng.uniform(45, 70)
            happy = rng.randint(5, 20)
            neutral = rng.randint(20, 50)
            nervous_cnt = rng.randint(10, 30)
            wpm = rng.uniform(90, 130)
            silence = rng.uniform(0.15, 0.35)
            clarity = rng.uniform(45, 70)
            filler_rate = rng.uniform(3, 8)
            vocab = rng.uniform(40, 65)
            grammar = rng.uniform(55, 75)
            structure = rng.uniform(45, 70)
            avg_len = rng.uniform(10, 18)
            readability = rng.uniform(45, 65)
            transitions = rng.randint(2, 6)
            conf_label = rng.uniform(45, 70)
            comm_label = rng.uniform(45, 68)

        # Low-performer profile
        else:
            eye_contact = rng.uniform(10, 45)
            nervous = rng.uniform(40, 80)
            stability = rng.uniform(20, 50)
            head_var = rng.uniform(500, 1500)
            cv_conf = rng.uniform(10, 45)
            happy = rng.randint(0, 10)
            neutral = rng.randint(10, 30)
            nervous_cnt = rng.randint(30, 80)
            wpm = rng.choice([rng.uniform(50, 90), rng.uniform(170, 220)])
            silence = rng.uniform(0.3, 0.6)
            clarity = rng.uniform(15, 45)
            filler_rate = rng.uniform(8, 20)
            vocab = rng.uniform(15, 40)
            grammar = rng.uniform(30, 55)
            structure = rng.uniform(20, 45)
            avg_len = rng.choice([rng.uniform(3, 9), rng.uniform(25, 40)])
            readability = rng.uniform(20, 45)
            transitions = rng.randint(0, 3)
            conf_label = rng.uniform(10, 45)
            comm_label = rng.uniform(10, 42)

        X.append([
            eye_contact, nervous, stability, head_var, cv_conf,
            happy, neutral, nervous_cnt, wpm, silence,
            clarity, filler_rate, vocab, grammar, structure,
            avg_len, readability, transitions
        ])
        y_confidence.append(conf_label)
        y_communication.append(comm_label)

    return np.array(X), np.array(y_confidence), np.array(y_communication)


def train_and_save_model():
    """Train scoring models and save them to disk."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import warnings

    logger.info("Training interview scoring models...")
    X, y_conf, y_comm = _generate_synthetic_training_data(n_samples=800)

    # Confidence model
    conf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ))
    ])
    conf_pipeline.fit(X, y_conf)

    # Communication model
    comm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        ))
    ])
    comm_pipeline.fit(X, y_comm)

    # Cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conf_cv = cross_val_score(conf_pipeline, X, y_conf, cv=5, scoring="r2")
        comm_cv = cross_val_score(comm_pipeline, X, y_comm, cv=5, scoring="r2")

    logger.info(f"Confidence model R² CV: {conf_cv.mean():.3f} ± {conf_cv.std():.3f}")
    logger.info(f"Communication model R² CV: {comm_cv.mean():.3f} ± {comm_cv.std():.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"confidence": conf_pipeline, "communication": comm_pipeline},
        MODEL_PATH
    )
    logger.info(f"Models saved to {MODEL_PATH}")
    return conf_pipeline, comm_pipeline


def load_or_train_models():
    """Load models from disk or train new ones if not found."""
    if MODEL_PATH.exists():
        try:
            models = joblib.load(MODEL_PATH)
            logger.info("Loaded pre-trained models")
            return models["confidence"], models["communication"]
        except Exception as e:
            logger.warning(f"Failed to load models ({e}); retraining...")

    conf_model, comm_model = train_and_save_model()
    return conf_model, comm_model


def _generate_feedback(scores: InterviewScores, cv_result, nlp_result, audio_result):
    """Generate human-readable strengths, weaknesses, and suggestions."""
    strengths = []
    weaknesses = []
    suggestions = []

    eye_contact = getattr(cv_result, "eye_contact_percentage", 0)
    nervous = getattr(cv_result, "nervous_behavior_score", 0)
    filler_rate = getattr(nlp_result, "filler_word_rate", 0)
    vocab = getattr(nlp_result, "vocabulary_richness", 0)
    grammar = getattr(nlp_result, "grammar_score", 0)
    wpm = getattr(audio_result, "words_per_minute", 0)
    clarity = getattr(nlp_result, "communication_clarity_score", 0)
    silence = getattr(audio_result, "silence_ratio", 0)
    transitions = getattr(nlp_result, "transition_word_count", 0)
    readability = getattr(nlp_result, "readability_score", 0)

    # Strengths
    if eye_contact > 65:
        strengths.append("Strong eye contact — conveys confidence and engagement")
    if filler_rate < 3:
        strengths.append("Very low filler word usage — speech is clean and direct")
    if vocab > 60:
        strengths.append("Rich vocabulary — demonstrates strong command of language")
    if grammar > 80:
        strengths.append("Good grammar and sentence construction")
    if 110 <= wpm <= 155:
        strengths.append("Optimal speaking pace — easy to follow and understand")
    if transitions > 5:
        strengths.append("Effective use of transitional language — structured responses")
    if clarity > 70:
        strengths.append("Clear and articulate communication style")
    if nervous < 15:
        strengths.append("Calm and composed demeanor throughout the interview")

    # Weaknesses
    if eye_contact < 40:
        weaknesses.append("Inconsistent eye contact — may appear disengaged")
    if nervous > 40:
        weaknesses.append("Visible signs of nervousness throughout the interview")
    if filler_rate > 8:
        weaknesses.append(f"High filler word usage ({filler_rate:.1f}%) — impacts fluency")
    if vocab < 35:
        weaknesses.append("Limited vocabulary range — responses lack variety")
    if grammar < 60:
        weaknesses.append("Grammar and sentence structure needs improvement")
    if wpm > 175:
        weaknesses.append("Speaking too fast — may be hard to follow")
    elif wpm < 80 and wpm > 0:
        weaknesses.append("Speaking too slowly — may lose listener engagement")
    if silence > 0.4:
        weaknesses.append("Excessive pauses — suggests hesitation or lack of preparation")
    if clarity < 45:
        weaknesses.append("Communication clarity needs significant improvement")

    # Suggestions
    if eye_contact < 60:
        suggestions.append("Practice maintaining eye contact: focus on the camera lens during video interviews")
    if filler_rate > 5:
        filler_words = getattr(nlp_result, "filler_words_found", {})
        top_fillers = sorted(filler_words.items(), key=lambda x: -x[1])[:3]
        top_names = ", ".join(f"'{w}'" for w, _ in top_fillers)
        suggestions.append(f"Reduce filler words (especially {top_names}) — record yourself and practice silent pauses instead")
    if nervous > 30:
        suggestions.append("Practice breathing exercises and mock interviews to build composure under pressure")
    if vocab < 50:
        suggestions.append("Expand vocabulary by reading industry publications and practicing with varied language")
    if wpm > 170:
        suggestions.append("Slow down — consciously pause between key points to aid comprehension")
    elif wpm < 90 and wpm > 0:
        suggestions.append("Work on increasing energy and speaking pace — practice timed responses")
    if transitions < 3:
        suggestions.append("Use more transitional phrases (e.g., 'Furthermore...', 'As a result...') to structure responses clearly")
    if silence > 0.35:
        suggestions.append("Prepare concise answers in advance — practice the STAR method for behavioral questions")
    if grammar < 65:
        suggestions.append("Review grammar fundamentals and practice speaking in complete, well-structured sentences")
    if readability < 50:
        suggestions.append("Simplify sentence structures — shorter sentences are often clearer and more impactful in verbal communication")

    scores.strengths = strengths[:5]
    scores.weaknesses = weaknesses[:5]
    scores.suggestions = suggestions[:6]


def score_interview(cv_result, nlp_result, audio_result) -> InterviewScores:
    """
    Main scoring function — combines CV and NLP features into final scores.
    
    Args:
        cv_result: CVAnalysisResult from cv_module
        nlp_result: NLPAnalysisResult from nlp_module
        audio_result: AudioAnalysisResult from audio_module
    
    Returns:
        InterviewScores with all metrics and feedback
    """
    scores = InterviewScores()

    try:
        conf_model, comm_model = load_or_train_models()
        feature_vector = _build_feature_vector(cv_result, nlp_result, audio_result)
        X = feature_vector.reshape(1, -1)

        raw_confidence = float(conf_model.predict(X)[0])
        raw_communication = float(comm_model.predict(X)[0])

        scores.confidence_score = round(np.clip(raw_confidence, 0, 100), 1)
        scores.communication_score = round(np.clip(raw_communication, 0, 100), 1)

    except Exception as e:
        logger.error(f"ML scoring failed, using rule-based fallback: {e}")
        # Rule-based fallback
        cv_conf = getattr(cv_result, "confidence_score", 50)
        nlp_clarity = getattr(nlp_result, "communication_clarity_score", 50)
        eye_contact = getattr(cv_result, "eye_contact_percentage", 50)
        filler_penalty = min(30, getattr(nlp_result, "filler_word_rate", 0) * 2)

        scores.confidence_score = round(cv_conf * 0.7 + eye_contact * 0.3, 1)
        scores.communication_score = round(nlp_clarity - filler_penalty, 1)
        scores.confidence_score = max(0, min(100, scores.confidence_score))
        scores.communication_score = max(0, min(100, scores.communication_score))

    # Overall score
    scores.overall_score = round(
        scores.confidence_score * 0.45 + scores.communication_score * 0.55, 1
    )

    # Score breakdown
    scores.score_breakdown = {
        "Eye Contact": round(getattr(cv_result, "eye_contact_percentage", 0), 1),
        "Composure": round(max(0, 100 - getattr(cv_result, "nervous_behavior_score", 0)), 1),
        "Vocabulary": round(getattr(nlp_result, "vocabulary_richness", 0), 1),
        "Grammar": round(getattr(nlp_result, "grammar_score", 0), 1),
        "Fluency": round(max(0, 100 - getattr(nlp_result, "filler_word_rate", 0) * 5), 1),
        "Readability": round(getattr(nlp_result, "readability_score", 0), 1),
        "Pace": _pace_score(getattr(audio_result, "words_per_minute", 130)),
    }

    # Performance tier
    if scores.overall_score >= 80:
        scores.performance_tier = "Excellent"
    elif scores.overall_score >= 65:
        scores.performance_tier = "Good"
    elif scores.overall_score >= 50:
        scores.performance_tier = "Average"
    elif scores.overall_score >= 35:
        scores.performance_tier = "Below Average"
    else:
        scores.performance_tier = "Needs Improvement"

    # Generate feedback
    _generate_feedback(scores, cv_result, nlp_result, audio_result)

    logger.info(
        f"Final scores — Confidence: {scores.confidence_score}, "
        f"Communication: {scores.communication_score}, "
        f"Overall: {scores.overall_score} ({scores.performance_tier})"
    )
    return scores


def _pace_score(wpm: float) -> float:
    """Convert words-per-minute to a 0-100 score (ideal = 120-150 WPM)."""
    if wpm == 0:
        return 0.0
    if 120 <= wpm <= 150:
        return 100.0
    elif 100 <= wpm < 120 or 150 < wpm <= 170:
        return 80.0
    elif 80 <= wpm < 100 or 170 < wpm <= 190:
        return 60.0
    elif 60 <= wpm < 80 or 190 < wpm <= 210:
        return 40.0
    else:
        return 20.0
