"""
NLP Analysis Module
Analyzes transcript for grammar, vocabulary, filler words, and communication clarity.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Common filler words in English
FILLER_WORDS = {
    "um", "uh", "like", "you know", "sort of", "kind of", "basically",
    "literally", "actually", "honestly", "right", "so", "well", "anyway",
    "i mean", "you see", "er", "hmm", "ah", "okay so", "and uh", "but uh"
}

# Transition and connector words (positive indicator)
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "consequently", "additionally",
    "moreover", "nevertheless", "specifically", "particularly", "in conclusion",
    "in summary", "for example", "for instance", "as a result", "in contrast",
    "on the other hand", "in addition", "first", "second", "finally"
}


@dataclass
class NLPAnalysisResult:
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    unique_word_ratio: float = 0.0
    filler_word_count: int = 0
    filler_word_rate: float = 0.0
    filler_words_found: dict = field(default_factory=dict)
    transition_word_count: int = 0
    vocabulary_richness: float = 0.0
    readability_score: float = 0.0
    avg_word_length: float = 0.0
    grammar_issues: list = field(default_factory=list)
    grammar_score: float = 0.0
    communication_clarity_score: float = 0.0
    sentence_structure_score: float = 0.0
    top_keywords: list = field(default_factory=list)
    named_entities: list = field(default_factory=list)


class NLPAnalyzer:
    """Analyzes speech transcript for communication quality metrics."""

    def __init__(self):
        self._nlp = None
        self._initialized = False

    def _init_spacy(self):
        """Lazy-load spaCy model."""
        if not self._initialized:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found; downloading en_core_web_sm...")
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                        check=True, capture_output=True
                    )
                    self._nlp = spacy.load("en_core_web_sm")
                self._initialized = True
            except Exception as e:
                logger.error(f"spaCy init failed: {e}")
                self._nlp = None
                self._initialized = True

    def _init_nltk(self):
        """Ensure NLTK data is available."""
        try:
            import nltk
            for resource in ["punkt", "stopwords", "averaged_perceptron_tagger"]:
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)
        except Exception as e:
            logger.warning(f"NLTK init partial: {e}")

    def _count_filler_words(self, text: str) -> dict:
        """Count occurrences of filler words/phrases."""
        text_lower = text.lower()
        counts = {}
        for filler in FILLER_WORDS:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                counts[filler] = len(matches)
        return counts

    def _compute_readability(self, text: str) -> float:
        """Compute Flesch Reading Ease score (0-100, higher = easier to read)."""
        try:
            import textstat
            score = textstat.flesch_reading_ease(text)
            return round(max(0, min(100, score)), 2)
        except Exception:
            # Manual calculation
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = text.split()
            syllables = sum(self._count_syllables(w) for w in words)
            if not sentences or not words:
                return 50.0
            asl = len(words) / len(sentences)
            asw = syllables / len(words)
            fre = 206.835 - (1.015 * asl) - (84.6 * asw)
            return round(max(0, min(100, fre)), 2)

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower().strip(".,!?;:")
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _check_grammar_patterns(self, text: str) -> list:
        """
        Simple rule-based grammar checks.
        Returns a list of issue descriptions.
        """
        issues = []

        # Double spaces
        if "  " in text:
            issues.append("Multiple consecutive spaces detected")

        # Repeated words
        words = text.lower().split()
        for i in range(1, len(words)):
            if words[i] == words[i - 1] and words[i] not in {"the", "a", "very"}:
                issues.append(f"Repeated word: '{words[i]}'")

        # Very long sentences (>50 words)
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            word_count = len(s.split())
            if word_count > 50:
                issues.append(f"Very long sentence ({word_count} words) — may reduce clarity")

        # Incomplete sentences (fewer than 3 words)
        for s in sentences:
            s = s.strip()
            if 1 <= len(s.split()) < 3:
                issues.append(f"Possibly incomplete sentence: '{s}'")

        # Starting sentence with filler
        for s in sentences:
            s = s.strip().lower()
            for filler in ["um ", "uh ", "like "]:
                if s.startswith(filler):
                    issues.append(f"Sentence starts with filler word: '{filler.strip()}'")
                    break

        return issues[:10]  # Cap at 10 issues

    def _get_top_keywords(self, doc, top_n: int = 10) -> list:
        """Extract top non-stop meaningful keywords."""
        if doc is None:
            return []
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 3
        ]
        counter = Counter(tokens)
        return [word for word, _ in counter.most_common(top_n)]

    def _get_named_entities(self, doc) -> list:
        """Extract named entities from transcript."""
        if doc is None:
            return []
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return list(set(entities))[:20]

    def analyze(self, transcript: str) -> NLPAnalysisResult:
        """
        Full NLP analysis of transcript.
        
        Args:
            transcript: Raw speech transcript text
        
        Returns:
            NLPAnalysisResult with all communication metrics
        """
        result = NLPAnalysisResult()

        if not transcript or len(transcript.strip()) < 10:
            logger.warning("Transcript too short for meaningful NLP analysis")
            return result

        self._init_nltk()
        self._init_spacy()

        # Basic text statistics
        words = transcript.split()
        result.word_count = len(words)
        result.avg_word_length = round(
            sum(len(w.strip(".,!?;:")) for w in words) / max(1, len(words)), 2
        )

        # Sentences
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
        result.sentence_count = len(sentences)
        result.avg_sentence_length = round(
            result.word_count / max(1, result.sentence_count), 2
        )

        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(w.lower().strip(".,!?;:") for w in words)
        result.unique_word_ratio = round(len(unique_words) / max(1, result.word_count), 4)
        # Normalized vocabulary richness (0-100)
        # TTR naturally decreases with length; normalize by sqrt
        result.vocabulary_richness = round(
            min(100, (len(unique_words) / (result.word_count ** 0.5 + 1)) * 20), 2
        )

        # Filler words
        result.filler_words_found = self._count_filler_words(transcript)
        result.filler_word_count = sum(result.filler_words_found.values())
        result.filler_word_rate = round(
            result.filler_word_count / max(1, result.word_count) * 100, 2
        )

        # Transition words
        text_lower = transcript.lower()
        for tw in TRANSITION_WORDS:
            if tw in text_lower:
                result.transition_word_count += 1

        # Readability
        result.readability_score = self._compute_readability(transcript)

        # Grammar issues
        result.grammar_issues = self._check_grammar_patterns(transcript)
        grammar_penalty = min(40, len(result.grammar_issues) * 5)
        result.grammar_score = round(max(0, 100 - grammar_penalty), 2)

        # spaCy enrichment
        if self._nlp is not None:
            doc = self._nlp(transcript[:100000])  # Limit to 100k chars
            result.top_keywords = self._get_top_keywords(doc)
            result.named_entities = self._get_named_entities(doc)
        else:
            # Fallback: simple frequency-based keywords
            common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was",
                            "were", "in", "on", "at", "to", "of", "for", "with", "i",
                            "it", "that", "this", "they", "we", "my", "your"}
            keyword_counts = Counter(
                w.lower().strip(".,!?;:")
                for w in words
                if w.lower().strip(".,!?;:") not in common_words and len(w) > 3
            )
            result.top_keywords = [w for w, _ in keyword_counts.most_common(10)]

        # Sentence structure score
        # Ideal: 15-20 words/sentence, >3 transition words, low filler rate
        length_score = 100 - min(50, abs(result.avg_sentence_length - 17) * 4)
        transition_score = min(100, result.transition_word_count * 10)
        result.sentence_structure_score = round(
            (length_score * 0.6 + transition_score * 0.4), 2
        )

        # Communication clarity score (composite)
        clarity_components = {
            "readability": (result.readability_score, 0.25),
            "vocabulary": (result.vocabulary_richness, 0.20),
            "grammar": (result.grammar_score, 0.25),
            "filler_penalty": (max(0, 100 - result.filler_word_rate * 5), 0.20),
            "structure": (result.sentence_structure_score, 0.10),
        }
        result.communication_clarity_score = round(
            sum(score * weight for score, weight in clarity_components.values()), 2
        )
        result.communication_clarity_score = max(0, min(100, result.communication_clarity_score))

        logger.info(
            f"NLP Analysis: {result.word_count} words, {result.sentence_count} sentences, "
            f"clarity={result.communication_clarity_score}, filler_rate={result.filler_word_rate}%"
        )
        return result
