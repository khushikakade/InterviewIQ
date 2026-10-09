import os
import re
import logging
from app.utils.text_utils import (
    count_words, detect_filler_words, calculate_vocabulary_richness, clean_text
)

logger = logging.getLogger(__name__)

# Cache whisper model instance so it is initialized once and reused across requests
_WHISPER_MODEL = None

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            import whisper
            # Load lightweight base or tiny model for fast local CPU execution
            _WHISPER_MODEL = whisper.load_model("tiny")
        except Exception as e:
            logger.warning(f"Could not load Whisper STT model: {e}. Will rely on client transcript or fallback STT.")
            _WHISPER_MODEL = False
    return _WHISPER_MODEL

class SpeechService:
    def transcribe_audio(self, audio_filepath):
        """Transcribes audio file to text using cached Whisper model or fallback."""
        if not audio_filepath or not os.path.exists(audio_filepath):
            return ""

        model = get_whisper_model()
        if model:
            try:
                res = model.transcribe(audio_filepath)
                return res.get("text", "").strip()
            except Exception as e:
                logger.error(f"Whisper transcription failed for {audio_filepath}: {e}")
        
        return ""

    def analyze_speech(self, transcript, duration_seconds=30.0):
        """
        Analyzes candidate speech metrics:
        - Words Per Minute (WPM)
        - Speaking duration & Pause metrics
        - Filler word count & breakdown
        - Repeated words
        - Speech clarity %
        - Vocabulary richness %
        """
        clean_t = clean_text(transcript)
        word_count = count_words(clean_t)
        
        duration_minutes = max(duration_seconds / 60.0, 0.05)
        wpm = round(word_count / duration_minutes, 1)

        # Detect Filler Words
        filler_count, filler_dict = detect_filler_words(clean_t)

        # Pause detection heuristic based on punctuation & transcript gaps
        pauses = len(re.findall(r'(\.\.\.|,|;|\?|\!)', clean_t))
        if duration_seconds > 10 and pauses == 0 and word_count > 10:
            pauses = int(duration_seconds / 8.0)

        # Repeated adjacent words (e.g. "I think think that")
        words = clean_t.lower().split()
        repeated_words = []
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 2:
                repeated_words.append(words[i])

        # Vocabulary richness
        vocab_richness = calculate_vocabulary_richness(clean_t)

        # Speech clarity % calculation
        # Optimal WPM for professional interview is 120 - 160 WPM.
        # Deduct score for excessive filler words, fast/slow WPM, or excessive repetitions.
        clarity = 100.0
        
        # WPM penalty
        if wpm < 90:
            clarity -= (90 - wpm) * 0.4
        elif wpm > 170:
            clarity -= (wpm - 170) * 0.5
            
        # Filler penalty
        clarity -= (filler_count * 3.5)
        
        # Repetition penalty
        clarity -= (len(repeated_words) * 2.5)

        clarity = round(min(max(clarity, 45.0), 98.0), 1)

        return {
            'transcript': clean_t,
            'word_count': word_count,
            'duration_seconds': round(duration_seconds, 1),
            'wpm': wpm,
            'wpm_status': 'Optimal' if 110 <= wpm <= 160 else ('Slow' if wpm < 110 else 'Fast'),
            'filler_word_count': filler_count,
            'filler_breakdown': filler_dict,
            'pause_count': pauses,
            'repeated_words': list(set(repeated_words)),
            'vocab_richness': vocab_richness,
            'speech_clarity': clarity
        }

speech_service = SpeechService()
