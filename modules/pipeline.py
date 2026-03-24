"""
Interview Analysis Pipeline
Orchestrates CV, Audio, NLP, and ML modules into a single end-to-end analysis.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from modules.cv_module import CVAnalyzer, CVAnalysisResult
from modules.audio_module import AudioProcessor, AudioAnalysisResult
from modules.nlp_module import NLPAnalyzer, NLPAnalysisResult
from modules.ml_module import score_interview, InterviewScores

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    cv_result: Optional[CVAnalysisResult] = None
    audio_result: Optional[AudioAnalysisResult] = None
    nlp_result: Optional[NLPAnalysisResult] = None
    scores: Optional[InterviewScores] = None
    processing_time: float = 0.0
    video_path: str = ""
    errors: list = field(default_factory=list)
    success: bool = False


class InterviewAnalysisPipeline:
    """End-to-end interview analysis pipeline."""

    def __init__(
        self,
        whisper_model: str = "base",
        cv_sample_rate: int = 5,
        progress_callback: Optional[Callable] = None,
    ):
        self.cv_analyzer = CVAnalyzer()
        self.audio_processor = AudioProcessor(model_size=whisper_model)
        self.nlp_analyzer = NLPAnalyzer()
        self.cv_sample_rate = cv_sample_rate
        self.progress_callback = progress_callback or (lambda step, msg: None)

    def _update_progress(self, step: str, message: str):
        try:
            self.progress_callback(step, message)
        except Exception:
            pass

    def analyze(self, video_path: str) -> AnalysisReport:
        """
        Run full analysis pipeline on a video file.
        
        Args:
            video_path: Path to interview video file
        
        Returns:
            AnalysisReport with all results
        """
        report = AnalysisReport(video_path=video_path)
        start_time = time.time()

        # Step 1: Computer Vision Analysis
        self._update_progress("cv", "Analyzing facial expressions and body language...")
        try:
            logger.info("Starting CV analysis...")
            report.cv_result = self.cv_analyzer.analyze_video(
                video_path, sample_rate=self.cv_sample_rate
            )
            logger.info("CV analysis complete")
        except Exception as e:
            logger.error(f"CV analysis failed: {e}")
            report.errors.append(f"CV Analysis Error: {e}")
            report.cv_result = CVAnalysisResult()  # empty fallback

        # Step 2: Audio Processing & Transcription
        self._update_progress("audio", "Extracting audio and transcribing speech...")
        try:
            logger.info("Starting audio processing...")
            report.audio_result = self.audio_processor.process_video(video_path)
            logger.info(
                f"Audio processing complete: '{report.audio_result.transcript[:80]}...'"
                if len(report.audio_result.transcript) > 80
                else f"Audio processing complete: '{report.audio_result.transcript}'"
            )
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            report.errors.append(f"Audio Processing Error: {e}")
            report.audio_result = AudioAnalysisResult()

        # Step 3: NLP Analysis
        self._update_progress("nlp", "Analyzing speech content and communication quality...")
        try:
            transcript = getattr(report.audio_result, "transcript", "") or ""
            logger.info(f"Starting NLP analysis on {len(transcript.split())} words...")
            report.nlp_result = self.nlp_analyzer.analyze(transcript)
            logger.info("NLP analysis complete")
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            report.errors.append(f"NLP Analysis Error: {e}")
            report.nlp_result = NLPAnalysisResult()

        # Step 4: ML Scoring
        self._update_progress("ml", "Computing final performance scores...")
        try:
            logger.info("Starting ML scoring...")
            report.scores = score_interview(
                report.cv_result, report.nlp_result, report.audio_result
            )
            logger.info(f"Scoring complete: {report.scores.overall_score}/100")
        except Exception as e:
            logger.error(f"ML scoring failed: {e}")
            report.errors.append(f"Scoring Error: {e}")
            report.scores = InterviewScores()

        report.processing_time = round(time.time() - start_time, 2)
        report.success = len(report.errors) == 0
        logger.info(f"Pipeline complete in {report.processing_time}s")

        return report
