"""
Demo data generator — creates realistic mock results for UI testing
without requiring a real video file or model downloads.
"""

from modules.cv_module import CVAnalysisResult
from modules.audio_module import AudioAnalysisResult, AudioSegment
from modules.nlp_module import NLPAnalysisResult
from modules.ml_module import InterviewScores
from modules.pipeline import AnalysisReport


SAMPLE_TRANSCRIPT = """
Thank you for having me today. I'm really excited to be here and to discuss this opportunity 
with your team. Um, I've been working in software engineering for about six years now, and 
I have extensive experience in building scalable backend systems. 

In my previous role at a fintech startup, I led the development of a microservices architecture 
that handled over two million transactions per day. We reduced latency by forty percent and 
improved system reliability significantly. I collaborated closely with product and data teams 
to ensure our solutions met both technical and business requirements.

I'm particularly passionate about clean code and uh, you know, writing maintainable systems. 
I believe that good engineering is as much about communication as it is about technical skill. 
For example, I introduced documentation practices that helped onboard new team members 
much more efficiently.

Looking ahead, I'm eager to bring this experience to a larger organization where I can 
contribute to more complex challenges. I'm especially interested in distributed systems 
and machine learning infrastructure, which aligns well with what your team is working on.

I'm a strong communicator, um, I work well under pressure, and I'm always looking to learn 
from my colleagues. I think collaboration is key to building great products.
"""


def create_demo_report() -> AnalysisReport:
    """Create a realistic mock AnalysisReport for UI demonstrations."""

    cv_result = CVAnalysisResult(
        total_frames=180,
        frames_with_face=168,
        emotion_counts={"neutral": 85, "happy": 42, "nervous": 28, "confident": 13},
        eye_contact_frames=118,
        nervous_frames=28,
        head_movement_variance=312.4,
        emotion_stability_score=50.6,
        eye_contact_percentage=70.2,
        nervous_behavior_score=16.7,
        confidence_score=67.4,
        timestamps_nervous=[12.3, 28.7, 45.1, 67.8, 89.2, 102.5, 134.0, 156.3],
    )

    audio_result = AudioAnalysisResult(
        transcript=SAMPLE_TRANSCRIPT,
        segments=[
            AudioSegment(0.0, 8.5, "Thank you for having me today. I'm really excited to be here."),
            AudioSegment(8.5, 22.0, "I've been working in software engineering for about six years now."),
            AudioSegment(22.0, 45.0, "In my previous role at a fintech startup, I led development of a microservices architecture."),
            AudioSegment(45.0, 68.0, "We reduced latency by forty percent and improved system reliability."),
            AudioSegment(68.0, 95.0, "I'm particularly passionate about clean code and writing maintainable systems."),
            AudioSegment(95.0, 130.0, "Looking ahead, I'm eager to bring this experience to a larger organization."),
            AudioSegment(130.0, 165.0, "I'm a strong communicator, I work well under pressure."),
        ],
        language="en",
        duration=168.4,
        words_per_minute=138.2,
        silence_ratio=0.18,
        audio_extracted=True,
        transcription_success=True,
    )

    nlp_result = NLPAnalysisResult(
        word_count=211,
        sentence_count=14,
        avg_sentence_length=15.1,
        unique_word_ratio=0.612,
        filler_word_count=5,
        filler_word_rate=2.37,
        filler_words_found={"um": 3, "uh": 1, "you know": 1},
        transition_word_count=7,
        vocabulary_richness=62.4,
        readability_score=71.3,
        avg_word_length=4.8,
        grammar_issues=["Very long sentence (32 words) — may reduce clarity"],
        grammar_score=85.0,
        communication_clarity_score=72.8,
        sentence_structure_score=68.5,
        top_keywords=["systems", "engineering", "microservices", "latency", "team",
                      "experience", "collaboration", "architecture", "documentation", "learning"],
        named_entities=[("fintech", "ORG"), ("six years", "DATE")],
    )

    scores = InterviewScores(
        confidence_score=68.4,
        communication_score=73.1,
        overall_score=71.1,
        score_breakdown={
            "Eye Contact": 70.2,
            "Composure": 83.3,
            "Vocabulary": 62.4,
            "Grammar": 85.0,
            "Fluency": 88.2,
            "Readability": 71.3,
            "Pace": 100.0,
        },
        strengths=[
            "Strong eye contact — conveys confidence and engagement",
            "Very low filler word usage — speech is clean and direct",
            "Good grammar and sentence construction",
            "Optimal speaking pace — easy to follow and understand",
            "Effective use of transitional language — structured responses",
        ],
        weaknesses=[
            "Visible signs of nervousness in some segments",
            "Vocabulary range could be broader for senior roles",
        ],
        suggestions=[
            "Practice breathing exercises and mock interviews to build composure under pressure",
            "Expand vocabulary by reading industry publications and practicing with varied language",
            "Use more concrete metrics and outcomes when describing past achievements",
            "Prepare 2-3 concise STAR-method stories for common behavioral questions",
        ],
        performance_tier="Good",
    )

    return AnalysisReport(
        cv_result=cv_result,
        audio_result=audio_result,
        nlp_result=nlp_result,
        scores=scores,
        processing_time=47.3,
        video_path="demo_interview.mp4",
        errors=[],
        success=True,
    )
