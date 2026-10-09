import json
from datetime import datetime
from app import db

class Answer(db.Model):
    __tablename__ = 'answers'

    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    
    transcript = db.Column(db.Text, nullable=True)
    media_filepath = db.Column(db.String(512), nullable=True)
    
    duration_seconds = db.Column(db.Float, default=0.0)
    wpm = db.Column(db.Float, default=0.0)
    filler_word_count = db.Column(db.Integer, default=0)
    pause_count = db.Column(db.Integer, default=0)
    speech_clarity = db.Column(db.Float, default=0.0)
    
    # NLP & Evaluation
    relevance_score = db.Column(db.Float, default=0.0)
    correctness_score = db.Column(db.Float, default=0.0)
    completeness_score = db.Column(db.Float, default=0.0)
    technical_depth_score = db.Column(db.Float, default=0.0)
    clarity_score = db.Column(db.Float, default=0.0)
    structure_score = db.Column(db.Float, default=0.0)
    overall_answer_score = db.Column(db.Float, default=0.0)
    
    feedback_text = db.Column(db.Text, nullable=True)
    
    # STAR Analysis (Behavioral)
    star_score = db.Column(db.Float, nullable=True)
    star_breakdown_json = db.Column(db.Text, nullable=True)
    star_feedback = db.Column(db.Text, nullable=True)
    
    # Vision metrics for this answer
    eye_contact_pct = db.Column(db.Float, default=0.0)
    head_stability_score = db.Column(db.Float, default=0.0)
    attention_stability = db.Column(db.Float, default=0.0)

    # Timeline events for replay
    timeline_events_json = db.Column(db.Text, nullable=True)

    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def star_breakdown(self):
        return json.loads(self.star_breakdown_json) if self.star_breakdown_json else {}

    @star_breakdown.setter
    def star_breakdown(self, val):
        self.star_breakdown_json = json.dumps(val)

    @property
    def timeline_events(self):
        return json.loads(self.timeline_events_json) if self.timeline_events_json else []

    @timeline_events.setter
    def timeline_events(self, val):
        self.timeline_events_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'question_id': self.question_id,
            'transcript': self.transcript,
            'media_filepath': self.media_filepath,
            'duration_seconds': round(self.duration_seconds, 1),
            'wpm': round(self.wpm, 1),
            'filler_word_count': self.filler_word_count,
            'pause_count': self.pause_count,
            'speech_clarity': round(self.speech_clarity, 1),
            'relevance_score': round(self.relevance_score, 1),
            'correctness_score': round(self.correctness_score, 1),
            'completeness_score': round(self.completeness_score, 1),
            'technical_depth_score': round(self.technical_depth_score, 1),
            'clarity_score': round(self.clarity_score, 1),
            'structure_score': round(self.structure_score, 1),
            'overall_answer_score': round(self.overall_answer_score, 1),
            'feedback_text': self.feedback_text,
            'star_score': round(self.star_score, 1) if self.star_score is not None else None,
            'star_breakdown': self.star_breakdown,
            'star_feedback': self.star_feedback,
            'eye_contact_pct': round(self.eye_contact_pct, 1),
            'head_stability_score': round(self.head_stability_score, 1),
            'attention_stability': round(self.attention_stability, 1),
            'timeline_events': self.timeline_events,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None
        }
