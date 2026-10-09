import json
from datetime import datetime
from app import db

class Score(db.Model):
    __tablename__ = 'scores'

    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey('interviews.id'), nullable=False)

    answer_quality_score = db.Column(db.Float, default=0.0)
    communication_score = db.Column(db.Float, default=0.0)
    technical_knowledge_score = db.Column(db.Float, default=0.0)
    speech_score = db.Column(db.Float, default=0.0)
    body_language_score = db.Column(db.Float, default=0.0)
    structure_score = db.Column(db.Float, default=0.0)

    final_weighted_score = db.Column(db.Float, default=0.0)
    readiness_score = db.Column(db.Float, default=0.0)

    # Explainable AI Breakdown
    xai_positive_factors_json = db.Column(db.Text, nullable=True)
    xai_improvement_areas_json = db.Column(db.Text, nullable=True)
    xai_contribution_weights_json = db.Column(db.Text, nullable=True)

    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def positive_factors(self):
        return json.loads(self.xai_positive_factors_json) if self.xai_positive_factors_json else []

    @positive_factors.setter
    def positive_factors(self, val):
        self.xai_positive_factors_json = json.dumps(val)

    @property
    def improvement_areas(self):
        return json.loads(self.xai_improvement_areas_json) if self.xai_improvement_areas_json else []

    @improvement_areas.setter
    def improvement_areas(self, val):
        self.xai_improvement_areas_json = json.dumps(val)

    @property
    def contribution_weights(self):
        return json.loads(self.xai_contribution_weights_json) if self.xai_contribution_weights_json else {}

    @contribution_weights.setter
    def contribution_weights(self, val):
        self.xai_contribution_weights_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'interview_id': self.interview_id,
            'answer_quality_score': round(self.answer_quality_score, 1),
            'communication_score': round(self.communication_score, 1),
            'technical_knowledge_score': round(self.technical_knowledge_score, 1),
            'speech_score': round(self.speech_score, 1),
            'body_language_score': round(self.body_language_score, 1),
            'structure_score': round(self.structure_score, 1),
            'final_weighted_score': round(self.final_weighted_score, 1),
            'readiness_score': round(self.readiness_score, 1),
            'positive_factors': self.positive_factors,
            'improvement_areas': self.improvement_areas,
            'contribution_weights': self.contribution_weights,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }
