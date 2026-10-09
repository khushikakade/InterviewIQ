from datetime import datetime
from app import db

class PracticeSession(db.Model):
    __tablename__ = 'practice_sessions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    target_weakness = db.Column(db.String(100), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    user_answer = db.Column(db.Text, nullable=True)
    
    score = db.Column(db.Float, nullable=True)
    feedback_text = db.Column(db.Text, nullable=True)
    improved = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'target_weakness': self.target_weakness,
            'question_text': self.question_text,
            'user_answer': self.user_answer,
            'score': round(self.score, 1) if self.score is not None else None,
            'feedback_text': self.feedback_text,
            'improved': self.improved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
