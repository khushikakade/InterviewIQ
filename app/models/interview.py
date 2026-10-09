import json
from datetime import datetime
from app import db

class Interview(db.Model):
    __tablename__ = 'interviews'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=True)
    job_description_id = db.Column(db.Integer, db.ForeignKey('job_descriptions.id'), nullable=True)

    role = db.Column(db.String(100), nullable=False)
    interview_type = db.Column(db.String(50), nullable=False, default='Technical')  # Technical, HR, Behavioral, Mixed
    difficulty = db.Column(db.String(20), nullable=False, default='Medium')        # Easy, Medium, Hard
    status = db.Column(db.String(20), default='in_progress')                        # in_progress, completed

    overall_score = db.Column(db.Float, nullable=True)
    readiness_score = db.Column(db.Float, nullable=True)
    readiness_status = db.Column(db.String(50), nullable=True)  # e.g., "READY FOR INTERVIEW", "NEEDS MORE PRACTICE"
    
    weaknesses_json = db.Column(db.Text, nullable=True)
    strengths_json = db.Column(db.Text, nullable=True)
    improvement_plan_json = db.Column(db.Text, nullable=True)
    
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Relationships
    questions = db.relationship('Question', backref='interview', lazy=True, cascade='all, delete-orphan')
    scores = db.relationship('Score', backref='interview', lazy=True, cascade='all, delete-orphan')

    @property
    def weaknesses(self):
        return json.loads(self.weaknesses_json) if self.weaknesses_json else []

    @weaknesses.setter
    def weaknesses(self, val):
        self.weaknesses_json = json.dumps(val)

    @property
    def strengths(self):
        return json.loads(self.strengths_json) if self.strengths_json else []

    @strengths.setter
    def strengths(self, val):
        self.strengths_json = json.dumps(val)

    @property
    def improvement_plan(self):
        return json.loads(self.improvement_plan_json) if self.improvement_plan_json else {}

    @improvement_plan.setter
    def improvement_plan(self, val):
        self.improvement_plan_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'role': self.role,
            'interview_type': self.interview_type,
            'difficulty': self.difficulty,
            'status': self.status,
            'overall_score': round(self.overall_score, 1) if self.overall_score is not None else None,
            'readiness_score': round(self.readiness_score, 1) if self.readiness_score is not None else None,
            'readiness_status': self.readiness_status,
            'weaknesses': self.weaknesses,
            'strengths': self.strengths,
            'improvement_plan': self.improvement_plan,
            'question_count': len(self.questions),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }
