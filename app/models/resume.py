import json
from datetime import datetime
from app import db

class Resume(db.Model):
    __tablename__ = 'resumes'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    extracted_text = db.Column(db.Text, nullable=True)
    parsed_data_json = db.Column(db.Text, nullable=True)  # JSON string of skills, education, projects, etc.
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    job_descriptions = db.relationship('JobDescription', backref='resume', lazy=True, cascade='all, delete-orphan')

    @property
    def parsed_data(self):
        if self.parsed_data_json:
            try:
                return json.loads(self.parsed_data_json)
            except Exception:
                return {}
        return {}

    @parsed_data.setter
    def parsed_data(self, val):
        self.parsed_data_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'parsed_data': self.parsed_data,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None
        }


class JobDescription(db.Model):
    __tablename__ = 'job_descriptions'

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    extracted_skills_json = db.Column(db.Text, nullable=True)
    match_score = db.Column(db.Float, default=0.0)
    match_details_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def extracted_skills(self):
        return json.loads(self.extracted_skills_json) if self.extracted_skills_json else []

    @extracted_skills.setter
    def extracted_skills(self, val):
        self.extracted_skills_json = json.dumps(val)

    @property
    def match_details(self):
        return json.loads(self.match_details_json) if self.match_details_json else {}

    @match_details.setter
    def match_details(self, val):
        self.match_details_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'match_score': round(self.match_score, 1),
            'extracted_skills': self.extracted_skills,
            'match_details': self.match_details,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
