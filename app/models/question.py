import json
from app import db

class Question(db.Model):
    __tablename__ = 'questions'

    id = db.Column(db.Integer, primary_key=True)
    interview_id = db.Column(db.Integer, db.ForeignKey('interviews.id'), nullable=False)
    order_num = db.Column(db.Integer, nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(50), default='Technical')  # Technical, HR, Behavioral
    category = db.Column(db.String(100), nullable=True)             # e.g., System Design, ML, Python
    expected_concepts_json = db.Column(db.Text, nullable=True)

    answer = db.relationship('Answer', backref='question', uselist=False, cascade='all, delete-orphan')

    @property
    def expected_concepts(self):
        return json.loads(self.expected_concepts_json) if self.expected_concepts_json else []

    @expected_concepts.setter
    def expected_concepts(self, val):
        self.expected_concepts_json = json.dumps(val)

    def to_dict(self):
        return {
            'id': self.id,
            'interview_id': self.interview_id,
            'order_num': self.order_num,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'category': self.category,
            'expected_concepts': self.expected_concepts,
            'has_answer': self.answer is not None
        }
