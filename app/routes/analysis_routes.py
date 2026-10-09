from flask import Blueprint, request, jsonify, render_template, session
from app.models.user import User
from app.models.interview import Interview
from app.models.question import Question
from app.models.score import Score

analysis_bp = Blueprint('analysis', __name__)

def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        user = User.query.first()
        if not user:
            user = User(username='demo_user', email='demo@interviewiq.com')
            user.set_password('demo123')
            session['user_id'] = user.id
        return user
    return User.query.get(user_id)

@analysis_bp.route('/report/<int:interview_id>', methods=['GET'])
def report_view(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    questions = Question.query.filter_by(interview_id=interview_id).order_by(Question.order_num.asc()).all()
    score = Score.query.filter_by(interview_id=interview_id).first()

    return render_template('report.html', interview=interview, questions=questions, score=score)

@analysis_bp.route('/history', methods=['GET'])
def history_view():
    user = get_current_user()
    interviews = Interview.query.filter_by(user_id=user.id).order_by(Interview.started_at.desc()).all()
    return render_template('history.html', interviews=interviews)

@analysis_bp.route('/api/interview/<int:interview_id>/report', methods=['GET'])
def get_report_json(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    questions = Question.query.filter_by(interview_id=interview_id).order_by(Question.order_num.asc()).all()
    score = Score.query.filter_by(interview_id=interview_id).first()

    q_data = []
    for q in questions:
        q_dict = q.to_dict()
        q_dict['answer'] = q.answer.to_dict() if q.answer else None
        q_data.append(q_dict)

    return jsonify({
        'interview': interview.to_dict(),
        'score': score.to_dict() if score else None,
        'questions': q_data
    })
