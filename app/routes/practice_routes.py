from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, session
from app import db
from app.models.user import User
from app.models.interview import Interview
from app.models.practice import PracticeSession
from app.services.recommendation_service import recommendation_service
from app.services.nlp_service import nlp_service
from app.services.speech_service import speech_service

practice_bp = Blueprint('practice', __name__)

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

@practice_bp.route('/practice', methods=['GET'])
def practice_view():
    user = get_current_user()

    # Fetch latest weaknesses
    latest_interview = Interview.query.filter_by(user_id=user.id, status='completed').order_by(Interview.completed_at.desc()).first()
    weaknesses = latest_interview.weaknesses if latest_interview and latest_interview.weaknesses else ['Technical depth', 'Filler words', 'Answer structure']

    practice_qs = recommendation_service.generate_practice_questions(weaknesses, count=3)
    past_sessions = PracticeSession.query.filter_by(user_id=user.id).order_by(PracticeSession.created_at.desc()).all()

    return render_template('practice.html', weaknesses=weaknesses, practice_questions=practice_qs, past_sessions=past_sessions)

@practice_bp.route('/api/practice/submit', methods=['POST'])
def submit_practice_answer():
    user = get_current_user()
    data = request.get_json(silent=True) or request.form or {}

    target_weakness = data.get('target_weakness', 'Technical depth')
    question_text = data.get('question_text', 'Explain Python decorators.')
    user_answer = data.get('user_answer', '').strip()

    if not user_answer or len(user_answer) < 5:
        return jsonify({'success': False, 'message': 'Please provide a valid answer to re-evaluate.'}), 400

    # Speech & NLP evaluation
    speech_res = speech_service.analyze_speech(user_answer)
    nlp_res = nlp_service.analyze_answer_nlp(question_text, ['decorator', 'wrapper', 'functools'], user_answer)

    score = round((nlp_res['relevance'] * 0.4) + (nlp_res['technical_depth'] * 0.3) + (speech_res['speech_clarity'] * 0.3), 1)
    improved = score >= 75.0

    feedback = f"Evaluation Score: {score}/100. "
    if improved:
        feedback += "Great job! You showed noticeable improvement in addressing this weakness."
    else:
        feedback += "Keep practicing! Incorporate more specific technical terms and maintain a clear structure."

    ps = PracticeSession(
        user_id=user.id,
        target_weakness=target_weakness,
        question_text=question_text,
        user_answer=user_answer,
        score=score,
        feedback_text=feedback,
        improved=improved,
        completed_at=datetime.utcnow()
    )
    db.session.add(ps)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Practice answer evaluated.',
        'session': ps.to_dict()
    })
