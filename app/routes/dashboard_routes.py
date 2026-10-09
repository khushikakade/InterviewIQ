from flask import Blueprint, jsonify, render_template, session
from app.models.user import User
from app.models.interview import Interview
from app.models.resume import Resume, JobDescription
from app.models.score import Score

dashboard_bp = Blueprint('dashboard', __name__)

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

@dashboard_bp.route('/')
@dashboard_bp.route('/index')
def landing_page():
    return render_template('index.html')

@dashboard_bp.route('/dashboard')
def dashboard_view():
    user = get_current_user()

    interviews = Interview.query.filter_by(user_id=user.id, status='completed').order_by(Interview.completed_at.desc()).all()
    latest_resume = Resume.query.filter_by(user_id=user.id).order_by(Resume.uploaded_at.desc()).first()
    latest_jd = JobDescription.query.filter_by(user_id=user.id).order_by(JobDescription.created_at.desc()).first()

    avg_readiness = None
    if interviews:
        scores = [i.readiness_score for i in interviews if i.readiness_score is not None]
        if scores:
            avg_readiness = round(sum(scores) / len(scores), 1)

    latest_score = Score.query.filter_by(interview_id=interviews[0].id).first() if interviews else None

    return render_template(
        'dashboard.html',
        user=user,
        recent_interviews=interviews[:5],
        avg_readiness=avg_readiness,
        latest_score=latest_score,
        resume=latest_resume,
        jd=latest_jd,
        has_interviews=len(interviews) > 0
    )

@dashboard_bp.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    user = get_current_user()

    interviews = Interview.query.filter_by(user_id=user.id, status='completed').order_by(Interview.started_at.asc()).all()

    labels = []
    scores = []
    confidence = []
    communication = []
    filler_words = []

    if not interviews:
        # Benchmark sample trend data for initial view
        labels = ['Session 1', 'Session 2', 'Session 3', 'Session 4']
        scores = [68, 74, 81, 87]
        confidence = [65, 72, 79, 85]
        communication = [62, 70, 78, 86]
        filler_words = [18, 14, 9, 4]
    else:
        for idx, item in enumerate(interviews):
            labels.append(f"Interview {idx + 1}")
            scores.append(item.overall_score or 75.0)
            
            sc = Score.query.filter_by(interview_id=item.id).first()
            if sc:
                confidence.append(sc.body_language_score)
                communication.append(sc.communication_score)
            else:
                confidence.append(75.0)
                communication.append(75.0)

            # Filler words aggregate count
            f_count = 0
            for q in item.questions:
                if q.answer:
                    f_count += q.answer.filler_word_count
            filler_words.append(f_count)

    return jsonify({
        'labels': labels,
        'overall_scores': scores,
        'confidence_trend': confidence,
        'communication_trend': communication,
        'filler_words_trend': filler_words
    })
