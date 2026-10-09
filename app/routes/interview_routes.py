from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from app import db
from app.models.user import User
from app.models.resume import Resume, JobDescription
from app.models.interview import Interview
from app.models.question import Question
from app.models.answer import Answer
from app.models.score import Score
from app.services.question_service import question_service
from app.services.answer_evaluator import answer_evaluator
from app.services.scoring_service import scoring_service
from app.services.recommendation_service import recommendation_service
from app.services.speech_service import speech_service
from app.utils.file_utils import save_uploaded_file
from config import Config

interview_bp = Blueprint('interview', __name__)

def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        user = User.query.first()
        if not user:
            user = User(username='demo_user', email='demo@interviewiq.com')
            user.set_password('demo123')
            db.session.add(user)
            db.session.commit()
        session['user_id'] = user.id
        return user
    return User.query.get(user_id)

@interview_bp.route('/interview/setup', methods=['GET', 'POST'])
def setup_view():
    user = get_current_user()
    if request.method == 'GET':
        resumes = Resume.query.filter_by(user_id=user.id).order_by(Resume.uploaded_at.desc()).all()
        jds = JobDescription.query.filter_by(user_id=user.id).order_by(JobDescription.created_at.desc()).all()
        return render_template('interview_setup.html', resumes=resumes, jds=jds)

    # Handle POST create interview
    data = request.get_json(silent=True) or request.form or {}
    role = data.get('role', 'Software Developer')
    interview_type = data.get('interview_type', 'Mixed')
    difficulty = data.get('difficulty', 'Medium')
    resume_id = data.get('resume_id')
    jd_id = data.get('jd_id')
    q_count = int(data.get('question_count', 5))

    resume = Resume.query.get(resume_id) if resume_id else Resume.query.filter_by(user_id=user.id).order_by(Resume.uploaded_at.desc()).first()
    skills = resume.parsed_data.get('skills', []) if resume else ['Python', 'SQL', 'Problem Solving']

    # Generate Personalized Questions
    generated_qs = question_service.generate_questions(
        role=role,
        interview_type=interview_type,
        difficulty=difficulty,
        skills=skills,
        count=q_count
    )

    interview = Interview(
        user_id=user.id,
        resume_id=resume.id if resume else None,
        job_description_id=jd_id if jd_id else None,
        role=role,
        interview_type=interview_type,
        difficulty=difficulty,
        status='in_progress'
    )
    db.session.add(interview)
    db.session.flush()

    for q in generated_qs:
        question_obj = Question(
            interview_id=interview.id,
            order_num=q['order_num'],
            question_text=q['question_text'],
            question_type=q['question_type'],
            category=q['category']
        )
        question_obj.expected_concepts = q['expected_concepts']
        db.session.add(question_obj)

    db.session.commit()

    if request.is_json:
        return jsonify({'success': True, 'interview_id': interview.id, 'redirect': url_for('interview.interview_session', interview_id=interview.id)})
    return redirect(url_for('interview.interview_session', interview_id=interview.id))

@interview_bp.route('/interview/<int:interview_id>', methods=['GET'])
def interview_session(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    questions = Question.query.filter_by(interview_id=interview_id).order_by(Question.order_num.asc()).all()
    return render_template('interview.html', interview=interview, questions=questions)

@interview_bp.route('/api/interview/<int:interview_id>/submit_answer', methods=['POST'])
def submit_answer(interview_id):
    interview = Interview.query.get_or_404(interview_id)

    json_data = request.get_json(silent=True) or {}
    question_id = request.form.get('question_id') or json_data.get('question_id')
    question = Question.query.get_or_404(question_id)

    transcript = request.form.get('transcript', '') or json_data.get('transcript', '')
    duration_seconds = float(request.form.get('duration_seconds', 30.0) or json_data.get('duration_seconds', 30.0) or 30.0)

    media_filepath = None
    if 'media' in request.files:
        media_file = request.files['media']
        saved_path, err = save_uploaded_file(media_file, allowed_extensions=Config.ALLOWED_MEDIA_EXTENSIONS)
        if not err:
            media_filepath = saved_path
            # Transcribe audio if no text transcript provided
            if not transcript or len(transcript.strip()) < 5:
                stt_text = speech_service.transcribe_audio(media_filepath)
                if stt_text:
                    transcript = stt_text

    transcript = transcript.strip() if transcript else ""
    if not transcript:
        transcript = "(No response provided)"

    # Comprehensive Evaluation
    eval_res = answer_evaluator.evaluate_answer(
        question=question,
        transcript=transcript,
        duration_seconds=duration_seconds,
        media_path=media_filepath
    )

    # Save or update Answer
    answer = Answer.query.filter_by(question_id=question.id).first()
    if not answer:
        answer = Answer(question_id=question.id)

    answer.transcript = eval_res['transcript']
    answer.media_filepath = media_filepath
    answer.duration_seconds = eval_res['duration_seconds']
    answer.wpm = eval_res['wpm']
    answer.filler_word_count = eval_res['filler_word_count']
    answer.pause_count = eval_res['pause_count']
    answer.speech_clarity = eval_res['speech_clarity']
    answer.relevance_score = eval_res['relevance_score']
    answer.correctness_score = eval_res['correctness_score']
    answer.completeness_score = eval_res['completeness_score']
    answer.technical_depth_score = eval_res['technical_depth_score']
    answer.clarity_score = eval_res['clarity_score']
    answer.structure_score = eval_res['structure_score']
    answer.overall_answer_score = eval_res['overall_answer_score']
    answer.feedback_text = eval_res['feedback_text']
    answer.star_score = eval_res['star_score']
    answer.star_breakdown = eval_res['star_breakdown']
    answer.star_feedback = eval_res['star_feedback']
    answer.eye_contact_pct = eval_res['eye_contact_pct']
    answer.head_stability_score = eval_res['head_stability_score']
    answer.attention_stability = eval_res['attention_stability']
    answer.timeline_events = eval_res['timeline_events']
    answer.submitted_at = datetime.utcnow()

    db.session.add(answer)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Answer evaluated and recorded.',
        'evaluation': answer.to_dict()
    })

@interview_bp.route('/api/interview/<int:interview_id>/complete', methods=['POST'])
def complete_interview(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    questions = Question.query.filter_by(interview_id=interview_id).all()
    answers = [q.answer for q in questions if q.answer is not None]

    # Calculate overall scores & XAI breakdown
    score_res = scoring_service.calculate_interview_score(answers)

    score_obj = Score.query.filter_by(interview_id=interview.id).first()
    if not score_obj:
        score_obj = Score(interview_id=interview.id)

    score_obj.answer_quality_score = score_res['answer_quality']
    score_obj.communication_score = score_res['communication']
    score_obj.technical_knowledge_score = score_res['technical_knowledge']
    score_obj.speech_score = score_res['speech']
    score_obj.body_language_score = score_res['body_language']
    score_obj.structure_score = score_res['structure']
    score_obj.final_weighted_score = score_res['final_weighted_score']
    score_obj.readiness_score = score_res['readiness_score']
    score_obj.positive_factors = score_res['positive_factors']
    score_obj.improvement_areas = score_res['improvement_areas']
    score_obj.contribution_weights = score_res['contribution_weights']
    db.session.add(score_obj)

    # Recommendations & 7-day plan
    recs = recommendation_service.detect_weaknesses_and_strengths(score_res)
    plan = recommendation_service.generate_7_day_plan(recs['weaknesses'])

    interview.overall_score = score_res['final_weighted_score']
    interview.readiness_score = score_res['readiness_score']
    interview.readiness_status = score_res['readiness_status']
    interview.weaknesses = recs['weaknesses']
    interview.strengths = recs['strengths']
    interview.improvement_plan = plan
    interview.status = 'completed'
    interview.completed_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Interview analysis completed.',
        'redirect': url_for('analysis.report_view', interview_id=interview.id)
    })
