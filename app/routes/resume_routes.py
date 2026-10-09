from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from app import db
from app.models.user import User
from app.models.resume import Resume, JobDescription
from app.services.resume_service import resume_service
from app.utils.file_utils import save_uploaded_file
from config import Config

resume_bp = Blueprint('resume', __name__)

def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        # Default user for local demo if unauthenticated
        user = User.query.first()
        if not user:
            user = User(username='demo_user', email='demo@interviewiq.com')
            user.set_password('demo123')
            db.session.add(user)
            db.session.commit()
        session['user_id'] = user.id
        return user
    return User.query.get(user_id)

@resume_bp.route('/resume', methods=['GET'])
def resume_view():
    user = get_current_user()
    latest_resume = Resume.query.filter_by(user_id=user.id).order_by(Resume.uploaded_at.desc()).first()
    return render_template('resume.html', resume=latest_resume)

@resume_bp.route('/api/resume/upload', methods=['POST'])
def upload_resume():
    user = get_current_user()

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded.'}), 400

    file = request.files['file']
    filepath, err = save_uploaded_file(file, allowed_extensions=Config.ALLOWED_RESUME_EXTENSIONS)
    if err:
        return jsonify({'success': False, 'message': err}), 400

    extracted_text = resume_service.extract_text_from_pdf(filepath)
    parsed_data = resume_service.parse_resume(extracted_text)

    resume = Resume(
        user_id=user.id,
        filename=file.filename,
        filepath=filepath,
        extracted_text=extracted_text
    )
    resume.parsed_data = parsed_data

    db.session.add(resume)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Resume parsed successfully.',
        'resume': resume.to_dict()
    })

@resume_bp.route('/api/jd/analyze', methods=['POST'])
def analyze_jd():
    user = get_current_user()
    data = request.get_json(silent=True) or request.form or {}

    jd_title = data.get('title', 'Software Engineer')
    jd_text = data.get('jd_text', '').strip()
    resume_id = data.get('resume_id')

    if not jd_text:
        return jsonify({'success': False, 'message': 'Job Description text is required.'}), 400

    resume = Resume.query.get(resume_id) if resume_id else Resume.query.filter_by(user_id=user.id).order_by(Resume.uploaded_at.desc()).first()
    resume_data = resume.parsed_data if resume else {'skills': ['Python', 'SQL', 'Problem Solving']}

    jd_parsed = resume_service.parse_job_description(jd_text)
    match_result = resume_service.match_resume_with_jd(resume_data, jd_parsed)

    jd = JobDescription(
        user_id=user.id,
        resume_id=resume.id if resume else None,
        title=jd_title,
        raw_text=jd_text,
        match_score=match_result['match_score']
    )
    jd.extracted_skills = jd_parsed['required_skills']
    jd.match_details = match_result

    db.session.add(jd)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Job Description analyzed.',
        'job_description': jd.to_dict()
    })
