import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure required data directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SAMPLE_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, 'models'), exist_ok=True)

    db.init_app(app)

    with app.app_context():
        # Import models so SQLAlchemy registers them
        from app.models.user import User
        from app.models.resume import Resume, JobDescription
        from app.models.interview import Interview
        from app.models.question import Question
        from app.models.answer import Answer
        from app.models.score import Score
        from app.models.practice import PracticeSession

        db.create_all()

    # Register Blueprints
    from app.routes.auth_routes import auth_bp
    from app.routes.resume_routes import resume_bp
    from app.routes.interview_routes import interview_bp
    from app.routes.analysis_routes import analysis_bp
    from app.routes.dashboard_routes import dashboard_bp
    from app.routes.practice_routes import practice_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(resume_bp)
    app.register_blueprint(interview_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(practice_bp)

    return app
