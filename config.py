import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'interviewiq-super-secret-production-key-2026')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "data", "interviewiq.db")}'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload Configurations
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploads')
    PROCESSED_FOLDER = os.path.join(BASE_DIR, 'data', 'processed')
    SAMPLE_FOLDER = os.path.join(BASE_DIR, 'data', 'sample')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB Max upload size
    
    ALLOWED_RESUME_EXTENSIONS = {'pdf', 'docx', 'txt'}
    ALLOWED_MEDIA_EXTENSIONS = {'webm', 'mp4', 'wav', 'mp3', 'ogg', 'm4a'}
    
    # API Keys (Optional with local fallback)
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

    # Scoring Weights (Total 100%)
    WEIGHTS = {
        'answer_quality': 0.25,
        'communication': 0.20,
        'technical_knowledge': 0.20,
        'speech': 0.15,
        'body_language': 0.10,
        'structure': 0.10
    }
