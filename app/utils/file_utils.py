import os
import uuid
from werkzeug.utils import secure_filename
from config import Config

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, folder=Config.UPLOAD_FOLDER, allowed_extensions=None):
    if not file or file.filename == '':
        return None, "No file selected"
    
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if allowed_extensions and ext not in allowed_extensions:
        return None, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    clean_name = secure_filename(file.filename)
    if not clean_name or clean_name.startswith('.'):
        clean_name = f"resume.{ext if ext else 'pdf'}"

    unique_prefix = str(uuid.uuid4())[:8]
    saved_filename = f"{unique_prefix}_{clean_name}"
    filepath = os.path.join(folder, saved_filename)
    
    os.makedirs(folder, exist_ok=True)
    file.save(filepath)
    return filepath, None
