import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    return True, None

def validate_username(username):
    if not username or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters long."
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain alphanumeric characters and underscores."
    return True, None
