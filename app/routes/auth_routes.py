from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from app import db
from app.models.user import User
from app.utils.validators import validate_email, validate_password, validate_username

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.get_json(silent=True) if request.is_json else request.form
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required.'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401

    session['user_id'] = user.id
    session['username'] = user.username

    if request.is_json:
        return jsonify({'success': True, 'message': 'Login successful.', 'redirect': url_for('dashboard.dashboard_view')})
    return redirect(url_for('dashboard.dashboard_view'))

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) if request.is_json else request.form
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    valid_u, u_err = validate_username(username)
    if not valid_u:
        return jsonify({'success': False, 'message': u_err}), 400

    if not validate_email(email):
        return jsonify({'success': False, 'message': 'Invalid email address.'}), 400

    valid_p, p_err = validate_password(password)
    if not valid_p:
        return jsonify({'success': False, 'message': p_err}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    
    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id
    session['username'] = user.username

    return jsonify({'success': True, 'message': 'Registration successful.', 'redirect': url_for('dashboard.dashboard_view')})

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))
