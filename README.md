# InterviewIQ — AI Candidate Intelligence & Performance Studio 🚀

[![Design System](https://img.shields.io/badge/UI/UX-Google%20Stitch%20Design%20System-6366F1)](https://stitch.withgoogle.com/)
[![Backend](https://img.shields.io/badge/Backend-Flask%20%7C%20Python%203.9+-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Database](https://img.shields.io/badge/Database-SQLite%20%7C%20SQLAlchemy-003B57?logo=sqlite)](https://www.sqlite.org/)
[![AI Engine](https://img.shields.io/badge/AI%20Core-MediaPipe%20%7C%20NLP%20%7C%20Speech-06B6D4)](https://google.github.io/mediapipe/)

**InterviewIQ** is a next-generation AI candidate evaluation platform redesigned with the **Google Stitch UI/UX Design System**. It provides real-time speech pacing, MediaPipe facial posture cues, NLP technical concept scoring, STAR behavioral framework analysis, and Explainable AI (XAI) multi-dimensional interview readiness tracking.

---

## 🎨 Google Stitch Design System

The application features a modern, dark-mode technical workspace built according to Google Stitch design principles:
- **Dark Slate Workspace Palette**: Deep obsidian background (`#0B0E14`), elevated surface cards (`#181C2A`), and glowing 1px subtle borders (`rgba(99, 102, 241, 0.35)`).
- **Vibrant Accents**: High-contrast indicators (Indigo `#6366F1`, Cyan `#06B6D4`, Emerald `#10B981`, Amber `#F59E0B`, Rose `#F43F5E`).
- **Responsive App Shell**: Left navigation sidebar, dynamic top bar header, radial score progress rings, metric tiles, dropzone upload boxes, and Google Sans typography.

---

## 🌟 Key Features

### 📊 1. Network & Performance Intelligence Dashboard
- **Radial Score Progress Ring**: Visual SVG readiness percentage gauge and status badges (`READY`, `MODERATE`, `NEEDS PRACTICE`).
- **6-Dimension Score Grid**: Technical Depth, Communication, Presentation Cues, Speech Pacing, Answer Structure (STAR), and Answer Quality.
- **Interactive Performance Trends**: Chart.js progression charts tracking overall score, confidence, communication, and filler word frequency over time.

### 📄 2. Resume & Job Description Skill Matcher
- **Resume Upload & Extraction**: Drag-and-drop file upload zone (PDF/DOCX) with PyMuPDF text parsing for candidate name, word count, skills cloud, education, experience, projects, and certifications.
- **Job Description Matcher**: Real-time Job Description skill matching (`/api/jd/analyze`) displaying match percentage, matched skills (emerald badges), missing skills (rose badges), and gap recommendations.

### ⚙️ 3. Personalized Mock Interview Studio
- Customized mock session setup by target role, interview type (Mixed, Technical, Behavioral, HR), difficulty, and question count.
- Linked resume skill integration for personalized question generation.
- Device permission pre-check card for WebCam and microphone readiness.

### 📹 4. Mock Interview Screen & Live Metrics HUD
- Live WebCam preview feed with MediaPipe eye contact tracking cue and real-time audio soundwave visualizer.
- **Live Metric Tiles**: Real-time Eye Contact %, Speech Rate (WPM), Filler Words count, and Head Stability.
- **Speech-to-Text & Fallback**: Automatic speech transcription with manual text editing fallback, question palette pills, and recording controls.

### 📑 5. Detailed Report & Explainable AI (XAI)
- **Transparent Scoring System**: Weighted metric calculation (Answer Quality 25%, Communication 20%, Technical Knowledge 20%, Speech Pacing 15%, Vision Cues 10%, Answer Structure 10%).
- **Explainable AI Cards**: Context-aware positive contributing factors and key improvement areas.
- **STAR Framework Analysis**: Checks behavioral answers for Situation, Task, Action, and Result components.
- **Replay Timeline Cues**: Synchronized event logs capturing posture, pacing, and eye contact cues during responses.
- **7-Day Action Plan**: Personalized day-by-day practice schedule based on detected weaknesses.

### 🎯 6. Targeted Weakness Practice Mode
- Adaptive micro-practice scenarios to re-evaluate detected weaknesses in real-time.
- Instant response re-evaluation (`/api/practice/submit`) with feedback reporting and attempt history.

---

## 🛠️ Technology Stack

- **Frontend**: Google Stitch CSS Tokens, HTML5, Vanilla JavaScript, Chart.js, FontAwesome 6
- **Backend**: Flask (Python 3.9+)
- **Database**: SQLite with SQLAlchemy ORM
- **AI / ML & Analytics Core**:
  - **MediaPipe & OpenCV**: Facial landmark tracking and posture stability
  - **Speech-to-Text**: Web Speech API & audio transcription pipeline
  - **spaCy / NLTK**: NLP keyword extraction, similarity, and technical depth scoring
  - **scikit-learn**: Feature engineering & scoring predictors

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/khushikakade/InterviewIQ.git
cd InterviewIQ
```

### 2. Set Up Virtual Environment

- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access in Browser
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📁 Repository Structure

```text
InterviewIQ/
├── app/
│   ├── ml/             # ML feature engineering & scoring models
│   ├── models/         # SQLAlchemy DB models (User, Resume, Interview, Question, Answer, Score, Practice)
│   ├── routes/         # Flask Blueprints (Auth, Resume, Interview, Analysis, Dashboard, Practice)
│   ├── services/       # AI services (NLP, Speech, Vision, Question, Scoring, Recommendation)
│   ├── static/         # Google Stitch CSS design system & JavaScript controllers
│   ├── templates/      # Jinja2 HTML templates with Google Stitch App Shell layout
│   └── utils/          # File utilities & text validators
├── app.py              # Main application entry point
├── config.py           # Configuration settings
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation
```

---

*Built with Google Stitch Design Principles for candidates who strive for interview mastery.*
