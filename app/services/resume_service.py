import re
import os
try:
    import pymupdf as fitz
except ImportError:
    import fitz

import spacy
from app.utils.text_utils import clean_text, tokenize, TECHNICAL_KEYWORDS_BANK, compute_cosine_similarity

# Load spaCy English model if available, else fallback to basic tokenizer
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

COMMON_SKILL_PATTERNS = [
    'python', 'java', 'c++', 'c#', 'c', 'javascript', 'typescript', 'html', 'html5', 'css', 'css3',
    'react', 'react.js', 'reactjs', 'vue', 'vue.js', 'angular', 'node', 'node.js', 'nodejs',
    'express', 'express.js', 'flask', 'django', 'fastapi', 'spring', 'spring boot',
    'sql', 'postgresql', 'postgres', 'mysql', 'sqlite', 'mongodb', 'mongo', 'redis', 'cassandra',
    'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s',
    'git', 'github', 'gitlab', 'jenkins', 'ci/cd', 'linux', 'bash', 'shell',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
    'opencv', 'mediapipe', 'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'sklearn',
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'power bi', 'tableau', 'excel',
    'data structures', 'algorithms', 'object-oriented programming', 'oop', 'system design',
    'rest api', 'restful api', 'graphql', 'microservices', 'agile', 'scrum', 'jira',
    'spark', 'pyspark', 'hadoop', 'kafka', 'airflow', 'snowflake', 'databricks',
    'transformers', 'huggingface', 'llm', 'langchain', 'generative ai', 'prompt engineering'
]

class ResumeService:
    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from PDF/DOCX/TXT files safely."""
        return self.extract_text_from_file(pdf_path)

    def extract_text_from_file(self, filepath):
        """Multi-format text extractor for PDF, DOCX, and TXT files."""
        if not filepath or not os.path.exists(filepath):
            return ""

        ext = filepath.rsplit('.', 1)[1].lower() if '.' in filepath else ''

        # DOCX File handling
        if ext == 'docx':
            try:
                import docx
                doc = docx.Document(filepath)
                text = "\n".join([p.text for p in doc.paragraphs if p.text])
                if text.strip():
                    return clean_text(text)
            except Exception:
                pass

        # TXT File handling
        if ext == 'txt':
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return clean_text(f.read())
            except Exception:
                pass

        # PDF File handling via PyMuPDF
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            if text.strip():
                return clean_text(text)
        except Exception:
            pass

        # Fallback UTF-8 plain text read
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return clean_text(f.read())
        except Exception:
            return ""

    def parse_resume(self, text):
        """
        Parses resume text using NLP & rule-based extraction to structure:
        - Candidate Name
        - Education
        - Skills
        - Projects
        - Experience
        - Certifications
        - Technologies
        - Achievements
        """
        clean_t = clean_text(text)
        lower_t = clean_t.lower()

        if not clean_t:
            return {
                'name': 'Candidate',
                'skills': ['Python', 'SQL', 'Problem Solving'],
                'technologies': ['Python', 'SQL'],
                'education': ['Computer Science / Engineering Degree'],
                'projects': ['Technical Project'],
                'experience': ['Entry Level Candidate'],
                'certifications': [],
                'achievements': []
            }

        # Extract Name using spaCy NER or header heuristics
        name = "Candidate"
        if nlp:
            doc = nlp(text[:400])
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) in [2, 3]:
                    name = ent.text.strip()
                    break

        if name == "Candidate":
            raw_lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in raw_lines[:3]:
                words = line.split()
                if 1 <= len(words) <= 3 and not any(k in line.lower() for k in ['resume', 'curriculum', 'cv', 'email', 'phone', 'page', 'github']):
                    name = line
                    break

        # Extract Skills
        found_skills = set()
        for skill in COMMON_SKILL_PATTERNS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, lower_t):
                # Standardize skill label formatting
                skill_label = skill.title()
                if skill.lower() in ['sql', 'html', 'css', 'aws', 'gcp', 'nlp', 'cv', 'llm', 'oop', 'api']:
                    skill_label = skill.upper()
                elif skill.lower() in ['react', 'react.js', 'reactjs']:
                    skill_label = 'React'
                elif skill.lower() in ['node', 'node.js', 'nodejs']:
                    skill_label = 'Node.js'
                elif skill.lower() in ['python']:
                    skill_label = 'Python'
                found_skills.add(skill_label)

        # Dynamic Section Extraction
        sections = {
            'education': [],
            'projects': [],
            'experience': [],
            'certifications': [],
            'achievements': []
        }

        # Match education lines
        edu_matches = re.findall(r'(?:b\.tech|bachelor|master|m\.tech|degree|university|institute|college|gpa|cgpa|b\.e\.|b\.s\.)[^\n]*', lower_t)
        sections['education'] = [m.title() for m in edu_matches[:5]]

        # Match project lines
        proj_matches = re.findall(r'(?:project|developed|built|created|system|application|platform|model)[^\n]*', lower_t)
        sections['projects'] = [p.capitalize() for p in proj_matches[:6]]

        # Match experience lines
        exp_matches = re.findall(r'(?:intern|developer|engineer|role|experience|worked at|company|analyst)[^\n]*', lower_t)
        sections['experience'] = [e.capitalize() for e in exp_matches[:5]]

        # Match certification lines
        cert_matches = re.findall(r'(?:certified|certification|certificate|coursera|udemy|aws certified|nptel)[^\n]*', lower_t)
        sections['certifications'] = [c.title() for c in cert_matches[:4]]

        # Match achievement lines
        achieve_matches = re.findall(r'(?:winner|hackathon|published|award|achievement|rank|selected)[^\n]*', lower_t)
        sections['achievements'] = [a.capitalize() for a in achieve_matches[:4]]

        skills_list = list(found_skills) if found_skills else ['Software Engineering', 'Problem Solving', 'Python']
        word_count = len(clean_t.split())
        
        # Calculate Resume Quality Score (0-100) based on structural completeness
        quality_score = 50
        if len(skills_list) >= 5: quality_score += 15
        elif len(skills_list) >= 3: quality_score += 10
        if sections['education']: quality_score += 15
        if sections['projects']: quality_score += 10
        if sections['experience']: quality_score += 10
        quality_score = min(100, quality_score)

        # Generate Strengths & Recommendations
        strengths = []
        recommendations = []

        if len(skills_list) >= 4:
            strengths.append(f"Identified {len(skills_list)} core technical and domain skills.")
        else:
            recommendations.append("Expand skills section to include more domain-specific technologies and libraries.")

        if sections['education']:
            strengths.append("Clear educational background in relevant engineering/STEM field.")
        else:
            recommendations.append("Explicitly state degree, major, and graduation institution.")

        if sections['projects']:
            strengths.append(f"Demonstrated hands-on expertise through {len(sections['projects'])} project showcase(s).")
        else:
            recommendations.append("Add detailed project highlights with tech stack used.")

        if sections['experience']:
            strengths.append("Relevant industry/academic experience highlighted.")

        if word_count < 150:
            recommendations.append("Resume content is concise. Consider expanding on key accomplishments and responsibilities.")
        elif word_count > 800:
            recommendations.append("Resume text is lengthy. Ensure key skills and achievements stand out clearly.")
        else:
            strengths.append("Optimal document length and readable detail density.")

        if not recommendations:
            recommendations.append("Maintain clear formatting and keep technical skills updated for upcoming roles.")

        return {
            'name': name,
            'word_count': word_count,
            'quality_score': quality_score,
            'skills': skills_list,
            'technologies': [s for s in skills_list if s.upper() in ['PYTHON', 'JAVA', 'SQL', 'REACT', 'FLASK', 'AWS', 'DOCKER', 'MACHINE LEARNING', 'PYTORCH', 'JAVASCRIPT', 'TYPESCRIPT', 'C++', 'C#', 'NODE.JS', 'POSTGRESQL', 'MONGODB']],
            'education': sections['education'] if sections['education'] else ['Degree in Computer Science / Information Technology / AI'],
            'projects': sections['projects'] if sections['projects'] else ['Software / Machine Learning Project'],
            'experience': sections['experience'] if sections['experience'] else ['Engineering Student / Candidate'],
            'certifications': sections['certifications'],
            'achievements': sections['achievements'],
            'strengths': strengths,
            'recommendations': recommendations
        }

    def parse_job_description(self, jd_text):
        """Extracts required skills, preferred skills, technologies from JD text."""
        clean_t = clean_text(jd_text)
        lower_t = clean_t.lower()

        required_skills = set()
        for skill in COMMON_SKILL_PATTERNS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, lower_t):
                skill_label = skill.title()
                if skill.lower() in ['sql', 'html', 'css', 'aws', 'gcp', 'nlp', 'cv', 'llm', 'oop', 'api']:
                    skill_label = skill.upper()
                required_skills.add(skill_label)

        preferred = set()
        if 'preferred' in lower_t or 'nice to have' in lower_t:
            parts = lower_t.split('preferred')
            if len(parts) > 1:
                for skill in COMMON_SKILL_PATTERNS:
                    if skill in parts[1]:
                        preferred.add(skill.title())

        req = list(required_skills - preferred)
        pref = list(preferred)

        return {
            'required_skills': req if req else (list(required_skills) if required_skills else ['Communication', 'Problem Solving']),
            'preferred_skills': pref if pref else ['Teamwork', 'Agile'],
            'technologies': list(required_skills),
            'raw_text': clean_t
        }

    def match_resume_with_jd(self, resume_data, jd_data):
        """
        Calculates a detailed, exact match score between resume and job description.
        Returns overall match score %, skill-level breakdown %, matched skills, missing skills, and skills to improve.
        """
        resume_skills = set([s.lower() for s in resume_data.get('skills', [])])
        jd_skills = set([s.lower() for s in jd_data.get('required_skills', []) + jd_data.get('preferred_skills', [])])

        if not jd_skills:
            sim = compute_cosine_similarity(str(resume_data), jd_data.get('raw_text', ''))
            return {
                'match_score': round(sim, 1),
                'skill_match_breakdown': {},
                'matched_skills': resume_data.get('skills', [])[:5],
                'missing_skills': [],
                'skills_to_improve': []
            }

        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills

        # Calculate exact match score percentage
        match_score = round((len(matched) / len(jd_skills)) * 100.0, 1) if jd_skills else 100.0

        # Generate exact granular breakdown: 100% for matched, 0% for missing
        skill_match_breakdown = {}
        for skill in jd_skills:
            title_skill = skill.title()
            if skill in matched:
                skill_match_breakdown[title_skill] = 100
            else:
                skill_match_breakdown[title_skill] = 0

        return {
            'match_score': match_score,
            'skill_match_breakdown': skill_match_breakdown,
            'matched_skills': [s.title() for s in matched],
            'missing_skills': [s.title() for s in missing],
            'skills_to_improve': [s.title() for s in missing][:5]
        }

resume_service = ResumeService()
