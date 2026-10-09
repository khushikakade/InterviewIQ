import re
from app.utils.text_utils import compute_cosine_similarity, tokenize, extract_keywords, clean_text

class NLPService:
    def analyze_answer_nlp(self, question_text, expected_concepts, transcript, question_type='Technical'):
        """
        Calculates NLP metrics for candidate answer:
        - Relevance %
        - Keyword / Concept coverage %
        - Technical depth %
        - Readability & Structure %
        - Answer completeness %
        - STAR score (if question_type is Behavioral)
        """
        clean_t = clean_text(transcript)
        words = tokenize(clean_t)
        word_count = len(words)

        if word_count < 5 or clean_t == "(no response provided)":
            return {
                'relevance': 0.0,
                'keyword_coverage': 0.0,
                'technical_depth': 0.0,
                'clarity': 0.0,
                'structure': 0.0,
                'completeness': 0.0,
                'matched_concepts': [],
                'missing_concepts': expected_concepts,
                'star_analysis': self.analyze_star_framework(clean_t) if question_type == 'Behavioral' else None
            }

        # 1. Relevance calculation
        relevance_sim = compute_cosine_similarity(question_text, clean_t)
        concept_matches = 0
        matched_concepts = []
        missing_concepts = []

        clean_lower = clean_t.lower()
        for concept in expected_concepts:
            pattern = r'\b' + re.escape(concept.lower()) + r'\b'
            if re.search(pattern, clean_lower):
                concept_matches += 1
                matched_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        concept_cov = (concept_matches / len(expected_concepts)) * 100 if expected_concepts else 70.0
        relevance = round(min(max((relevance_sim * 0.4) + (concept_cov * 0.6), 10.0), 98.0), 1)

        # 2. Technical depth calculation based on domain vocabulary & answer detail
        extracted_tech = extract_keywords(clean_t)
        tech_score = (len(extracted_tech) * 15) + (word_count * 0.3)
        tech_depth = round(min(max(tech_score, 10.0), 96.0), 1)

        # 3. Structure & Clarity
        sentences = [s.strip() for s in re.split(r'[\.\?!]', clean_t) if s.strip()]
        avg_sent_len = word_count / max(len(sentences), 1)
        
        clarity = 90.0
        if avg_sent_len < 6 or avg_sent_len > 30:
            clarity -= 20.0
        clarity = round(min(max(clarity, 20.0), 96.0), 1)

        structure = 85.0 if len(sentences) >= 2 else 50.0

        # 4. Completeness
        completeness = round(min(max((word_count / 60.0) * 100, 15.0), 98.0), 1)

        # 5. STAR Framework analysis for behavioral questions
        star_analysis = self.analyze_star_framework(clean_t) if question_type == 'Behavioral' else None

        return {
            'relevance': relevance,
            'keyword_coverage': round(concept_cov, 1),
            'technical_depth': tech_depth,
            'clarity': clarity,
            'structure': structure,
            'completeness': completeness,
            'matched_concepts': matched_concepts,
            'missing_concepts': missing_concepts,
            'star_analysis': star_analysis
        }

    def analyze_star_framework(self, text):
        """
        Analyzes behavioral answer using STAR framework:
        S = Situation (Context/Background)
        T = Task (Challenge/Objective)
        A = Action (Steps taken/Implementation)
        R = Result (Outcome/Impact/Metrics)
        """
        lower = text.lower()

        # Situation indicators
        situation_words = ['when i was', 'at my previous', 'during', 'project where', 'situation', 'team was working']
        has_situation = any(w in lower for w in situation_words) or len(text) > 40

        # Task indicators
        task_words = ['my task', 'my role', 'responsible for', 'needed to', 'goal was', 'objective was', 'deadline']
        has_task = any(w in lower for w in task_words) or ('had to' in lower)

        # Action indicators
        action_words = ['i implemented', 'i designed', 'i developed', 'i initiated', 'i refactored', 'step i took', 'decided to', 'used python']
        has_action = any(w in lower for w in action_words) or ('i ' in lower)

        # Result indicators
        result_words = ['as a result', 'outcome', 'reduced', 'improved', 'increased', '%', 'percent', 'successfully delivered', 'saved time']
        has_result = any(w in lower for w in result_words) or bool(re.search(r'\d+%', lower))

        # Calculate STAR Score / 100
        score = 0
        if has_situation: score += 25
        if has_task: score += 25
        if has_action: score += 30
        if has_result: score += 20

        # Generate specific feedback
        missing_parts = []
        if not has_situation: missing_parts.append("Situation (context)")
        if not has_task: missing_parts.append("Task (objective)")
        if not has_action: missing_parts.append("Action (your specific steps)")
        if not has_result: missing_parts.append("Result (measurable impact/outcome)")

        if missing_parts:
            feedback = f"Your answer explains the core story, but is missing: {', '.join(missing_parts)}."
        else:
            feedback = "Excellent STAR structure! You provided clear context, task, action, and measurable result."

        return {
            'star_score': score,
            'breakdown': {
                'situation': has_situation,
                'task': has_task,
                'action': has_action,
                'result': has_result
            },
            'feedback': feedback
        }

nlp_service = NLPService()
