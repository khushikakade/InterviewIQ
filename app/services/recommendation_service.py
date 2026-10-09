class RecommendationService:
    def detect_weaknesses_and_strengths(self, score_dict):
        """
        Analyzes interview scores to identify top weaknesses, top strengths, and recommended actions.
        """
        dimensions = [
            ('Technical Depth', score_dict.get('technical_knowledge', 70)),
            ('Filler Words & Speech Pacing', score_dict.get('speech', 70)),
            ('Answer Structure (STAR)', score_dict.get('structure', 70)),
            ('Eye Contact & Presentation', score_dict.get('body_language', 70)),
            ('Communication & Relevance', score_dict.get('communication', 70)),
            ('Answer Completeness', score_dict.get('answer_quality', 70))
        ]

        # Sort by score ascending (lowest scores = top weaknesses)
        sorted_dims = sorted(dimensions, key=lambda x: x[1])

        top_weaknesses = [d[0] for d in sorted_dims[:3]]
        top_strengths = [d[0] for d in sorted_dims[-3:]][::-1]

        recommendations = []
        for w in top_weaknesses:
            if 'Technical' in w:
                recommendations.append("Provide concrete code examples, trade-offs, and underlying architecture in technical answers.")
            elif 'Filler' in w:
                recommendations.append("Pause intentionally before speaking instead of using filler words like 'um', 'uh', or 'like'.")
            elif 'Structure' in w:
                recommendations.append("Apply the STAR method (Situation, Task, Action, Result) systematically for behavioral questions.")
            elif 'Eye Contact' in w:
                recommendations.append("Position your camera at eye level and look directly at the webcam lens while articulating key points.")
            elif 'Communication' in w:
                recommendations.append("Answer the core question directly in the first 20 seconds before elaborating on context.")

        return {
            'weaknesses': top_weaknesses,
            'strengths': top_strengths,
            'recommendations': recommendations
        }

    def generate_7_day_plan(self, weaknesses):
        """Generates a structured 7-day personalized improvement plan based on detected weaknesses."""
        weak_str = ", ".join(weaknesses[:2]) if weaknesses else "Technical & Communication skills"

        return {
            'Day 1': 'Practice professional introduction and 60-second elevator pitch.',
            'Day 2': f'Speech & Pace Tuning: Focus on eliminating filler words and targeting 130 WPM.',
            'Day 3': 'Behavioral Mastery: Practice 5 STAR-structured responses for conflict and leadership scenarios.',
            'Day 4': f'Technical Deep-Dive: Review core fundamentals and architecture trade-offs for {weak_str}.',
            'Day 5': 'Full Timed Mock Interview: Complete a 5-question technical & behavioral session.',
            'Day 6': 'Weak-Area Practice: Complete targeted re-evaluation questions on InterviewIQ.',
            'Day 7': 'Final Readiness Assessment: Review progress trend graph and verify >80% readiness score.'
        }

    def generate_practice_questions(self, weaknesses, count=3):
        """Generates targeted practice questions directly linked to the candidate's top weaknesses."""
        questions = []
        
        for w in weaknesses:
            if 'Technical' in w:
                questions.append({
                    'weakness': 'Technical Depth',
                    'question_text': 'Explain the difference between process and thread in operating systems, including memory sharing and context switching.'
                })
                questions.append({
                    'weakness': 'Technical Depth',
                    'question_text': 'How does indexing work in relational databases, and what are the trade-offs of creating too many indexes?'
                })
            elif 'Structure' in w or 'Behavioral' in w:
                questions.append({
                    'weakness': 'Answer Structure (STAR)',
                    'question_text': 'Describe a project where you had to adapt quickly to changing requirements. Detail the Situation, Task, Action, and Result.'
                })
            elif 'Speech' in w or 'Filler' in w:
                questions.append({
                    'weakness': 'Filler Words & Speech Pacing',
                    'question_text': 'Explain how you approach debugging a complex, production-only bug, maintaining a deliberate 130 WPM pace.'
                })

        if not questions:
            questions.append({
                'weakness': 'General Interview Readiness',
                'question_text': 'Walk me through your most impactful software engineering project from architecture to production deployment.'
            })

        return questions[:count]

recommendation_service = RecommendationService()
