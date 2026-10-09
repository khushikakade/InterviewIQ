from config import Config

class ScoringService:
    def calculate_interview_score(self, answers):
        """
        Calculates the transparent weighted score for an entire interview session across all answered questions.
        Includes Explainable AI (XAI) breakdown explaining WHY the score was given.
        """
        if not answers:
            return {
                'answer_quality': 0.0,
                'communication': 0.0,
                'technical_knowledge': 0.0,
                'speech': 0.0,
                'body_language': 0.0,
                'structure': 0.0,
                'final_weighted_score': 0.0,
                'readiness_score': 0.0,
                'readiness_status': 'NEEDS MORE PRACTICE',
                'positive_factors': [],
                'improvement_areas': [],
                'contribution_weights': Config.WEIGHTS
            }

        # Average metrics across all answered questions
        avg_overall = sum(a.overall_answer_score for a in answers) / len(answers)
        avg_relevance = sum(a.relevance_score for a in answers) / len(answers)
        avg_tech_depth = sum(a.technical_depth_score for a in answers) / len(answers)
        avg_clarity = sum(a.clarity_score for a in answers) / len(answers)
        avg_speech = sum(a.speech_clarity for a in answers) / len(answers)
        avg_vision = sum(a.eye_contact_pct for a in answers) / len(answers)
        avg_structure = sum(a.structure_score for a in answers) / len(answers)

        # Dimension scores out of 100
        answer_quality = round(avg_overall, 1)
        communication = round((avg_clarity * 0.6) + (avg_relevance * 0.4), 1)
        technical_knowledge = round(avg_tech_depth, 1)
        speech_score = round(avg_speech, 1)
        body_language = round(avg_vision, 1)
        structure_score = round(avg_structure, 1)

        # Weighted calculation based on configured weights
        w = Config.WEIGHTS
        final_score = (
            (answer_quality * w['answer_quality']) +
            (communication * w['communication']) +
            (technical_knowledge * w['technical_knowledge']) +
            (speech_score * w['speech']) +
            (body_language * w['body_language']) +
            (structure_score * w['structure'])
        )
        final_score = round(min(max(final_score, 0.0), 100.0), 1)

        # Readiness Score %
        readiness_score = final_score
        if readiness_score >= 75.0:
            readiness_status = "READY FOR INTERVIEW"
        elif readiness_score >= 60.0:
            readiness_status = "MODERATE READINESS - FEW REFINEMENTS NEEDED"
        else:
            readiness_status = "NEEDS MORE PRACTICE"

        # Explainable AI (XAI) factors derivation
        positive_factors = []
        improvement_areas = []

        if body_language >= 78:
            positive_factors.append(f"Consistent eye-contact stability ({body_language}%)")
        else:
            improvement_areas.append("Inconsistent eye contact with camera feed")

        if speech_score >= 80:
            positive_factors.append("Optimal speaking pace and high speech clarity")
        else:
            improvement_areas.append("Speech clarity and pause frequency can be optimized")

        if technical_knowledge >= 75:
            positive_factors.append(f"Strong technical keyword depth ({technical_knowledge}%)")
        else:
            improvement_areas.append("Provide deeper technical concepts and practical examples")

        if communication >= 80:
            positive_factors.append("Clear and relevant answer articulation")
        else:
            improvement_areas.append("Improve direct relevance to the question asked")

        if structure_score >= 80:
            positive_factors.append("Well-structured response organization (STAR framework)")
        else:
            improvement_areas.append("Structure answers using clear Situation-Task-Action-Result format")

        if not positive_factors:
            positive_factors.append("Completed full mock interview session")

        if not improvement_areas:
            improvement_areas.append("Maintain consistency across varied interview roles")

        return {
            'answer_quality': answer_quality,
            'communication': communication,
            'technical_knowledge': technical_knowledge,
            'speech': speech_score,
            'body_language': body_language,
            'structure': structure_score,
            'final_weighted_score': final_score,
            'readiness_score': readiness_score,
            'readiness_status': readiness_status,
            'positive_factors': positive_factors,
            'improvement_areas': improvement_areas,
            'contribution_weights': {
                'Answer Quality': f"{int(w['answer_quality']*100)}%",
                'Communication': f"{int(w['communication']*100)}%",
                'Technical Knowledge': f"{int(w['technical_knowledge']*100)}%",
                'Speech': f"{int(w['speech']*100)}%",
                'Body Language': f"{int(w['body_language']*100)}%",
                'Structure': f"{int(w['structure']*100)}%"
            }
        }

scoring_service = ScoringService()
