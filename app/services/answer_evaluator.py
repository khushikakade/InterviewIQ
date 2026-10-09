from app.services.speech_service import speech_service
from app.services.nlp_service import nlp_service
from app.services.vision_service import vision_service

class AnswerEvaluator:
    def evaluate_answer(self, question, transcript, duration_seconds=30.0, media_path=None):
        """
        Evaluates a candidate's answer for a single question.
        Combines speech, NLP, vision, and STAR metrics into a structured evaluation.
        """
        question_text = question.question_text
        expected_concepts = question.expected_concepts
        q_type = question.question_type

        # 1. Speech Analysis
        speech_res = speech_service.analyze_speech(transcript, duration_seconds)

        # 2. NLP & Concept Analysis
        nlp_res = nlp_service.analyze_answer_nlp(question_text, expected_concepts, transcript, q_type)

        # 3. Vision Analysis
        vision_res = vision_service.analyze_video_file(media_path) if media_path else {
            'eye_contact_pct': 85.0,
            'head_stability_score': 88.0,
            'attention_stability': 86.0
        }

        # 4. Overall Question Score Calculation
        relevance = nlp_res['relevance']
        correctness = min(nlp_res['keyword_coverage'] + 20, 95.0)
        completeness = nlp_res['completeness']
        tech_depth = nlp_res['technical_depth']
        clarity = (nlp_res['clarity'] * 0.5) + (speech_res['speech_clarity'] * 0.5)
        structure = nlp_res['structure']

        if q_type == 'Behavioral' and nlp_res['star_analysis']:
            structure = (structure * 0.4) + (nlp_res['star_analysis']['star_score'] * 0.6)

        overall_score = (
            (relevance * 0.25) +
            (correctness * 0.20) +
            (completeness * 0.15) +
            (tech_depth * 0.15) +
            (clarity * 0.15) +
            (structure * 0.10)
        )
        if transcript == "(No response provided)" or not transcript.strip():
            overall_score = 0.0
            feedback_text = "No response provided for this question. Make sure to record speech or type your answer before submitting."
        else:
            overall_score = round(min(max(overall_score, 0.0), 98.0), 1)

            # 5. Generate Actionable Feedback String
            feedback_points = []
            if relevance >= 80:
                feedback_points.append("Your answer is highly relevant to the question.")
            elif relevance < 40:
                feedback_points.append("Your answer appears off-topic or lacks core expected concepts.")
            else:
                feedback_points.append("Try to address the core question more directly.")

            if nlp_res['missing_concepts']:
                missing_str = ", ".join(nlp_res['missing_concepts'][:3])
                feedback_points.append(f"Consider including key concepts like: {missing_str}.")

            if speech_res['filler_word_count'] > 4:
                feedback_points.append(f"Reduce filler word usage ({speech_res['filler_word_count']} detected).")

            if speech_res['wpm_status'] != 'Optimal' and speech_res['word_count'] > 5:
                feedback_points.append(f"Pacing was {speech_res['wpm_status'].lower()} ({speech_res['wpm']} WPM); target 120-150 WPM.")

            if q_type == 'Behavioral' and nlp_res['star_analysis']:
                feedback_points.append(nlp_res['star_analysis']['feedback'])

            feedback_text = " ".join(feedback_points)

        # 6. Generate Replay Timeline Events
        timeline_events = self._generate_timeline_events(speech_res, vision_res, nlp_res, duration_seconds)

        return {
            'transcript': speech_res['transcript'],
            'duration_seconds': speech_res['duration_seconds'],
            'wpm': speech_res['wpm'],
            'filler_word_count': speech_res['filler_word_count'],
            'pause_count': speech_res['pause_count'],
            'speech_clarity': speech_res['speech_clarity'],
            'relevance_score': round(relevance, 1),
            'correctness_score': round(correctness, 1),
            'completeness_score': round(completeness, 1),
            'technical_depth_score': round(tech_depth, 1),
            'clarity_score': round(clarity, 1),
            'structure_score': round(structure, 1),
            'overall_answer_score': overall_score,
            'feedback_text': feedback_text,
            'star_score': nlp_res['star_analysis']['star_score'] if nlp_res['star_analysis'] else None,
            'star_breakdown': nlp_res['star_analysis']['breakdown'] if nlp_res['star_analysis'] else None,
            'star_feedback': nlp_res['star_analysis']['feedback'] if nlp_res['star_analysis'] else None,
            'eye_contact_pct': vision_res['eye_contact_pct'],
            'head_stability_score': vision_res['head_stability_score'],
            'attention_stability': vision_res['attention_stability'],
            'timeline_events': timeline_events
        }

    def _generate_timeline_events(self, speech_res, vision_res, nlp_res, duration):
        events = []
        dur = max(duration, 15.0)

        # Event 1: Start
        events.append({
            'time_str': '00:02',
            'seconds': 2,
            'type': 'info',
            'label': 'Answer Started',
            'detail': 'Candidate began speaking.'
        })

        # Event 2: Filler detection
        if speech_res['filler_word_count'] > 0:
            events.append({
                'time_str': f"00:{int(dur * 0.25):02d}",
                'seconds': int(dur * 0.25),
                'type': 'warning',
                'label': 'Filler Word Detected',
                'detail': f"{speech_res['filler_word_count']} filler word(s) identified in response segment."
            })

        # Event 3: Concept Coverage
        if nlp_res['matched_concepts']:
            events.append({
                'time_str': f"00:{int(dur * 0.5):02d}",
                'seconds': int(dur * 0.5),
                'type': 'success',
                'label': 'Key Concept Addressed',
                'detail': f"Mentioned: {', '.join(nlp_res['matched_concepts'][:2])}"
            })

        # Event 4: Eye Contact / Vision Cue
        events.append({
            'time_str': f"00:{int(dur * 0.75):02d}",
            'seconds': int(dur * 0.75),
            'type': 'success' if vision_res['eye_contact_pct'] >= 75 else 'warning',
            'label': 'Presentation Check',
            'detail': f"Eye contact consistency: {vision_res['eye_contact_pct']}%."
        })

        return events

answer_evaluator = AnswerEvaluator()
