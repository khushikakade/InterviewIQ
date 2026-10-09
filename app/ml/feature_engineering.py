import numpy as np

class FeatureEngineer:
    def extract_features_from_session(self, answers):
        """
        Converts session answer evaluations into a structured ML feature vector:
        - Speech features: [avg_wpm, avg_fillers, avg_pauses, avg_speech_clarity]
        - NLP features: [avg_relevance, avg_correctness, avg_tech_depth, avg_structure, avg_completeness]
        - Vision features: [avg_eye_contact, avg_head_stability, avg_attention_stability]
        """
        if not answers:
            return np.zeros((1, 12))

        wpm_list = [a.wpm for a in answers]
        fillers_list = [a.filler_word_count for a in answers]
        pauses_list = [a.pause_count for a in answers]
        speech_clarity_list = [a.speech_clarity for a in answers]

        relevance_list = [a.relevance_score for a in answers]
        correctness_list = [a.correctness_score for a in answers]
        tech_depth_list = [a.technical_depth_score for a in answers]
        structure_list = [a.structure_score for a in answers]
        completeness_list = [a.completeness_score for a in answers]

        eye_contact_list = [a.eye_contact_pct for a in answers]
        head_stab_list = [a.head_stability_score for a in answers]
        att_stab_list = [a.attention_stability for a in answers]

        feature_vector = np.array([
            np.mean(wpm_list),
            np.mean(fillers_list),
            np.mean(pauses_list),
            np.mean(speech_clarity_list),
            np.mean(relevance_list),
            np.mean(correctness_list),
            np.mean(tech_depth_list),
            np.mean(structure_list),
            np.mean(completeness_list),
            np.mean(eye_contact_list),
            np.mean(head_stab_list),
            np.mean(att_stab_list)
        ]).reshape(1, -1)

        return feature_vector

feature_engineer = FeatureEngineer()
