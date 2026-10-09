from app.ml.feature_engineering import feature_engineer
from app.ml.model import interview_ml_model

class Predictor:
    def predict_interview_performance(self, answers):
        """
        Uses trained ML model to predict overall interview score & readiness probability.
        """
        if not answers:
            return {'predicted_score': 0.0, 'readiness_prob': 0.0}

        X = feature_engineer.extract_features_from_session(answers)
        
        pred_score = interview_ml_model.regressor.predict(X)[0]
        readiness_prob = interview_ml_model.classifier.predict_proba(X)[0][1] * 100.0

        return {
            'predicted_score': round(float(pred_score), 1),
            'readiness_probability': round(float(readiness_prob), 1)
        }

predictor = Predictor()
