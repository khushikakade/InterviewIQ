import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from config import Config

MODEL_FILE = os.path.join(Config.BASE_DIR, 'models', 'interview_ml_model.joblib')

class InterviewMLModel:
    def __init__(self):
        self.regressor = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        self.is_trained = False
        
        self.load_or_train()

    def generate_synthetic_training_data(self, n_samples=300):
        """Generates synthetic interview feature matrix for model training & baseline calibration."""
        np.random.seed(42)
        
        # 12 Features: [wpm, fillers, pauses, speech_clarity, relevance, correctness, tech_depth, structure, completeness, eye_contact, head_stab, att_stab]
        wpm = np.random.normal(135, 20, n_samples)
        fillers = np.random.poisson(3, n_samples)
        pauses = np.random.poisson(4, n_samples)
        speech_clarity = np.random.normal(82, 10, n_samples)
        
        relevance = np.random.normal(80, 12, n_samples)
        correctness = np.random.normal(78, 14, n_samples)
        tech_depth = np.random.normal(75, 15, n_samples)
        structure = np.random.normal(77, 12, n_samples)
        completeness = np.random.normal(76, 13, n_samples)
        
        eye_contact = np.random.normal(82, 10, n_samples)
        head_stab = np.random.normal(85, 8, n_samples)
        att_stab = np.random.normal(84, 9, n_samples)

        X = np.column_stack([
            wpm, fillers, pauses, speech_clarity,
            relevance, correctness, tech_depth, structure, completeness,
            eye_contact, head_stab, att_stab
        ])

        # Ground truth score formula with small gaussian noise
        y_score = (
            (relevance * 0.25) +
            (correctness * 0.20) +
            (tech_depth * 0.20) +
            (speech_clarity * 0.15) +
            (eye_contact * 0.10) +
            (structure * 0.10)
        ) + np.random.normal(0, 2, n_samples)

        y_score = np.clip(y_score, 0, 100)
        y_readiness = (y_score >= 75).astype(int)  # 1 = Ready, 0 = Needs Practice

        return X, y_score, y_readiness

    def train(self):
        """Trains the Random Forest model pipeline."""
        X, y_score, y_readiness = self.generate_synthetic_training_data()
        self.regressor.fit(X, y_score)
        self.classifier.fit(X, y_readiness)
        self.is_trained = True

        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump({'regressor': self.regressor, 'classifier': self.classifier}, MODEL_FILE)

    def load_or_train(self):
        """Loads pre-trained model if available, else trains new instance."""
        if os.path.exists(MODEL_FILE):
            try:
                data = joblib.load(MODEL_FILE)
                self.regressor = data['regressor']
                self.classifier = data['classifier']
                self.is_trained = True
                return
            except Exception:
                pass
        
        self.train()

interview_ml_model = InterviewMLModel()
