import os
import json
import logging
from config import Config

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY or Config.OPENAI_API_KEY
        self.provider = 'gemini' if Config.GEMINI_API_KEY else ('openai' if Config.OPENAI_API_KEY else None)
        
        if self.provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini API: {e}. Falling back to local engine.")
                self.provider = None

    def generate_json(self, prompt, fallback_dict):
        """
        Attempts to call external AI API if configured.
        Returns parsed JSON or falls back to local fallback_dict.
        """
        if not self.provider:
            return fallback_dict

        try:
            if self.provider == 'gemini':
                response = self.model.generate_content(
                    f"{prompt}\nReturn your response strictly as valid JSON with no extra commentary or markdown backticks."
                )
                text = response.text.strip()
                if text.startswith('```'):
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                return json.loads(text.strip())
        except Exception as e:
            logger.error(f"AI API call failed: {e}. Using deterministic local fallback.")
            return fallback_dict

        return fallback_dict

ai_service = AIService()
