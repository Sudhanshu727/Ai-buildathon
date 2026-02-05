"""
LLM Client: Handles communication with OpenRouter API.
"""

import json
import requests
from typing import Dict, Any
# We comment out settings for this test to bypass .env issues
# from .config import settings 

class LLMClient:
    """Client for interacting with OpenRouter LLMs."""
    
    def __init__(self):
        # --- HARDCODED CONFIGURATION FOR TESTING ---
        # We are bypassing the .env file to ensure the key is sent correctly.
        self.api_key = "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"
        self.model = "xiaomi/mimo-v2-flash"
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # DEBUG PRINT: Verify what is loaded
        masked_key = self.api_key[:10] + "..." + self.api_key[-5:] if self.api_key else "None"
        print(f"\nDEBUG: LLM Client Initialized")
        print(f"DEBUG: Key being used: {masked_key}")
        print(f"DEBUG: Model: {self.model}")

    def classify_voice(
        self, 
        rag_data: Dict[str, Any], 
        acoustic_features: Dict[str, float],
        language: str
    ) -> Dict[str, Any]:
        """
        Send data to LLM for final classification.
        """
        
        # 1. System Prompt
        system_prompt = """
        You are a forensic audio analyst. Analyze the acoustic features and database matches to classify the audio.
        Output STRICT JSON:
        {
            "classification": "AI_GENERATED" | "HUMAN",
            "confidenceScore": <float 0.0-1.0>,
            "explanation": "<short reason>"
        }
        """
        
        # 2. User Data
        user_content = f"""
        Language: {language}
        [ACOUSTIC]
        Silence: {acoustic_features.get('silence_ratio', 0):.4f}
        ZCR Var: {acoustic_features.get('zcr_variance', 0):.4f}
        
        [DATABASE]
        Matches: {rag_data.get('ai_count', 0)}/5 AI
        Nearest: {rag_data.get('nearest_label', 'UNKNOWN')} ({rag_data.get('nearest_distance', 0):.2f})
        """

        # 3. Headers
        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}", # Strip removes accidental spaces
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "VoiceDetectionAPI"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "response_format": { "type": "json_object" }
        }

        # 4. Request
        try:
            print(f"DEBUG: Sending request to OpenRouter...")
            # Print the exact header (masked) to be 100% sure
            print(f"DEBUG: Auth Header sent: 'Bearer {self.api_key[:5]}...{self.api_key[-5:]}'")
            
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"LLM Error Body: {response.text}")
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            print("DEBUG: LLM Response received successfully.")
            return json.loads(content)
            
        except Exception as e:
            print(f"LLM Exception: {str(e)}")
            # Fallback response so API doesn't crash
            return {
                "classification": "HUMAN",
                "confidenceScore": 0.0,
                "explanation": f"LLM analysis failed: {str(e)}"
            }

# Global Instance
llm_client = LLMClient()