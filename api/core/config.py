import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "your_secret_api_key_here"  # Default that triggers validation error if load fails
    
    # OpenRouter Configuration
    openrouter_api_key: str = "your_openrouter_api_key_here"
    openrouter_model: str = "xiaomi/mimo-v2-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Vector Database Paths
    # We use ./vectordb because the app runs from the project root
    vectordb_path: str = "./vectordb/voice_embeddings.faiss"
    vectordb_metadata_path: str = "./vectordb/metadata.pkl"
    
    # Model Configuration
    wav2vec_model: str = "facebook/wav2vec2-base-960h"
    embedding_dimension: int = 768
    
    # Supported Languages
    supported_languages: List[str] = [
        "Tamil", "English", "Hindi", "Malayalam", "Telugu"
    ]
    
    # RAG Configuration
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    
    # Acoustic Analysis Thresholds
    silence_threshold_db: float = -40.0
    silence_ratio_ai_threshold: float = 0.01  # 1%
    zcr_variance_ai_threshold: float = 0.001
    spectral_rolloff_cutoff: int = 16000  # Hz
    tempo_variance_ai_threshold: float = 5.0  # BPM
    
    # Fast tempo languages (more lenient on tempo variance)
    fast_tempo_languages: List[str] = ["Tamil", "Malayalam"]

    # --- ROBUST ENV LOADING ---
    model_config = SettingsConfigDict(
        # This calculates the absolute path to: project_root/config/.env
        env_file=os.path.join(os.path.dirname(__file__), "../../config/.env"),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Prevents crashing if .env has extra variables
    )

# Global settings instance
settings = Settings()

def validate_settings():
    """Validate that critical settings are configured."""
    issues = []
    
    # Debug print to help you see what is being loaded
    print(f"DEBUG: Loaded API Key: {settings.api_key[:5]}... (Masked)")
    print(f"DEBUG: Loading .env from: {settings.model_config.get('env_file')}")

    if settings.api_key == "your_secret_api_key_here":
        issues.append("API_KEY is not configured (Still using default)")
    
    if settings.openrouter_api_key == "your_openrouter_api_key_here":
        issues.append("OPENROUTER_API_KEY is not configured (Still using default)")
    
    # Check if vector DB exists
    if not os.path.exists(settings.vectordb_path):
        issues.append(f"Vector database not found at {os.path.abspath(settings.vectordb_path)}")
    
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True