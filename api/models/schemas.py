"""
Pydantic models for API request and response validation.
"""

from typing import Literal
from pydantic import BaseModel, Field, validator


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint."""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio sample"
    )
    
    audioFormat: Literal["mp3"] = Field(
        ...,
        description="Audio format (must be mp3)"
    )
    
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio data"
    )
    
    @validator("audioBase64")
    def validate_base64(cls, v):
        """Validate that audioBase64 is not empty."""
        if not v or len(v.strip()) == 0:
            raise ValueError("audioBase64 cannot be empty")
        return v


class VoiceDetectionResponse(BaseModel):
    """Response model for successful voice detection."""
    
    status: Literal["success"] = "success"
    
    language: str = Field(
        ...,
        description="Language of the analyzed audio"
    )
    
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result"
    )
    
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    
    explanation: str = Field(
        ...,
        description="Explanation for the classification decision"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    status: Literal["error"] = "error"
    message: str = Field(..., description="Error message")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str
    message: str
    vectordb_loaded: bool
    model_loaded: bool
