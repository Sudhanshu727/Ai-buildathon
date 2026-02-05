"""
API Routes for Voice Detection Endpoint.
Handles the main /api/voice-detection endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from ..models.schemas import (
    VoiceDetectionRequest,
    VoiceDetectionResponse,
    ErrorResponse
)
from ..core.security import validate_api_key
from ..core.llm_client import llm_client
from ..utils.audio_processor import process_base64_audio, AudioProcessor
from ..utils.feature_extractor import feature_extractor
from ..utils.rag_engine import get_rag_engine


router = APIRouter()


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Detect AI-Generated vs Human Voice",
    description="Analyze an audio sample and classify it as AI-generated or human voice"
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Main endpoint for voice detection.
    
    Process flow:
    1. Validate and decode base64 audio
    2. Extract acoustic features
    3. Query RAG engine for similar voices
    4. Send data to LLM for final verdict
    """
    temp_wav_path = None
    
    try:
        # Step 1: Process audio
        y, sr, temp_wav_path = process_base64_audio(request.audioBase64)
        
        # Step 2: Extract acoustic features
        acoustic_features = feature_extractor.extract_all_features(
            y, sr, request.language
        )
        
        # Step 3: Get RAG consensus (Vector DB Search)
        rag_engine = get_rag_engine()
        rag_results = rag_engine.calculate_rag_consensus(temp_wav_path)
        
        # Step 4: Get classification from LLM
        classification_result = llm_client.classify_voice(
            rag_data=rag_results,
            acoustic_features=acoustic_features,
            language=request.language
        )
        
        # --- FIX: Ensure classification is valid for Pydantic ---
        raw_class = classification_result.get('classification', 'HUMAN').upper()
        if raw_class not in ['AI_GENERATED', 'HUMAN']:
            raw_class = 'HUMAN'  # Default to HUMAN if LLM returns garbage
            
        # Step 5: Validate and Return
        return VoiceDetectionResponse(
            language=request.language,
            classification=raw_class,
            confidenceScore=float(classification_result.get('confidenceScore', 0.0)),
            explanation=classification_result.get('explanation', 'Analysis completed.')
        )
    
    except ValueError as e:
        # Audio processing or decoding errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Unexpected server errors
        print(f"Server Error: {str(e)}") # Log it
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_wav_path:
            AudioProcessor.cleanup_temp_file(temp_wav_path)