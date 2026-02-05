"""
Security module for API key authentication.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from .config import settings

# API Key header configuration
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Validate API key from request header.
    
    Args:
        api_key: API key from x-api-key header
    
    Returns:
        Validated API key
    
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide x-api-key header."
        )
    
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
