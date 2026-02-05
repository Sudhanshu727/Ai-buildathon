"""
Main FastAPI Application for AI Voice Detection API.
Entry point for the API server.
"""

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse  # <--- IMPORT ADDED
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys

# Relative imports
from .core.config import settings, validate_settings
from .models.schemas import HealthResponse, ErrorResponse
from .routes.detection import router as detection_router
from .utils.rag_engine import initialize_rag_engine, get_rag_engine


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 70)
    print("AI Voice Detection API - Starting Up")
    print("=" * 70)
    
    if not validate_settings():
        print("\nERROR: Configuration validation failed")
        sys.exit(1)
    
    try:
        print("\nInitializing RAG Engine...")
        initialize_rag_engine()
        print("RAG Engine initialized successfully")
    except Exception as e:
        print(f"\nERROR: Failed to initialize RAG engine: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("API Server Ready")
    print(f"Endpoint: http://{settings.api_host}:{settings.api_port}/api/voice-detection")
    print("=" * 70 + "\n")
    
    yield
    
    print("\nShutting down API server...")


app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated voices using RAG and acoustic analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        rag_engine = get_rag_engine()
        return HealthResponse(
            status="healthy",
            message="API is running",
            vectordb_loaded=rag_engine.is_loaded(),
            model_loaded=rag_engine.model is not None
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}",
            vectordb_loaded=False,
            model_loaded=False
        )


@app.get("/")
async def root():
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


app.include_router(detection_router, prefix="/api", tags=["Voice Detection"])


# --- FIXED EXCEPTION HANDLERS ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions properly returning JSONResponse."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions properly returning JSONResponse."""
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )