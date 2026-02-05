"""
RAG Engine: Vector Database Retrieval for Voice Classification.
Handles embedding generation and similarity search in FAISS.
"""

import pickle
import numpy as np
import torch
import librosa  # <--- Changed: Using librosa instead of torchaudio
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import faiss
from ..core.config import settings


class RAGEngine:
    """Retrieval-Augmented Generation engine for voice classification."""
    
    def __init__(self):
        """Initialize RAG engine with vector database and embedding model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Wav2Vec2 model
        print(f"Loading Wav2Vec2 model: {settings.wav2vec_model}")
        self.processor = Wav2Vec2Processor.from_pretrained(settings.wav2vec_model)
        self.model = Wav2Vec2Model.from_pretrained(settings.wav2vec_model).to(self.device)
        self.model.eval()
        
        # Load FAISS index
        print(f"Loading FAISS index from: {settings.vectordb_path}")
        self.index = faiss.read_index(settings.vectordb_path)
        
        # Load metadata
        print(f"Loading metadata from: {settings.vectordb_metadata_path}")
        with open(settings.vectordb_metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"RAG Engine initialized: {self.index.ntotal} vectors loaded")
    
    def load_audio_for_embedding(self, wav_path: Path, target_sr: int = 16000) -> np.ndarray:
        """
        Load audio file and prepare for Wav2Vec2 embedding using Librosa.
        
        Args:
            wav_path: Path to WAV file
            target_sr: Target sample rate (16kHz for Wav2Vec2)
        
        Returns:
            Audio waveform as numpy array
        """
        # FIX: Use librosa instead of torchaudio to avoid TorchCodec errors
        # librosa.load automatically resamples and converts to mono
        waveform, _ = librosa.load(str(wav_path), sr=target_sr, mono=True)
        return waveform
    
    def generate_embedding(self, wav_path: Path) -> np.ndarray:
        """
        Generate embedding for audio file using Wav2Vec2.
        
        Args:
            wav_path: Path to WAV file
        
        Returns:
            Embedding vector (768-dimensional)
        """
        # Load audio (Returns numpy array)
        waveform = self.load_audio_for_embedding(wav_path)
        
        # Process through Wav2Vec2
        # Note: input_values expects numpy array or tensor
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over time dimension
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding.squeeze()
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search for k most similar embeddings in FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to retrieve (default: settings.rag_top_k)
        
        Returns:
            Tuple of (distances, indices, neighbor_metadata)
        """
        if k is None:
            k = settings.rag_top_k
        
        # Ensure query is 2D array
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query, k)
        
        # Get metadata for neighbors
        neighbor_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return distances[0], indices[0], neighbor_metadata
    
    def calculate_rag_consensus(
        self,
        wav_path: Path
    ) -> Dict[str, any]:
        """
        Calculate RAG consensus for voice classification.
        
        Args:
            wav_path: Path to audio file to classify
        
        Returns:
            Dictionary with RAG consensus results
        """
        # Generate embedding for query audio
        query_embedding = self.generate_embedding(wav_path)
        
        # Search for similar neighbors
        distances, indices, neighbors = self.search_similar(query_embedding)
        
        # Count votes
        ai_count = sum(1 for n in neighbors if n['label'] == 'AI_GENERATED')
        human_count = sum(1 for n in neighbors if n['label'] == 'HUMAN')
        
        # Calculate average distance
        avg_distance = float(np.mean(distances))
        
        # Get nearest neighbor details
        nearest_neighbor = neighbors[0]
        nearest_distance = float(distances[0])
        
        # Prepare results
        results = {
            'ai_count': ai_count,
            'human_count': human_count,
            'total_neighbors': len(neighbors),
            'avg_distance': avg_distance,
            'nearest_label': nearest_neighbor['label'],
            'nearest_distance': nearest_distance,
            'nearest_language': nearest_neighbor['language'],
            'all_neighbors': [
                {
                    'label': n['label'],
                    'language': n['language'],
                    'distance': float(d)
                }
                for n, d in zip(neighbors, distances)
            ]
        }
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if RAG engine is properly loaded."""
        return (
            self.index is not None and
            self.metadata is not None and
            self.model is not None
        )


# Global RAG engine instance (will be initialized on startup)
rag_engine = None


def initialize_rag_engine():
    """Initialize global RAG engine instance."""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


def get_rag_engine() -> RAGEngine:
    """Get global RAG engine instance."""
    if rag_engine is None:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")
    return rag_engine