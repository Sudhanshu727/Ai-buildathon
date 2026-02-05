#!/usr/bin/env python3
"""
Vector Database Builder: Create FAISS index from audio dataset
This script processes all MP3 files, generates embeddings using Wav2Vec2,
and builds a FAISS vector database for retrieval.

Usage:
    python build_vectordb.py
"""

import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import librosa  # <--- CHANGED: Used for robust audio loading
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import faiss
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="../config/.env")

# Configuration
CONVERTED_DATASET_PATH = os.getenv("CONVERTED_DATASET_PATH", "E:\\ai_voice_detection_api\\converted_dataset")
VECTORDB_PATH = os.getenv("VECTORDB_PATH", "../vectordb/voice_embeddings.faiss")
VECTORDB_METADATA_PATH = os.getenv("VECTORDB_METADATA_PATH", "../vectordb/metadata.pkl")
WAV2VEC_MODEL = os.getenv("WAV2VEC_MODEL", "facebook/wav2vec2-base-960h")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AudioEmbedder:
    """Generate embeddings for audio files using Wav2Vec2."""
    
    def __init__(self, model_name: str = WAV2VEC_MODEL):
        """Initialize Wav2Vec2 model and processor."""
        print(f"Loading Wav2Vec2 model: {model_name}")
        print(f"Device: {DEVICE}")
        
        # Load processor and model
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load audio file and resample to target sample rate using librosa.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Audio waveform as numpy array
        """
        # FIX: Using librosa instead of torchaudio to avoid Windows TorchCodec errors
        try:
            # sr=16000 ensures resampling to what Wav2Vec2 expects
            waveform, _ = librosa.load(str(audio_path), sr=16000)
            return waveform
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def generate_embedding(self, audio_path: Path) -> np.ndarray:
        """
        Generate embedding for a single audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Embedding vector (768-dimensional)
        """
        try:
            # Load audio
            waveform = self.load_audio(audio_path)
            
            if waveform is None:
                return None

            # Process through Wav2Vec2
            # Librosa returns numpy, so we pass it directly to the processor
            inputs = self.processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over time dimension
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            return embedding.squeeze()
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None


def scan_dataset(base_path: Path) -> List[Dict]:
    """
    Scan converted dataset and collect metadata.
    
    Returns:
        List of metadata dictionaries
    """
    dataset_info = []
    
    for category in ["ai_generated", "human"]:
        category_path = base_path / category
        
        if not category_path.exists():
            continue
        
        for language in ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]:
            language_path = category_path / language
            
            if not language_path.exists():
                continue
            
            # Find all .mp3 files
            for mp3_file in language_path.glob("*.mp3"):
                dataset_info.append({
                    "filepath": str(mp3_file),
                    "filename": mp3_file.name,
                    "category": category,
                    "language": language,
                    "label": "AI_GENERATED" if category == "ai_generated" else "HUMAN"
                })
    
    return dataset_info


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build FAISS index for efficient similarity search.
    
    Args:
        embeddings: Array of embeddings (n_samples, dimension)
        use_gpu: Whether to use GPU for FAISS
    
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    
    # Create index (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings
    index.add(embeddings)
    
    # Move to GPU if requested and available
    if use_gpu and torch.cuda.is_available():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"Warning: Could not move index to GPU: {e}. Keeping on CPU.")
    
    return index


def main():
    """Main process to build vector database."""
    print("=" * 70)
    print("FAISS Vector Database Builder for AI Voice Detection")
    print("=" * 70)
    
    # Validate paths
    dataset_path = Path(CONVERTED_DATASET_PATH)
    if not dataset_path.exists():
        print(f"Error: Converted dataset path does not exist: {dataset_path}")
        print("Please run convert_wav_to_mp3.py first")
        sys.exit(1)
    
    # Create output directory
    vectordb_path = Path(VECTORDB_PATH)
    vectordb_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset Path: {dataset_path}")
    print(f"Output Index: {vectordb_path}")
    print(f"Output Metadata: {VECTORDB_METADATA_PATH}")
    
    # Scan dataset
    print("\n[1/4] Scanning dataset...")
    dataset_info = scan_dataset(dataset_path)
    
    if not dataset_info:
        print("Error: No MP3 files found in converted dataset")
        sys.exit(1)
    
    print(f"Found {len(dataset_info)} audio files")
    print(f"  - AI Generated: {sum(1 for x in dataset_info if x['label'] == 'AI_GENERATED')}")
    print(f"  - Human: {sum(1 for x in dataset_info if x['label'] == 'HUMAN')}")
    
    # Initialize embedder
    print("\n[2/4] Initializing Wav2Vec2 model...")
    embedder = AudioEmbedder()
    
    # Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embeddings_list = []
    valid_metadata = []
    
    for item in tqdm(dataset_info, desc="Processing audio", unit="file"):
        audio_path = Path(item["filepath"])
        embedding = embedder.generate_embedding(audio_path)
        
        if embedding is not None:
            embeddings_list.append(embedding)
            valid_metadata.append(item)
    
    if not embeddings_list:
        print("Error: Failed to generate any embeddings")
        sys.exit(1)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    print(f"\nGenerated {len(embeddings_array)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Build FAISS index
    print("\n[4/4] Building FAISS index...")
    index = build_faiss_index(embeddings_array, use_gpu=False)
    
    # Save index
    print(f"\nSaving FAISS index to {vectordb_path}")
    faiss.write_index(index, str(vectordb_path))
    
    # Save metadata
    print(f"Saving metadata to {VECTORDB_METADATA_PATH}")
    with open(VECTORDB_METADATA_PATH, "wb") as f:
        pickle.dump(valid_metadata, f)
    
    # Summary
    print("\n" + "=" * 70)
    print("VECTOR DATABASE BUILD SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(dataset_info)}")
    print(f"Successful embeddings: {len(embeddings_array)}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    print(f"Index size: {index.ntotal} vectors")
    print(f"\nVector database saved successfully!")
    print("\nYou can now start the API server.")


if __name__ == "__main__":
    main()