"""
Audio processing utilities for decoding and handling audio files.
"""

import base64
import io
import tempfile
from pathlib import Path
from typing import Tuple
import librosa
import soundfile as sf
from pydub import AudioSegment


class AudioProcessor:
    """Handle audio decoding, conversion, and validation."""
    
    @staticmethod
    @staticmethod
    def decode_base64_to_audio(base64_string: str) -> bytes:
        """
        Decode base64 string to audio bytes.
        
        Args:
            base64_string: Base64-encoded audio data
        
        Returns:
            Decoded audio bytes
        
        Raises:
            ValueError: If base64 decoding fails
        """
        try:
            # Remove any whitespace or newlines
            base64_string = base64_string.strip().replace("\n", "").replace("\r", "")
            
            # Fix: Use b64decode instead of decode
            audio_bytes = base64.b64decode(base64_string)
            
            return audio_bytes
        
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {str(e)}")
    
    @staticmethod
    def validate_mp3(audio_bytes: bytes) -> bool:
        """
        Validate that audio bytes are valid MP3 format.
        
        Args:
            audio_bytes: Audio data in bytes
        
        Returns:
            True if valid MP3, False otherwise
        """
        try:
            # Check for MP3 signature (ID3 tag or MPEG frame sync)
            # MP3 files typically start with "ID3" or have MPEG sync bytes (0xFF 0xFB)
            if audio_bytes[:3] == b'ID3':
                return True
            
            if len(audio_bytes) > 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
                return True
            
            # Try to load with pydub as fallback validation
            audio_io = io.BytesIO(audio_bytes)
            AudioSegment.from_mp3(audio_io)
            return True
        
        except Exception:
            return False
    
    @staticmethod
    def mp3_to_wav(audio_bytes: bytes) -> Tuple[str, Path]:
        """
        Convert MP3 bytes to temporary WAV file for processing.
        
        Args:
            audio_bytes: MP3 audio data in bytes
        
        Returns:
            Tuple of (temp_file_path, Path object)
        
        Raises:
            Exception: If conversion fails
        """
        try:
            # Create temporary file for MP3
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_temp:
                mp3_temp.write(audio_bytes)
                mp3_path = mp3_temp.name
            
            # Convert MP3 to WAV using pydub
            audio = AudioSegment.from_mp3(mp3_path)
            
            # Create temporary WAV file
            wav_path = mp3_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
            
            # Clean up MP3 temp file
            Path(mp3_path).unlink()
            
            return wav_path, Path(wav_path)
        
        except Exception as e:
            raise Exception(f"Failed to convert MP3 to WAV: {str(e)}")
    
    @staticmethod
    def load_audio_for_analysis(wav_path: Path, sr: int = 22050) -> Tuple:
        """
        Load audio file for analysis with librosa.
        
        Args:
            wav_path: Path to WAV file
            sr: Target sample rate (default: 22050 for acoustic analysis)
        
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            # Load audio with librosa
            y, sr_actual = librosa.load(str(wav_path), sr=sr, mono=True)
            return y, sr_actual
        
        except Exception as e:
            raise Exception(f"Failed to load audio with librosa: {str(e)}")
    
    @staticmethod
    def cleanup_temp_file(file_path: Path):
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors


# Utility function for quick audio processing
def process_base64_audio(base64_string: str) -> Tuple:
    """
    Complete pipeline: base64 → audio bytes → WAV → librosa array.
    
    Args:
        base64_string: Base64-encoded MP3 audio
    
    Returns:
        Tuple of (waveform, sample_rate, temp_wav_path)
    
    Raises:
        ValueError: If processing fails
    """
    processor = AudioProcessor()
    
    # Decode base64
    audio_bytes = processor.decode_base64_to_audio(base64_string)
    
    # Validate MP3
    if not processor.validate_mp3(audio_bytes):
        raise ValueError("Invalid MP3 audio format")
    
    # Convert to WAV
    wav_path_str, wav_path = processor.mp3_to_wav(audio_bytes)
    
    # Load for analysis
    y, sr = processor.load_audio_for_analysis(wav_path)
    
    return y, sr, wav_path
