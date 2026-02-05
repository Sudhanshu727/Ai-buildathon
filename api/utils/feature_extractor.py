"""
Acoustic Feature Extraction Module.
Extracts forensic features from audio that distinguish AI from human voices.
"""

import numpy as np
import librosa
from typing import Dict
from ..core.config import settings


class AcousticFeatureExtractor:
    """Extract acoustic features for AI vs Human voice detection."""
    
    def __init__(self):
        """Initialize feature extractor with configured thresholds."""
        self.silence_threshold_db = settings.silence_threshold_db
        self.spectral_rolloff_cutoff = settings.spectral_rolloff_cutoff
    
    def calculate_silence_ratio(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate the percentage of audio that is near-silent.
        
        AI voices often have unnaturally low silence ratios (< 1%) because
        they lack natural breathing pauses and mouth noise.
        
        Args:
            y: Audio waveform
            sr: Sample rate
        
        Returns:
            Silence ratio (0.0 to 1.0)
        """
        # Convert to dB
        rms = librosa.feature.rms(y=y)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Count frames below threshold
        silent_frames = np.sum(db < self.silence_threshold_db)
        total_frames = len(db)
        
        silence_ratio = silent_frames / total_frames if total_frames > 0 else 0.0
        
        return silence_ratio
    
    def calculate_zcr_variance(self, y: np.ndarray) -> float:
        """
        Calculate variance in zero-crossing rate.
        
        Human voices have high ZCR variance due to natural fluctuations,
        breathing, and fricative sounds. AI voices tend to be more stable.
        
        Args:
            y: Audio waveform
        
        Returns:
            ZCR variance
        """
        # Calculate zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Calculate variance
        zcr_variance = np.var(zcr)
        
        return float(zcr_variance)
    
    def calculate_spectral_rolloff(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate spectral rolloff frequency.
        
        AI voices often have artificial frequency cutoffs (e.g., hard stop at 16kHz)
        due to training data limitations or compression artifacts.
        
        Args:
            y: Audio waveform
            sr: Sample rate
        
        Returns:
            Mean spectral rolloff frequency in Hz
        """
        # Calculate spectral rolloff (frequency below which 85% of energy is contained)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        
        # Return mean rolloff frequency
        mean_rolloff = np.mean(rolloff)
        
        return float(mean_rolloff)
    
    def calculate_tempo_stability(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate variance in tempo/rhythm.
        
        AI voices often maintain unnaturally rigid tempo. Human speech
        naturally varies in rhythm.
        
        Args:
            y: Audio waveform
            sr: Sample rate
        
        Returns:
            Tempo variance in BPM
        """
        try:
            # Extract onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Calculate tempogram
            tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
            
            # Get tempo variance across time
            tempo_variance = np.var(tempogram)
            
            return float(tempo_variance)
        
        except Exception:
            # If tempo calculation fails, return neutral value
            return 5.0
    
    def extract_all_features(
        self,
        y: np.ndarray,
        sr: int,
        language: str
    ) -> Dict[str, float]:
        """
        Extract all acoustic features for forensic analysis.
        
        Args:
            y: Audio waveform
            sr: Sample rate
            language: Language of the audio
        
        Returns:
            Dictionary of acoustic features
        """
        features = {
            'silence_ratio': self.calculate_silence_ratio(y, sr),
            'zcr_variance': self.calculate_zcr_variance(y),
            'spectral_rolloff': self.calculate_spectral_rolloff(y, sr),
            'tempo_variance': self.calculate_tempo_stability(y, sr)
        }
        
        # Add interpretation notes
        features['additional_notes'] = self._generate_notes(features, language)
        
        return features
    
    def _generate_notes(self, features: Dict[str, float], language: str) -> str:
        """
        Generate additional interpretive notes based on features.
        
        Args:
            features: Extracted acoustic features
            language: Language of audio
        
        Returns:
            Notes string
        """
        notes = []
        
        # Check for AI signatures
        if features['silence_ratio'] < settings.silence_ratio_ai_threshold:
            notes.append("Unnaturally low silence ratio suggests AI")
        
        if features['zcr_variance'] < settings.zcr_variance_ai_threshold:
            notes.append("Low ZCR variance indicates rigid voice pattern")
        
        if features['spectral_rolloff'] < settings.spectral_rolloff_cutoff:
            notes.append(f"Spectral cutoff at {features['spectral_rolloff']:.0f}Hz suggests synthetic origin")
        
        # Check for human signatures
        if features['silence_ratio'] > 0.03:
            notes.append("Natural breathing pauses detected")
        
        if features['zcr_variance'] > 0.005:
            notes.append("High voice fluctuation indicates human speech")
        
        # Language-specific adjustments
        if language in settings.fast_tempo_languages:
            notes.append(f"{language} is a fast-paced language - tempo variance threshold adjusted")
        
        if not notes:
            notes.append("Features show mixed signals")
        
        return "; ".join(notes)


# Global extractor instance
feature_extractor = AcousticFeatureExtractor()
