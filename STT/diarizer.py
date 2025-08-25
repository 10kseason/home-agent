import numpy as np
import librosa
from sklearn.cluster import KMeans

class SpeakerDiarizer:
    """Naive speaker diarizer using KMeans clustering on MFCC features."""
    def __init__(self, n_speakers: int = 2):
        self.n_speakers = n_speakers
        self.features = []
        self.model = None

    def _extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(13)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)

    def assign_speaker(self, segment: bytes, sr: int) -> int:
        audio = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
        feat = self._extract(audio, sr)
        self.features.append(feat)
        if self.model is None and len(self.features) >= self.n_speakers:
            self.model = KMeans(n_clusters=self.n_speakers, random_state=0).fit(self.features)
        if self.model is not None:
            return int(self.model.predict([feat])[0])
        return len(self.features) - 1
