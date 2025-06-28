# voice_classifier.py
"""
AI Voice Detection Core Module
Provides the main AIVoiceDetector class for training and classifying audio samples
"""

import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import scipy.stats
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

class AIVoiceDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, audio_file_path):
        """Extract comprehensive audio features for classification"""
        try:
            # Load audio
            y, sr = librosa.load(audio_file_path, sr=22050, duration=5.0)
            
            # Basic audio features
            features = []
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Statistical measures of spectral features
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
            ])
            
            # 2. MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([np.mean(chroma), np.std(chroma)])
            
            # 4. Temporal features
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms)])
            
            # 5. Pitch and fundamental frequency
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features.extend([np.mean(pitch_values), np.std(pitch_values)])
            else:
                features.extend([0, 0])
            
            # 6. Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])
            
            # 7. Tonnetz (harmonic features)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.extend([np.mean(tonnetz), np.std(tonnetz)])
            
            # 8. Custom AI detection features
            # Periodicity measure
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            periodicity = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] != 0 else 0
            features.append(periodicity)
            
            # Harmonic-to-noise ratio
            harmonic, percussive = librosa.effects.hpss(y)
            hnr = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
            features.append(hnr)
            
            # Spectral flux (measure of spectral change)
            stft = librosa.stft(y)
            spectral_flux = np.mean(np.diff(np.abs(stft), axis=1))
            features.append(spectral_flux)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def train_model(self, human_audio_files, ai_audio_files):
        """Train the AI voice detection model"""
        print("Extracting features from training data...")
        
        # Extract features from human voices
        human_features = []
        for file_path in human_audio_files:
            features = self.extract_features(file_path)
            if features is not None:
                human_features.append(features)
        
        # Extract features from AI voices
        ai_features = []
        for file_path in ai_audio_files:
            features = self.extract_features(file_path)
            if features is not None:
                ai_features.append(features)
        
        if not human_features or not ai_features:
            raise ValueError("No valid features extracted from training data")
        
        # Prepare training data
        X = np.vstack([human_features, ai_features])
        y = np.hstack([np.zeros(len(human_features)), np.ones(len(ai_features))])  # 0=human, 1=AI
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")
        
        self.is_trained = True
        
    def predict(self, audio_file_path):
        """Predict if the voice is human or AI generated"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Extract features
        features = self.extract_features(audio_file_path)
        if features is None:
            return None, 0.0
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled)[0])
        
        return "AI" if prediction == 1 else "Human", confidence
    
    def save_model(self, model_path):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {model_path}")

# Real-time detection function
def detect_voice_realtime(detector, audio_file_path):
    """Simple function to detect voice type from audio file"""
    try:
        result, confidence = detector.predict(audio_file_path)
        if result:
            print(f"Detection Result: {result}")
            print(f"Confidence: {confidence:.2%}")
            return result, confidence
        else:
            print("Error: Could not analyze audio file")
            return None, 0.0
    except Exception as e:
        print(f"Error during detection: {e}")
        return None, 0.0

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = AIVoiceDetector()
    
    # Example training (you would replace these with actual file paths)
    """
    human_files = [
        "human_voice_1.wav", "human_voice_2.wav", "human_voice_3.wav"
    ]
    ai_files = [
        "ai_voice_1.wav", "ai_voice_2.wav", "ai_voice_3.wav"
    ]
    
    # Train the model
    detector.train_model(human_files, ai_files)
    
    # Save the model
    detector.save_model("ai_voice_detector.pkl")
    """
    
    # For inference, load pre-trained model
    # detector.load_model("ai_voice_detector.pkl")
    
    # Test on new audio file
    # result, confidence = detect_voice_realtime(detector, "test_audio.wav")
    
    print("AI Voice Detector initialized successfully!")
    print("To use:")
    print("1. Collect training data (human and AI voice samples)")
    print("2. Train the model using train_model()")
    print("3. Use predict() to classify new audio files")

