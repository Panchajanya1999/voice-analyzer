# AI Voice Detector API Documentation

## Core Classes

### AIVoiceDetector

The main class for voice classification and detection.

#### Constructor

```python
detector = AIVoiceDetector()
```

Creates a new instance of the AI Voice Detector.

#### Methods

##### extract_features(audio_file_path)

Extract comprehensive audio features for classification.

**Parameters:**
- `audio_file_path` (str): Path to the audio file

**Returns:**
- `numpy.ndarray`: Array of extracted features
- `None`: If feature extraction fails

**Features extracted:**
- Spectral features (centroid, rolloff, bandwidth)
- MFCC coefficients
- Chroma features
- Temporal features (RMS energy)
- Pitch and fundamental frequency
- Spectral contrast
- Harmonic features (Tonnetz)
- Custom AI detection features

##### train_model(human_audio_files, ai_audio_files)

Train the AI voice detection model.

**Parameters:**
- `human_audio_files` (list): List of paths to human voice samples
- `ai_audio_files` (list): List of paths to AI voice samples

**Raises:**
- `ValueError`: If no valid features can be extracted

##### predict(audio_file_path)

Predict if the voice is human or AI generated.

**Parameters:**
- `audio_file_path` (str): Path to the audio file to analyze

**Returns:**
- `tuple`: (result, confidence)
  - `result` (str): "Human" or "AI"
  - `confidence` (float): Confidence score (0-1)

**Raises:**
- `ValueError`: If model is not trained

##### save_model(model_path)

Save the trained model to disk.

**Parameters:**
- `model_path` (str): Path where to save the model

**Raises:**
- `ValueError`: If no trained model exists

##### load_model(model_path)

Load a pre-trained model from disk.

**Parameters:**
- `model_path` (str): Path to the saved model file

### RealTimeVoiceDetector

Class for real-time voice detection with recording capabilities.

#### Constructor

```python
detector = RealTimeVoiceDetector(model_path=None)
```

**Parameters:**
- `model_path` (str, optional): Path to pre-trained model

#### Methods

##### record_audio(filename="temp_recording.wav")

Record audio from the microphone.

**Parameters:**
- `filename` (str): Output filename for the recording

**Returns:**
- `str`: Path to the recorded file

##### detect_voice_type(audio_file=None)

Detect if voice is human or AI.

**Parameters:**
- `audio_file` (str, optional): Path to audio file. If None, records new audio

**Returns:**
- `tuple`: (result, confidence)

##### continuous_monitoring()

Start continuous voice monitoring mode.

## Utility Functions

### detect_voice_realtime(detector, audio_file_path)

Simple function to detect voice type from audio file.

**Parameters:**
- `detector` (AIVoiceDetector): Trained detector instance
- `audio_file_path` (str): Path to audio file

**Returns:**
- `tuple`: (result, confidence)

## Example Usage

### Basic Detection

```python
from src.voice_classifier import AIVoiceDetector

# Initialize and load model
detector = AIVoiceDetector()
detector.load_model("models/voice_detector_model.pkl")

# Analyze audio file
result, confidence = detector.predict("sample.wav")
print(f"Voice type: {result}, Confidence: {confidence:.2%}")
```

### Real-time Recording

```python
from main import RealTimeVoiceDetector

# Initialize with pre-trained model
detector = RealTimeVoiceDetector("models/voice_detector_model.pkl")

# Record and analyze
result, confidence = detector.detect_voice_type()
```

### Training Custom Model

```python
from src.voice_classifier import AIVoiceDetector

# Initialize detector
detector = AIVoiceDetector()

# Prepare training data
human_voices = ["human1.wav", "human2.wav", "human3.wav"]
ai_voices = ["ai1.wav", "ai2.wav", "ai3.wav"]

# Train model
detector.train_model(human_voices, ai_voices)

# Save model
detector.save_model("models/custom_model.pkl")
```