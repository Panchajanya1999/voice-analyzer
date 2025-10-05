"""
Real-time Voice Detection System.

Main application for recording and analyzing voice samples to detect AI-generated speech.
This module provides a command-line interface for recording audio, training models,
and detecting whether voice samples are human or AI-generated.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading
import os
from src.voice_classifier import AIVoiceDetector, detect_voice_realtime

class RealTimeVoiceDetector:
    """
    Real-time voice detection system for identifying AI-generated speech.

    This class provides methods for recording audio, detecting voice types,
    and continuous monitoring of voice samples.

    Attributes:
        detector (AIVoiceDetector): The underlying AI voice detection model.
        is_recording (bool): Flag indicating if currently recording.
        sample_rate (int): Audio sample rate in Hz (default: 22050).
        duration (int): Recording duration in seconds (default: 5).
    """

    def __init__(self, model_path=None):
        """
        Initialize the real-time voice detector.

        Args:
            model_path (str, optional): Path to a pre-trained model file.
                If provided, attempts to load the model. Defaults to None.
        """
        self.detector = AIVoiceDetector()
        self.is_recording = False
        self.sample_rate = 22050
        self.duration = 5  # 5 seconds

        if model_path:
            try:
                self.detector.load_model(model_path)
                print("Pre-trained model loaded successfully!")
            except Exception as e:
                print(f"Could not load pre-trained model: {e}. Please train first.")
    
    def record_audio(self, filename="temp_recording.wav"):
        """
        Record audio from the default microphone.

        Records audio for the specified duration (default 5 seconds) and saves
        it to a WAV file. Displays a countdown during recording.

        Args:
            filename (str, optional): Output filename for the recording.
                Defaults to "temp_recording.wav".

        Returns:
            str: The filename where the audio was saved.
        """
        print(f"Recording for {self.duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        
        # Show countdown
        for i in range(self.duration, 0, -1):
            print(f"Recording... {i}")
            time.sleep(1)
        
        sd.wait()  # Wait for recording to finish
        
        # Save to file
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Recording saved as {filename}")
        
        return filename
    
    def detect_voice_type(self, audio_file=None):
        """
        Detect whether a voice sample is human or AI-generated.

        If no audio file is provided, records a new sample. Requires a trained model.

        Args:
            audio_file (str, optional): Path to audio file to analyze.
                If None, records a new sample. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - result (str or None): "Human" or "AI" if detection succeeds, None on error.
                - confidence (float): Confidence score between 0.0 and 1.0.
        """
        if not self.detector.is_trained:
            print("Error: Model not trained. Please train the model first.")
            return None, 0.0
        
        if audio_file is None:
            audio_file = self.record_audio()
        
        print("\nAnalyzing voice...")
        result, confidence = detect_voice_realtime(self.detector, audio_file)
        
        return result, confidence
    
    def continuous_monitoring(self):
        """
        Run continuous voice monitoring mode.

        Repeatedly prompts the user to record audio samples and analyzes each one.
        Continues until interrupted with Ctrl+C.

        Raises:
            KeyboardInterrupt: When user presses Ctrl+C to stop monitoring.
        """
        print("Starting continuous monitoring mode...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                input("Press Enter to start 5-second recording...")
                result, confidence = self.detect_voice_type()
                
                if result:
                    print(f"\n{'='*50}")
                    print(f"DETECTION RESULT: {result}")
                    print(f"CONFIDENCE: {confidence:.2%}")
                    print(f"{'='*50}\n")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping continuous monitoring...")

def quick_train_example():
    """
    Automatically discover and train on voice samples from UI recordings.

    Scans the UI/recordings directory for human and AI voice samples,
    checks for data quality and balance, then trains a new model.
    Saves the trained model as 'voice_detector_model.pkl'.

    The function looks for WAV files in:
        - UI/recordings/human/ for human voice samples
        - UI/recordings/ai/ for AI-generated voice samples

    Provides warnings if insufficient samples are found or if the dataset
    is imbalanced.
    """
    detector = AIVoiceDetector()
    
    print("🔍 Scanning for voice recordings...")
    
    # Look for WAV files in UI recordings folders
    ui_recordings_path = os.path.join(os.path.dirname(__file__), "UI", "recordings")
    human_dir = os.path.join(ui_recordings_path, "human")
    ai_dir = os.path.join(ui_recordings_path, "ai")
    
    # Find all WAV files
    human_files = []
    ai_files = []
    
    if os.path.exists(human_dir):
        human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.wav')]
        print(f"📁 Found {len(human_files)} human voice samples in {human_dir}")
    else:
        print(f"❌ Human recordings directory not found: {human_dir}")
    
    if os.path.exists(ai_dir):
        ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.endswith('.wav')]
        print(f"📁 Found {len(ai_files)} AI voice samples in {ai_dir}")
    else:
        print(f"❌ AI recordings directory not found: {ai_dir}")
    
    # Check if we have enough samples
    if len(human_files) < 3:
        print("⚠️  Warning: Less than 3 human voice samples found. Model may not train well.")
    if len(ai_files) < 3:
        print("⚠️  Warning: Less than 3 AI voice samples found. Model may not train well.")
    
    if human_files and ai_files:
        print(f"\n🏋️  Training model with {len(human_files)} human and {len(ai_files)} AI samples...")
        try:
            detector.train_model(human_files, ai_files)
            detector.save_model("voice_detector_model.pkl")
            print("✅ Model trained and saved as 'voice_detector_model.pkl'!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
    else:
        print("\n❌ No voice samples found!")
        print("💡 To collect samples:")
        print("   1. Go to the UI folder: cd UI")
        print("   2. Start the web server: python app.py") 
        print("   3. Visit http://localhost:6969")
        print("   4. Record some voice samples")
        print("   5. Come back and run training again")

def main():
    """
    Run the main command-line interface for the AI Voice Detection System.

    Provides an interactive menu with options to:
        1. Record and detect voice samples
        2. Enter continuous monitoring mode
        3. Train a new model
        4. Exit the application

    Attempts to load an existing model at startup if available.
    """
    print("AI Voice Detection System")
    print("=" * 40)
    
    model_path = "voice_detector_model.pkl"
    
    # Try to load existing model
    try:
        detector_system = RealTimeVoiceDetector(model_path)
    except:
        detector_system = RealTimeVoiceDetector()
    
    while True:
        print("\nOptions:")
        print("1. Record and detect voice (5 seconds)")
        print("2. Continuous monitoring mode")
        print("3. Train new model")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            result, confidence = detector_system.detect_voice_type()
            if result:
                print(f"\n🎯 RESULT: {result} voice detected!")
                print(f"📊 Confidence: {confidence:.1%}")
                
                # Simple interpretation
                if confidence > 0.8:
                    print("🟢 High confidence in result")
                elif confidence > 0.6:
                    print("🟡 Moderate confidence in result")
                else:
                    print("🔴 Low confidence - result may be uncertain")
        
        elif choice == "2":
            detector_system.continuous_monitoring()
        
        elif choice == "3":
            quick_train_example()
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

