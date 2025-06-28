# main.py
"""
Real-time Voice Detection System
Main application for recording and analyzing voice samples to detect AI-generated speech
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading
from src.voice_classifier import AIVoiceDetector, detect_voice_realtime

class RealTimeVoiceDetector:
    def __init__(self, model_path=None):
        self.detector = AIVoiceDetector()
        self.is_recording = False
        self.sample_rate = 22050
        self.duration = 5  # 5 seconds
        
        if model_path:
            try:
                self.detector.load_model(model_path)
                print("Pre-trained model loaded successfully!")
            except:
                print("Could not load pre-trained model. Please train first.")
    
    def record_audio(self, filename="temp_recording.wav"):
        """Record 5 seconds of audio"""
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
        """Detect if voice is human or AI"""
        if not self.detector.is_trained:
            print("Error: Model not trained. Please train the model first.")
            return None, 0.0
        
        if audio_file is None:
            audio_file = self.record_audio()
        
        print("\nAnalyzing voice...")
        result, confidence = detect_voice_realtime(self.detector, audio_file)
        
        return result, confidence
    
    def continuous_monitoring(self):
        """Continuous voice monitoring mode"""
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

# Quick setup for training with example data
def quick_train_example():
    """Example training function - replace with your actual data"""
    detector = AIVoiceDetector()
    
    print("For training, you need:")
    print("1. Multiple human voice samples (5-10 seconds each)")
    print("2. Multiple AI voice samples (5-10 seconds each)")
    print("3. Save them as .wav files")
    
    # Example file lists (replace with your actual files)
    human_files = [
        # Add paths to your human voice recordings
        # "recordings/human_1.wav",
        # "recordings/human_2.wav",
        # etc.
    ]
    
    ai_files = [
        # Add paths to your AI voice recordings
        # "recordings/ai_1.wav", 
        # "recordings/ai_2.wav",
        # etc.
    ]
    
    if human_files and ai_files:
        print("Training model...")
        detector.train_model(human_files, ai_files)
        detector.save_model("voice_detector_model.pkl")
        print("Model trained and saved!")
    else:
        print("Please add your training files to the lists above")

# Command line interface
def main():
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

