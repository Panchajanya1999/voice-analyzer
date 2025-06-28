#!/usr/bin/env python3
"""
Example Usage Script for AI Voice Detector
Simple examples showing different ways to use the voice detection system
"""

import os
import sys
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_classifier import AIVoiceDetector
from main import RealTimeVoiceDetector

def example_single_detection():
    """Example: Single voice detection"""
    print("=== Single Voice Detection Example ===")
    
    # Initialize detector with pre-trained model (if available)
    try:
        detector = RealTimeVoiceDetector("voice_detector_model.pkl")
        print("✅ Loaded pre-trained model")
    except:
        print("❌ No pre-trained model found. Please train first.")
        return
    
    print("This will record 5 seconds of audio and analyze it...")
    input("Press Enter when ready to speak...")
    
    result, confidence = detector.detect_voice_type()
    
    if result:
        print(f"\n🎯 Detection Result: {result}")
        print(f"📊 Confidence: {confidence:.1%}")
        
        # Interpret confidence level
        if confidence > 0.8:
            print("🟢 High confidence - very reliable result")
        elif confidence > 0.6:
            print("🟡 Moderate confidence - fairly reliable")
        else:
            print("🔴 Low confidence - result uncertain")
    else:
        print("❌ Detection failed")

def example_batch_processing():
    """Example: Process multiple audio files"""
    print("\n=== Batch Processing Example ===")
    
    # Sample audio files (replace with your actual files)
    audio_files = [
        "sample1.wav",
        "sample2.wav", 
        "sample3.wav"
    ]
    
    detector = AIVoiceDetector()
    
    try:
        detector.load_model("voice_detector_model.pkl")
    except:
        print("❌ No model found. Please train first.")
        return
    
    print(f"Processing {len(audio_files)} audio files...")
    
    results = []
    for i, audio_file in enumerate(audio_files, 1):
        if os.path.exists(audio_file):
            print(f"Processing file {i}/{len(audio_files)}: {audio_file}")
            result, confidence = detector.predict(audio_file)
            results.append((audio_file, result, confidence))
            print(f"  → {result} ({confidence:.1%} confidence)")
        else:
            print(f"  → File not found: {audio_file}")
    
    # Summary
    print(f"\n📋 Summary:")
    human_count = sum(1 for _, result, _ in results if result == "Human")
    ai_count = sum(1 for _, result, _ in results if result == "AI")
    print(f"Human voices: {human_count}")
    print(f"AI voices: {ai_count}")

def example_training():
    """Example: Train a new model"""
    print("\n=== Model Training Example ===")
    
    # Example training data paths (replace with your actual files)
    human_samples = [
        "training_data/human/person1_sample1.wav",
        "training_data/human/person1_sample2.wav",
        "training_data/human/person2_sample1.wav",
        "training_data/human/person2_sample2.wav",
    ]
    
    ai_samples = [
        "training_data/ai/tts1_sample1.wav",
        "training_data/ai/tts1_sample2.wav", 
        "training_data/ai/tts2_sample1.wav",
        "training_data/ai/tts2_sample2.wav",
    ]
    
    # Check if training files exist
    existing_human = [f for f in human_samples if os.path.exists(f)]
    existing_ai = [f for f in ai_samples if os.path.exists(f)]
    
    if not existing_human or not existing_ai:
        print("❌ Training data not found.")
        print("Please create the following directory structure:")
        print("training_data/")
        print("  ├── human/")
        print("  │   ├── person1_sample1.wav")
        print("  │   └── ...")
        print("  └── ai/")
        print("      ├── tts1_sample1.wav")
        print("      └── ...")
        return
    
    print(f"Found {len(existing_human)} human samples")
    print(f"Found {len(existing_ai)} AI samples")
    
    # Train model
    detector = AIVoiceDetector()
    print("🏋️ Training model...")
    
    try:
        detector.train_model(existing_human, existing_ai)
        detector.save_model("custom_voice_model.pkl")
        print("✅ Model trained and saved as 'custom_voice_model.pkl'")
    except Exception as e:
        print(f"❌ Training failed: {e}")

def example_continuous_monitoring():
    """Example: Continuous monitoring mode"""
    print("\n=== Continuous Monitoring Example ===")
    
    try:
        detector = RealTimeVoiceDetector("voice_detector_model.pkl")
    except:
        print("❌ No pre-trained model found.")
        return
    
    print("Starting continuous monitoring...")
    print("The system will repeatedly ask you to speak for 5 seconds")
    print("Press Ctrl+C to stop")
    
    try:
        count = 0
        while count < 3:  # Limit to 3 samples for example
            count += 1
            print(f"\n--- Sample {count} ---")
            input("Press Enter to start recording...")
            
            result, confidence = detector.detect_voice_type()
            if result:
                print(f"🎯 Result: {result} ({confidence:.1%})")
            
            time.sleep(1)
            
        print("\n✅ Monitoring example completed")
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

def main():
    """Main example runner"""
    print("🎙️ AI Voice Detector - Examples")
    print("=" * 40)
    
    examples = [
        ("1", "Single Voice Detection", example_single_detection),
        ("2", "Batch Processing", example_batch_processing),
        ("3", "Model Training", example_training),
        ("4", "Continuous Monitoring", example_continuous_monitoring),
    ]
    
    while True:
        print("\nAvailable Examples:")
        for key, name, _ in examples:
            print(f"  {key}. {name}")
        print("  q. Quit")
        
        choice = input("\nSelect example (1-4, q): ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        
        # Find and run selected example
        for key, name, func in examples:
            if choice == key:
                try:
                    func()
                except KeyboardInterrupt:
                    print("\n🛑 Example interrupted")
                except Exception as e:
                    print(f"❌ Example failed: {e}")
                break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

