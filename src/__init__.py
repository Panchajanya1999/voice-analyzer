"""
AI Voice Detection Source Package
Contains core voice classification and detection modules
"""

from .voice_classifier import AIVoiceDetector, detect_voice_realtime

__all__ = ['AIVoiceDetector', 'detect_voice_realtime']