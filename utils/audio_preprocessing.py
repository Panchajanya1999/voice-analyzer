"""
Audio preprocessing utilities.

Provides functions for audio normalization, silence removal, noise reduction,
and resampling. These utilities prepare audio data for feature extraction
and classification.
"""

import numpy as np
import librosa

def normalize_audio(audio_signal, target_db=-20):
    """
    Normalize audio signal to target dB level.

    Adjusts the amplitude of the audio signal so that its RMS (Root Mean Square)
    level matches the specified target level in decibels.

    Args:
        audio_signal (numpy.ndarray): Array of audio samples to normalize.
        target_db (float, optional): Target RMS level in dB. Defaults to -20.

    Returns:
        numpy.ndarray: Normalized audio signal with adjusted amplitude.
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_signal**2))
    
    # Convert target dB to linear scale
    target_rms = 10**(target_db/20)
    
    # Calculate scaling factor
    if rms > 0:
        scale = target_rms / rms
        return audio_signal * scale
    return audio_signal

def remove_silence(audio_signal, sr=22050, threshold_db=-40):
    """
    Remove silence from beginning and end of audio.

    Identifies and removes silent intervals from the start and end of the audio
    signal using a decibel threshold. Preserves all non-silent audio in between.

    Args:
        audio_signal (numpy.ndarray): Array of audio samples to process.
        sr (int, optional): Sample rate in Hz. Defaults to 22050.
        threshold_db (float, optional): Silence threshold in dB. Audio below
            this level is considered silent. Defaults to -40.

    Returns:
        numpy.ndarray: Trimmed audio signal with leading and trailing silence removed.
    """
    # Convert threshold to amplitude
    threshold = 10**(threshold_db/20)
    
    # Find non-silent intervals
    intervals = librosa.effects.split(audio_signal, top_db=-threshold_db)
    
    if len(intervals) > 0:
        # Get start and end of non-silent audio
        start = intervals[0][0]
        end = intervals[-1][1]
        return audio_signal[start:end]
    
    return audio_signal

def apply_noise_reduction(audio_signal, sr=22050):
    """
    Apply basic noise reduction.

    Placeholder function for noise reduction functionality. Currently returns
    the input signal unchanged. In a complete implementation, this would apply
    spectral subtraction or other noise reduction techniques.

    Args:
        audio_signal (numpy.ndarray): Array of audio samples to process.
        sr (int, optional): Sample rate in Hz. Defaults to 22050.

    Returns:
        numpy.ndarray: Audio signal (currently unmodified).

    Note:
        This is a placeholder. Real implementation would include sophisticated
        noise reduction algorithms like spectral subtraction or Wiener filtering.
    """
    # Simple high-pass filter to remove low-frequency noise
    # This is a placeholder - real implementation would be more sophisticated
    return audio_signal

def resample_audio(audio_signal, original_sr, target_sr=22050):
    """
    Resample audio to target sample rate.

    Converts audio from one sample rate to another using high-quality resampling.
    If the original and target sample rates are the same, returns the signal unchanged.

    Args:
        audio_signal (numpy.ndarray): Array of audio samples to resample.
        original_sr (int): Original sample rate in Hz.
        target_sr (int, optional): Target sample rate in Hz. Defaults to 22050.

    Returns:
        numpy.ndarray: Resampled audio signal at the target sample rate.
    """
    if original_sr != target_sr:
        return librosa.resample(audio_signal, orig_sr=original_sr, target_sr=target_sr)
    return audio_signal