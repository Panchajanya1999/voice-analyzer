"""
Voice Data Collection Web UI.

Flask application for collecting and managing voice samples. Provides a web
interface for recording human and AI-generated voices, generates unique sentences
using OpenAI's API, and handles audio file conversion from WebM to WAV format.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'recordings')

# Ensure upload folders exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'human'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'ai'), exist_ok=True)

# Check for OpenAI API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment variables!")
    print("The app will use pre-defined sentences instead of generating new ones.")
else:
    openai.api_key = OPENAI_API_KEY

# Pre-defined sentences as fallback
FALLBACK_SENTENCES = [
    "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains.",
    "Technology advances rapidly, changing how we communicate and interact with the world around us.",
    "In the garden, colorful butterflies dance among the blooming flowers on a warm summer day.",
    "The chef carefully prepared the gourmet meal, adding spices and herbs for perfect flavor.",
    "Ancient civilizations built remarkable structures that continue to amaze archaeologists today.",
    "The orchestra performed a beautiful symphony that moved the audience to tears of joy.",
    "Scientists discovered a new species of bird in the remote rainforest last month.",
    "Children played happily in the park while their parents watched from nearby benches.",
    "The library contains thousands of books covering every subject imaginable.",
    "Waves crashed against the rocky shore as seagulls circled overhead in the salty breeze."
]

@app.route('/')
def index():
    """
    Render the main page.

    Returns:
        str: Rendered HTML template for the voice data collection interface.
    """
    return render_template('index.html')

@app.route('/generate_sentence', methods=['GET'])
def generate_sentence():
    """
    Generate a sentence for voice recording.

    Uses OpenAI's API to generate a unique, natural sentence suitable for
    voice recording. Falls back to pre-defined sentences if the API is
    unavailable or if no API key is configured.

    Returns:
        flask.Response: JSON response containing:
            - success (bool): Whether sentence generation succeeded.
            - sentence (str): The generated or fallback sentence.
    """
    try:
        if OPENAI_API_KEY:
            # Use OpenAI to generate a sentence
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate only a single natural sentence for voice recording. Return just the sentence without any introduction, explanation, or quotes. The sentence should be 10-20 words long and easy to read aloud."
                    },
                    {
                        "role": "user",
                        "content": "Generate one sentence."
                    }
                ],
                max_tokens=50,
                temperature=0.9
            )
            sentence = response.choices[0].message.content.strip()
            
            # Clean up the response - extract just the sentence
            # Remove common prefixes and quotes
            prefixes_to_remove = [
                "Sure! Here is a sentence for voice recording:",
                "Here is a sentence for voice recording:",
                "Here's a sentence for you:",
                "Here is a sentence:",
                "Sure! Here's a sentence:",
                "Here's a sentence for voice recording:",
            ]
            
            for prefix in prefixes_to_remove:
                if sentence.startswith(prefix):
                    sentence = sentence[len(prefix):].strip()
                    break
            
            # Remove quotes and extra whitespace
            sentence = sentence.strip('"\'').strip()
            
            # If there are multiple sentences, take the first one
            if '.' in sentence and not sentence.endswith('.'):
                sentence = sentence.split('.')[0] + '.'
        else:
            # Use fallback sentence
            import random
            sentence = random.choice(FALLBACK_SENTENCES)
        
        return jsonify({
            'success': True,
            'sentence': sentence
        })
    
    except Exception as e:
        print(f"Error generating sentence: {e}")
        # Return a fallback sentence on error
        import random
        return jsonify({
            'success': True,
            'sentence': random.choice(FALLBACK_SENTENCES)
        })

@app.route('/upload_recording', methods=['POST'])
def upload_recording():
    """
    Handle audio file upload and convert to WAV format.

    Receives WebM audio files from the browser, converts them to WAV format
    with standardized settings (22050 Hz, mono, 16-bit), and saves them to
    the appropriate directory (human or ai) with associated metadata.

    Form Parameters:
        audio (file): The audio file in WebM format.
        voice_type (str): Type of voice - either 'human' or 'ai'.
        sentence (str): The sentence that was recorded.

    Returns:
        flask.Response: JSON response containing:
            - success (bool): Whether upload and conversion succeeded.
            - message (str): Success or error message.
            - filename (str): Name of the saved WAV file (on success).
            - error (str): Error description (on failure).
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        voice_type = request.form.get('voice_type', 'human').lower()
        sentence = request.form.get('sentence', 'unknown')
        
        if voice_type not in ['human', 'ai']:
            return jsonify({'success': False, 'error': 'Invalid voice type'}), 400
        
        # Generate unique filename for WAV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        wav_filename = f"{voice_type}_{timestamp}_{unique_id}.wav"
        
        # Create temporary file for WebM
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            audio_file.save(temp_webm.name)
            temp_webm_path = temp_webm.name
        
        try:
            # Convert WebM to WAV using pydub
            print(f"Converting {temp_webm_path} to WAV...")
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            
            # Ensure consistent audio format
            audio = audio.set_frame_rate(22050)  # Match the sample rate used in voice_classifier
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Save as WAV
            folder = os.path.join(app.config['UPLOAD_FOLDER'], voice_type)
            wav_filepath = os.path.join(folder, wav_filename)
            audio.export(wav_filepath, format="wav")
            
            print(f"Successfully converted to WAV: {wav_filename}")
            
        except Exception as e:
            print(f"Error converting audio: {e}")
            return jsonify({'success': False, 'error': f'Audio conversion failed: {str(e)}'}), 500
        
        finally:
            # Clean up temporary WebM file
            if os.path.exists(temp_webm_path):
                os.unlink(temp_webm_path)
        
        # Save metadata
        metadata = {
            'filename': wav_filename,
            'voice_type': voice_type,
            'sentence': sentence,
            'timestamp': timestamp,
            'format': 'wav',
            'sample_rate': 22050,
            'channels': 1,
            'file_size': os.path.getsize(wav_filepath)
        }
        
        metadata_file = os.path.join(folder, f"{wav_filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Recording saved successfully as {voice_type} voice sample (converted to WAV)',
            'filename': wav_filename
        })
    
    except Exception as e:
        print(f"Error uploading recording: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get statistics about collected voice recordings.

    Counts the number of WAV files in both the human and AI recording directories.

    Returns:
        flask.Response: JSON response containing:
            - success (bool): Whether statistics were retrieved successfully.
            - human_recordings (int): Number of human voice samples.
            - ai_recordings (int): Number of AI voice samples.
            - total_recordings (int): Total number of all samples.
            - error (str): Error description (on failure).
    """
    try:
        # Count .wav files only
        human_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'human')
        ai_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'ai')
        
        human_count = len([f for f in os.listdir(human_dir) if f.endswith('.wav')]) if os.path.exists(human_dir) else 0
        ai_count = len([f for f in os.listdir(ai_dir) if f.endswith('.wav')]) if os.path.exists(ai_dir) else 0
        
        return jsonify({
            'success': True,
            'human_recordings': human_count,
            'ai_recordings': ai_count,
            'total_recordings': human_count + ai_count
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Voice Data Collection Server")
    print("=" * 50)
    if OPENAI_API_KEY:
        print("✅ OpenAI API key found - will generate unique sentences")
    else:
        print("⚠️  No OpenAI API key found - using fallback sentences")
    print(f"📁 Recordings will be saved to: {app.config['UPLOAD_FOLDER']}")
    print("🌐 Starting server on http://localhost:6969")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=6969, debug=True)