"""
Voice Data Collection Web UI
Flask application for collecting voice samples with OpenAI-generated sentences
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
    """Render the main page"""
    return render_template('index.html')

@app.route('/generate_sentence', methods=['GET'])
def generate_sentence():
    """Generate a sentence using OpenAI or return a fallback sentence"""
    try:
        if OPENAI_API_KEY:
            # Use OpenAI to generate a sentence
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates simple, natural sentences for voice recording. The sentences should be 10-20 words long, use common vocabulary, and be easy to read aloud."
                    },
                    {
                        "role": "user",
                        "content": "Generate a single sentence that someone can read aloud for voice recording. Make it natural and conversational."
                    }
                ],
                max_tokens=50,
                temperature=0.9
            )
            sentence = response.choices[0].message.content.strip()
            # Remove quotes if present
            sentence = sentence.strip('"\'')
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
    """Handle audio file upload"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        voice_type = request.form.get('voice_type', 'human').lower()
        sentence = request.form.get('sentence', 'unknown')
        
        if voice_type not in ['human', 'ai']:
            return jsonify({'success': False, 'error': 'Invalid voice type'}), 400
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{voice_type}_{timestamp}_{unique_id}.webm"
        
        # Save the file
        folder = os.path.join(app.config['UPLOAD_FOLDER'], voice_type)
        filepath = os.path.join(folder, filename)
        audio_file.save(filepath)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'voice_type': voice_type,
            'sentence': sentence,
            'timestamp': timestamp,
            'file_size': os.path.getsize(filepath)
        }
        
        metadata_file = os.path.join(folder, f"{filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Recording saved successfully as {voice_type} voice sample',
            'filename': filename
        })
    
    except Exception as e:
        print(f"Error uploading recording: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about collected recordings"""
    try:
        human_count = len(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'human'))) // 2  # Divide by 2 because of .json files
        ai_count = len(os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'ai'))) // 2
        
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