# Voice Data Collection UI

A web interface for collecting voice samples to train the AI voice detection model.

## Features

- 🎙️ **Voice Recording**: 15-second voice recordings via web browser
- 🤖 **AI Sentence Generation**: Uses OpenAI API to generate unique sentences
- 👤 **Voice Type Selection**: Contributors can specify Human or AI voice
- 📊 **Real-time Stats**: Track collection progress
- 💾 **Automatic Storage**: Organized file storage with metadata
- 🌐 **Easy Sharing**: Simple web interface for friends to contribute

## Setup

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Note**: If no API key is provided, the app will use pre-defined fallback sentences.

### Running the Server

```bash
python app.py
```

The server will start on `http://localhost:6969`

## Usage

1. **Share the URL**: Send `http://your-ip:6969` to friends
2. **Select Voice Type**: Choose "Human Voice" or "AI Voice"
3. **Generate Sentence**: Click to get a new sentence to read
4. **Record**: Click record and read the sentence (auto-stops after 15s)
5. **Upload**: Preview and upload the recording

## File Structure

```
UI/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── js/
│       └── recorder.js   # Recording functionality
└── recordings/           # Stored voice samples
    ├── human/           # Human voice recordings
    └── ai/              # AI voice recordings
```

## Features Explained

### Voice Type Selection
- **Human Voice**: For actual human speech recordings
- **AI Voice**: For playing and recording AI-generated speech (TTS, etc.)

### Sentence Generation
- Uses OpenAI GPT-3.5-turbo to generate natural, conversational sentences
- Fallback sentences available if API is unavailable
- 10-20 words long, perfect for voice recording

### Recording Process
- Browser-based recording using WebRTC
- Automatic 15-second duration
- WebM format for cross-browser compatibility
- Real-time preview before upload

### Data Storage
- Organized into `human/` and `ai/` folders
- Each recording includes metadata JSON file
- Timestamped filenames with unique IDs
- Statistics tracking for progress monitoring

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Edge
- ❌ Safari (limited WebRTC support)

## Security Notes

- 🔒 Recordings stored locally on server
- 🚫 No data sent to third parties (except OpenAI for sentence generation)
- 🎯 Used only for AI voice detection research

## Troubleshooting

### Microphone Access Denied
- Check browser permissions
- Ensure HTTPS for production deployment
- Try refreshing the page

### OpenAI API Issues
- Verify `OPENAI_API_KEY` environment variable
- Check API quota and billing
- App will use fallback sentences if API fails

### Network Issues
- Ensure port 6969 is not blocked
- For remote access, use your actual IP address
- Consider using ngrok for external access

## Development

### Adding New Features
- Backend: Modify `app.py`
- Frontend: Update `templates/index.html` and `static/js/recorder.js`
- Styling: Edit `static/css/style.css`

### Custom Sentences
Edit the `FALLBACK_SENTENCES` list in `app.py` to add custom sentences.

## License

This tool is for educational and research purposes.