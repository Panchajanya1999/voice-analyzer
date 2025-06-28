// Voice Recording Application
let mediaRecorder;
let audioChunks = [];
let recordingTimer;
let seconds = 0;
let currentSentence = '';
let voiceType = 'human';
let audioBlob = null;

// DOM Elements
const recordBtn = document.getElementById('record-btn');
const generateBtn = document.getElementById('generate-btn');
const uploadBtn = document.getElementById('upload-btn');
const retryBtn = document.getElementById('retry-btn');
const sentenceText = document.getElementById('sentence-text');
const timer = document.getElementById('timer');
const timerText = document.getElementById('timer-text');
const audioPreview = document.getElementById('audio-preview');
const audioPlayer = document.getElementById('audio-player');
const statusMessage = document.getElementById('status-message');
const voiceTypeBtns = document.querySelectorAll('.voice-type-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    updateStats();
    
    // Voice type selection
    voiceTypeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            voiceTypeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            voiceType = btn.dataset.type;
        });
    });
    
    // Button event listeners
    generateBtn.addEventListener('click', generateSentence);
    recordBtn.addEventListener('click', toggleRecording);
    uploadBtn.addEventListener('click', uploadRecording);
    retryBtn.addEventListener('click', resetRecording);
});

// Generate new sentence
async function generateSentence() {
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';
    
    try {
        const response = await fetch('/generate_sentence');
        const data = await response.json();
        
        if (data.success) {
            currentSentence = data.sentence;
            sentenceText.textContent = currentSentence;
            recordBtn.disabled = false;
            showStatus('New sentence generated! Ready to record.', 'info');
        } else {
            showStatus('Failed to generate sentence. Please try again.', 'error');
        }
    } catch (error) {
        console.error('Error generating sentence:', error);
        showStatus('Connection error. Please check your internet.', 'error');
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate New Sentence';
    }
}

// Toggle recording
async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        await startRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
            audioPreview.style.display = 'block';
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        
        // Update UI
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.btn-text').textContent = 'Stop Recording';
        timer.style.display = 'block';
        generateBtn.disabled = true;
        
        // Start timer
        seconds = 0;
        updateTimer();
        recordingTimer = setInterval(() => {
            seconds++;
            updateTimer();
            
            // Auto-stop after 15 seconds
            if (seconds >= 15) {
                stopRecording();
            }
        }, 1000);
        
        showStatus('Recording... Speak clearly!', 'info');
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        showStatus('Microphone access denied. Please allow microphone access.', 'error');
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(recordingTimer);
        
        // Update UI
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.btn-text').textContent = 'Start Recording';
        recordBtn.disabled = true;
        timer.style.display = 'none';
        
        showStatus('Recording complete! Preview your recording below.', 'success');
    }
}

// Upload recording
async function uploadRecording() {
    if (!audioBlob) {
        showStatus('No recording to upload!', 'error');
        return;
    }
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    const formData = new FormData();
    formData.append('audio', audioBlob, `recording_${Date.now()}.webm`);
    formData.append('voice_type', voiceType);
    formData.append('sentence', currentSentence);
    
    try {
        const response = await fetch('/upload_recording', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(`✅ ${data.message}`, 'success');
            updateStats();
            
            // Reset after successful upload
            setTimeout(() => {
                resetRecording();
                generateSentence();
            }, 2000);
        } else {
            showStatus(`Upload failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showStatus('Upload failed. Please check your connection.', 'error');
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Recording';
    }
}

// Reset recording
function resetRecording() {
    audioChunks = [];
    audioBlob = null;
    audioPreview.style.display = 'none';
    recordBtn.disabled = false;
    generateBtn.disabled = false;
    seconds = 0;
    updateTimer();
    statusMessage.style.display = 'none';
}

// Update timer display
function updateTimer() {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    timerText.textContent = `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Show status message
function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusMessage.style.display = 'none';
    }, 5000);
}

// Update statistics
async function updateStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('human-count').textContent = data.human_recordings;
            document.getElementById('ai-count').textContent = data.ai_recordings;
            document.getElementById('total-count').textContent = data.total_recordings;
        }
    } catch (error) {
        console.error('Error fetching stats:', error);
    }
}

// Check browser support
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStatus('Your browser does not support audio recording. Please use Chrome, Firefox, or Edge.', 'error');
    recordBtn.disabled = true;
}