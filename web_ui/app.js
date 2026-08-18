// nova/web_ui/app.js
// WebSocket client for NOVA's dashboard: handles typed text, streamed
// audio replies, and (optionally) mic capture for voice input.

const chatBox = document.getElementById('chat-box');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const micBtn = document.getElementById('micBtn');

const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${wsProtocol}://${location.host}/ws/chat`);

ws.onopen = () => {
    statusDot.classList.add('online');
    statusLabel.textContent = 'Connected';
    refreshStatus();
};

ws.onclose = () => {
    statusDot.classList.remove('online');
    statusLabel.textContent = 'Disconnected';
};

ws.onmessage = (event) => {
    let payload;
    try {
        payload = JSON.parse(event.data);
    } catch (e) {
        appendMessage('NOVA', event.data, 'nova');
        return;
    }

    if (payload.type === 'text') {
        appendMessage('NOVA', payload.text, 'nova');
    } else if (payload.type === 'metrics' && payload.llm) {
        showLlmSpeed(payload.llm);
    } else if (payload.type === 'audio' && payload.audio_b64) {
        playAudioBase64(payload.audio_b64);
    }
};

function sendMsg() {
    const input = document.getElementById('userInput');
    if (!input.value) return;
    appendMessage('You', input.value, 'user');
    ws.send(JSON.stringify({ type: 'text', text: input.value }));
    input.value = '';
}

function showLlmSpeed(metrics) {
    const speed = Number(metrics.tokens_per_second || 0);
    const tokenCount = Number(metrics.completion_tokens || 0);
    const display = tokenCount > 0 ? `${speed.toFixed(1)} tok/s` : '0.0 tok/s';
    document.getElementById('statTokensPerSecond').innerText = display;
}

function appendMessage(sender, text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${className}`;
    msgDiv.innerText = `${sender}: ${text}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function playAudioBase64(b64) {
    // Raw PCM from Piper needs a WAV header to play directly in <audio>.
    // For a quick dashboard test this assumes 22050Hz mono 16-bit PCM;
    // adjust to match your chosen Piper voice's sample rate.
    const pcmBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const wavBuffer = pcmToWav(pcmBytes, 22050);
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const audio = new Audio(URL.createObjectURL(blob));
    audio.play().catch(() => {/* autoplay may be blocked until user interacts */});
}

function pcmToWav(pcmBytes, sampleRate) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + pcmBytes.length, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);   // PCM
    view.setUint16(22, 1, true);   // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, pcmBytes.length, true);

    const wav = new Uint8Array(44 + pcmBytes.length);
    wav.set(new Uint8Array(header), 0);
    wav.set(pcmBytes, 44);
    return wav.buffer;
}

async function refreshStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('statBrain').innerText = data.brain_loaded ? 'Loaded' : 'Stub';
        document.getElementById('statStt').innerText = data.stt_loaded ? 'Loaded' : 'Stub';
        document.getElementById('statTts').innerText = data.tts_loaded ? 'Loaded' : 'Stub';
        document.getElementById('statProject').innerText = data.active_project || 'None';
    } catch (e) {
        console.warn('Status refresh failed', e);
    }
}
setInterval(refreshStatus, 8000);

// --- Mic capture (hold-to-talk) -------------------------------------------
let mediaRecorder = null;
let audioChunks = [];

micBtn.addEventListener('mousedown', startRecording);
micBtn.addEventListener('mouseup', stopRecording);
micBtn.addEventListener('mouseleave', stopRecording);

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = sendRecordedAudio;
        mediaRecorder.start();
        micBtn.classList.add('recording');
    } catch (e) {
        console.warn('Microphone unavailable:', e);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    micBtn.classList.remove('recording');
}

async function sendRecordedAudio() {
    if (!audioChunks.length) return;
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    const arrayBuffer = await blob.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    ws.send(JSON.stringify({ type: 'audio', audio_b64: b64, final: true }));
    appendMessage('You', '[voice message]', 'user');
}
