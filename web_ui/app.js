// WebSocket & System Control Client for NOVA HUD

const chatBox = document.getElementById('chat-box');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const micBtn = document.getElementById('micBtn');
const modelSelect = document.getElementById('modelSelect');
const drawer = document.getElementById('drawer');

let uiState = 'idle'; // 'idle' | 'user' | 'nova'
let speakTimeout;

const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${wsProtocol}://${location.host}/ws/chat`);

ws.onopen = () => {
    statusDot.classList.add('online');
    statusLabel.textContent = 'ONLINE';
    refreshStatus();
};

ws.onclose = () => {
    statusDot.classList.remove('online');
    statusLabel.textContent = 'OFFLINE';
};

ws.onmessage = (event) => {
    let payload;
    setUiState('nova', 4000);
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

function setUiState(state, timeoutMs = 0) {
    uiState = state;
    clearTimeout(speakTimeout);
    if (timeoutMs > 0) {
        speakTimeout = setTimeout(() => { uiState = 'idle'; }, timeoutMs);
    }
}

function toggleDrawer() {
    drawer.classList.toggle('open');
}

function sendMsg() {
    const input = document.getElementById('userInput');
    if (!input.value) return;
    appendMessage('You', input.value, 'user');
    ws.send(JSON.stringify({ type: 'text', text: input.value }));
    input.value = '';
    setUiState('nova', 3000);
}

function showLlmSpeed(metrics) {
    const speed = Number(metrics.tokens_per_second || 0);
    const tokenCount = Number(metrics.completion_tokens || 0);
    document.getElementById('statTokensPerSecond').innerText = tokenCount > 0 ? `${speed.toFixed(1)} tok/s` : '0.0 tok/s';
}

function appendMessage(sender, text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${className}`;
    msgDiv.innerText = `${sender}: ${text}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function playAudioBase64(b64) {
    const pcmBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const wavBuffer = pcmToWav(pcmBytes, 22050);
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const audio = new Audio(URL.createObjectURL(blob));
    
    setUiState('nova');
    audio.play().then(() => {
        setUiState('nova', (blob.size / 44100) * 1000 + 500);
    }).catch(() => setUiState('idle'));
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
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
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
        document.getElementById('statBrain').innerText = data.brain_loaded ? 'READY' : 'OFFLINE';
        document.getElementById('statStt').innerText = data.stt_loaded ? 'READY' : 'OFFLINE';
        document.getElementById('statTts').innerText = data.tts_loaded ? 'READY' : 'OFFLINE';
        populateModelSelect(data.models || []);
    } catch (e) {
        console.warn('Status refresh failed', e);
    }
}

function populateModelSelect(models) {
    const current = modelSelect.value;
    modelSelect.replaceChildren();
    for (const model of models) {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.installed ? model.label : `${model.label} [DL REQ]`;
        option.disabled = !model.installed;
        option.selected = model.active;
        modelSelect.appendChild(option);
    }
    modelSelect.disabled = !models.some(model => model.installed);
    if (current && !models.some(model => model.active)) modelSelect.value = current;
}

modelSelect.addEventListener('change', async () => {
    const selected = modelSelect.value;
    modelSelect.disabled = true;
    statusLabel.textContent = 'LOADING MODEL...';
    try {
        const response = await fetch(`/api/models/${encodeURIComponent(selected)}`, { method: 'POST' });
        if (!response.ok) throw new Error('Switch failed');
        await refreshStatus();
        statusLabel.textContent = 'ONLINE';
    } catch (error) {
        statusLabel.textContent = 'FAIL';
        await refreshStatus();
    }
});
setInterval(refreshStatus, 8000);

// Mic Capture (Hold-to-Talk)
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
        setUiState('user');
    } catch (e) {
        console.warn('Microphone unavailable:', e);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    micBtn.classList.remove('recording');
    setUiState('idle');
}

async function sendRecordedAudio() {
    if (!audioChunks.length) return;
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    const arrayBuffer = await blob.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    ws.send(JSON.stringify({ type: 'audio', audio_b64: b64, mime_type: blob.type, final: true }));
    appendMessage('You', '[Voice Message]', 'user');
    setUiState('nova', 3000);
}

// Canvas Visualizer Core
const canvas = document.getElementById('orbCanvas');
const ctx = canvas.getContext('2d');
let angle = 0;
let waveOffset = 0;

function drawHUDCore() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    let strokeColor = 'rgba(88, 80, 236, '; // Idle Violet/Indigo
    if (uiState === 'nova') strokeColor = 'rgba(0, 210, 255, '; // NOVA Cyan/Blue
    if (uiState === 'user') strokeColor = 'rgba(99, 102, 241, '; // User Violet

    // 1. Static Outer Ring
    ctx.beginPath();
    ctx.arc(cx, cy, 190, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.setLineDash([2, 10]);
    ctx.stroke();

    // 2. Rotating Segment Ring
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.arc(0, 0, 150, 0, Math.PI * 2);
    ctx.strokeStyle = strokeColor + '0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([40, 20, 10, 20]);
    ctx.stroke();
    ctx.restore();

    // 3. Counter-Rotating Inner Ring
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-angle * 1.6);
    ctx.beginPath();
    ctx.arc(0, 0, 110, 0, Math.PI * 2);
    ctx.strokeStyle = strokeColor + '0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([90, 40]);
    ctx.stroke();
    ctx.restore();

    // 4. Dynamic Audio Wave Core
    ctx.beginPath();
    const points = 120;
    const baseRadius = 70;
    let amp = uiState === 'idle' ? 3 : (uiState === 'user' ? 16 : 28);

    for (let i = 0; i <= points; i++) {
        const theta = (i / points) * Math.PI * 2;
        const distortion = Math.sin(theta * 7 + waveOffset) * Math.cos(theta * 3 - waveOffset) * amp;
        const r = baseRadius + distortion;
        const x = cx + r * Math.cos(theta);
        const y = cy + r * Math.sin(theta);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = strokeColor + '0.95)';
    ctx.lineWidth = 2;
    ctx.shadowBlur = uiState === 'idle' ? 8 : 20;
    ctx.shadowColor = strokeColor + '0.8)';
    ctx.stroke();
    ctx.shadowBlur = 0;

    angle += uiState === 'idle' ? 0.003 : 0.012;
    waveOffset += uiState === 'idle' ? 0.04 : 0.15;

    requestAnimationFrame(drawHUDCore);
}

drawHUDCore();