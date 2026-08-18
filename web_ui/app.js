// WebSocket & System Control Client for NOVA HUD

const chatBox = document.getElementById('chat-box');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const micBtn = document.getElementById('micBtn');
const modelSelect = document.getElementById('modelSelect');
const drawer = document.getElementById('drawer');

let uiState = 'idle'; // 'idle' | 'user' | 'thinking' | 'nova'
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
    try {
        payload = JSON.parse(event.data);
    } catch (e) {
        appendMessage('NOVA', event.data, 'nova');
        setUiState('nova', 4000);
        return;
    }

    if (payload.type === 'text') {
        appendMessage('NOVA', payload.text, 'nova');
        setUiState('nova', 4000);
    } else if (payload.type === 'metrics' && payload.llm) {
        showLlmSpeed(payload.llm);
    } else if (payload.type === 'audio' && payload.audio_b64) {
        playAudioBase64(payload.audio_b64);
    }
};

function setUiState(state, timeoutMs = 0) {
    uiState = state;
    clearTimeout(speakTimeout);

    if (state === 'thinking') {
        statusLabel.textContent = 'THINKING...';
    } else if (state === 'nova') {
        statusLabel.textContent = 'RESPONDING...';
    } else if (state === 'user') {
        statusLabel.textContent = 'LISTENING...';
    } else if (ws.readyState === WebSocket.OPEN) {
        statusLabel.textContent = 'ONLINE';
    }

    if (timeoutMs > 0) {
        speakTimeout = setTimeout(() => setUiState('idle'), timeoutMs);
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
    setUiState('thinking', 20000);
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
let mediaStream = null;
let isRecording = false;
let permissionPending = false;

function setRecordingUi(recording) {
    micBtn.classList.toggle('recording', recording);
    if (recording) {
        statusLabel.textContent = 'LISTENING...';
    } else if (ws.readyState === WebSocket.OPEN && uiState === 'idle') {
        statusLabel.textContent = 'ONLINE';
    }
}

function getPreferredRecorderMimeType() {
    const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/ogg;codecs=opus',
    ];
    return candidates.find(type => MediaRecorder.isTypeSupported(type)) || '';
}

function startRecording() {
    if (isRecording || permissionPending) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        statusLabel.textContent = 'MIC UNSUPPORTED';
        console.warn('Microphone API unavailable in this browser.');
        return;
    }

    if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(location.hostname)) {
        statusLabel.textContent = 'USE HTTPS/LOCAL';
        console.warn('getUserMedia is blocked on insecure origins like Tailscale addresses. Use localhost or HTTPS.');
        return;
    }

    permissionPending = true;
    setRecordingUi(true);

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            mediaStream = stream;
            const mimeType = getPreferredRecorderMimeType();
            const options = mimeType ? { mimeType } : undefined;
            mediaRecorder = new MediaRecorder(stream, options);
            audioChunks = [];
            mediaRecorder.ondataavailable = (e) => {
                if (e.data && e.data.size > 0) audioChunks.push(e.data);
            };
            mediaRecorder.onstop = sendRecordedAudio;
            mediaRecorder.start();
            isRecording = true;
            permissionPending = false;
            setUiState('user');
        })
        .catch((e) => {
            console.warn('Microphone unavailable:', e);
            statusLabel.textContent = 'MIC BLOCKED';
            permissionPending = false;
            isRecording = false;
            setRecordingUi(false);
            setUiState('idle');
        });
}

function stopRecording() {
    if (permissionPending) {
        permissionPending = false;
        setRecordingUi(false);
        setUiState('idle');
        return;
    }

    if (!isRecording || !mediaRecorder) {
        setRecordingUi(false);
        setUiState('idle');
        return;
    }

    if (mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    isRecording = false;
    setRecordingUi(false);
    setUiState('thinking', 20000);
}

function handlePointerDown(event) {
    event.preventDefault();
    startRecording();
}

function handlePointerUp(event) {
    if (event) event.preventDefault();
    stopRecording();
}

micBtn.addEventListener('pointerdown', handlePointerDown);
micBtn.addEventListener('pointerup', handlePointerUp);
micBtn.addEventListener('pointerleave', handlePointerUp);
micBtn.addEventListener('pointercancel', handlePointerUp);
micBtn.addEventListener('touchstart', (event) => {
    if (event.touches && event.touches.length > 0) {
        startRecording();
    }
}, { passive: true });
micBtn.addEventListener('touchend', stopRecording, { passive: true });
micBtn.addEventListener('touchcancel', stopRecording, { passive: true });

async function sendRecordedAudio() {
    if (!audioChunks.length) return;
    const mimeType = mediaRecorder && mediaRecorder.mimeType ? mediaRecorder.mimeType : 'audio/webm';
    const blob = new Blob(audioChunks, { type: mimeType });
    const arrayBuffer = await blob.arrayBuffer();
    const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
    ws.send(JSON.stringify({ type: 'audio', audio_b64: b64, mime_type: blob.type, final: true }));
    appendMessage('You', '[Voice Message]', 'user');
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

    // Stark / Jarvis Color Palette Configurations
    let strokeColor = 'rgba(0, 240, 255, ';     // Idle: Minimal Cyan
    if (uiState === 'nova') strokeColor = 'rgba(0, 210, 255, ';       // Nova Active: Bright Cyan
    if (uiState === 'user') strokeColor = 'rgba(56, 189, 248, ';      // User Voice: Electric Arc Blue
    if (uiState === 'thinking') strokeColor = 'rgba(245, 158, 11, ';  // Thinking Mode: Stark Gold

    const thinking = uiState === 'thinking';

    // 1. Static Ambient Grid Ring
    ctx.beginPath();
    ctx.arc(cx, cy, 190, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.06)';
    ctx.setLineDash([2, 12]);
    ctx.stroke();

    // 2. Rotating Segment Ring (Outer Orbital Arc)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.beginPath();
    if (thinking) {
        ctx.setLineDash([60, 20, 10, 20]);
        ctx.arc(0, 0, 150, 0, Math.PI * 1.5);
        ctx.strokeStyle = strokeColor + '0.85)';
        ctx.lineWidth = 2.5;
    } else {
        ctx.setLineDash([40, 20, 10, 20]);
        ctx.arc(0, 0, 150, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor + '0.35)';
        ctx.lineWidth = 1.5;
    }
    ctx.stroke();
    ctx.restore();

    // 3. Counter-Rotating Inner Ring (Thinking / Processing Node)
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-angle * 1.8);
    ctx.beginPath();
    if (thinking) {
        ctx.setLineDash([120, 30]);
        ctx.arc(0, 0, 110, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor + '0.75)';
        ctx.lineWidth = 2;
    } else {
        ctx.setLineDash([90, 40]);
        ctx.arc(0, 0, 110, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor + '0.2)';
        ctx.lineWidth = 1;
    }
    ctx.stroke();
    ctx.restore();

    // 4. Dynamic Wave Core (Voice-Reactive Reactor Circle)
    ctx.beginPath();
    const points = 120;
    const baseRadius = 70;
    let amp = 2.5; // Idle subtle wave
    if (uiState === 'user') amp = 18;
    if (uiState === 'nova') amp = 26;
    if (thinking) amp = 5 + Math.sin(waveOffset * 3) * 4; // Pulsing thinking core

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
    ctx.shadowBlur = uiState === 'idle' ? 6 : 18;
    ctx.shadowColor = strokeColor + '0.8)';
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Motion Velocities
    let angleStep = 0.003;
    if (uiState === 'user' || uiState === 'nova') angleStep = 0.012;
    if (thinking) angleStep = 0.025; // Accelerate orbit while thinking
    angle += angleStep;

    let waveStep = 0.04;
    if (uiState === 'user' || uiState === 'nova') waveStep = 0.14;
    if (thinking) waveStep = 0.09;
    waveOffset += waveStep;

    requestAnimationFrame(drawHUDCore);
}

drawHUDCore();