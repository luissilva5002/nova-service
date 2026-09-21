// WebSocket & System Control Client for NOVA HUD

const chatBox = document.getElementById('chat-box');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const micBtn = document.getElementById('micBtn');
const modelSelect = document.getElementById('modelSelect');
const drawer = document.getElementById('drawer');
const coreWrapper = document.getElementById('coreWrapper');
const memoryMapPanel = document.getElementById('memoryMapPanel');
const memoryMapSvg = document.getElementById('memoryMapSvg');
const memoryNoteViewer = document.getElementById('memoryNoteViewer');

let uiState = 'idle'; // 'idle' | 'user' | 'thinking' | 'nova'
let speakTimeout;
let audioFallbackTimeout;
let thinkingStartedAt = 0;
let memoryGraphFocus = null;
let liveNovaMessage = null;
let audioContext = null;
let nextAudioStart = 0;
const MIN_THINKING_MS = 700; // spinner stays visible at least this long, even on instant replies

const SESSION_STORAGE_KEY = 'nova_session_id';

function getOrCreateSessionId() {
    try {
        const existing = localStorage.getItem(SESSION_STORAGE_KEY);
        if (existing) return existing;
        const id = (window.crypto && typeof window.crypto.randomUUID === 'function')
            ? window.crypto.randomUUID()
            : `nova-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        localStorage.setItem(SESSION_STORAGE_KEY, id);
        return id;
    } catch (e) {
        console.warn('Falling back to transient session id because localStorage is unavailable.', e);
        return `nova-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
}

const sessionId = getOrCreateSessionId();
const wsProtocol = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${wsProtocol}://${location.host}/ws/chat?session_id=${encodeURIComponent(sessionId)}`);

ws.onopen = () => {
    statusDot.classList.add('online');
    statusLabel.textContent = 'ONLINE';
    ws.send(JSON.stringify({ type: 'session', session_id: sessionId }));
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

    if (payload.type === 'text_delta') {
        if (!liveNovaMessage) liveNovaMessage = appendMessage('NOVA', '', 'nova');
        liveNovaMessage.innerText += payload.text || '';
    } else if (payload.type === 'text') {
        if (liveNovaMessage) {
            liveNovaMessage.innerText = `NOVA: ${payload.text || ''}`;
            liveNovaMessage = null;
        } else {
            appendMessage('NOVA', payload.text, 'nova');
        }
        // Stay in 'thinking' - audio hasn't started yet, TTS is still
        // synthesizing. If no audio message follows shortly (e.g. TTS
        // stub mode / disabled, or an open_url action with no TTS),
        // fall back so we don't get stuck.
        clearTimeout(audioFallbackTimeout);
        audioFallbackTimeout = setTimeout(() => {
            if (uiState === 'thinking') setUiState('nova', 4000);
        }, 4000);
    } else if (payload.type === 'metrics' && payload.llm) {
        showLlmSpeed(payload.llm);
        // Still thinking/streaming - leave the spinner running.
    } else if (payload.type === 'open_url' && payload.url) {
        if (payload.text) appendMessage('NOVA', payload.text, 'nova');
        // Slight delay so the message above renders before navigation.
        // Using location.href (not window.open) so mobile OSes can route
        // this to an installed app via universal/app links, falling back
        // to the browser automatically if no app claims the link.
        setTimeout(() => { window.location.href = payload.url; }, 300);
    } else if (payload.type === 'audio_chunk' && payload.audio_b64) {
        playAudioChunk(payload.audio_b64, payload.sample_rate || 22050);
    } else if (payload.type === 'audio' && payload.audio_b64) {
        clearTimeout(audioFallbackTimeout); // audio arrived, cancel the fallback
        playAudioBase64(payload.audio_b64);
    }
};

function setUiState(state, timeoutMs = 0) {
    // Enforce a minimum visible duration for 'thinking' before allowing a
    // transition away from it - otherwise a fast model can make the
    // spinner flash for a single frame.
    if (uiState === 'thinking' && state !== 'thinking') {
        const elapsed = Date.now() - thinkingStartedAt;
        if (elapsed < MIN_THINKING_MS) {
            setTimeout(() => setUiState(state, timeoutMs), MIN_THINKING_MS - elapsed);
            return;
        }
    }

    uiState = state;
    clearTimeout(speakTimeout);

    // Toggle the CSS-driven spinner rings via a class instead of per-frame
    // JS. These animate on the compositor thread, so they keep spinning
    // even if the main JS thread (and canvas rAF loop) stalls - which it
    // will while the host CPU is pinned at 100% during LLM inference.
    coreWrapper.classList.toggle('is-thinking', state === 'thinking');

    if (state === 'thinking') {
        thinkingStartedAt = Date.now();
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

async function toggleMemoryMap() {
    if (!memoryMapPanel) return;
    const hidden = memoryMapPanel.classList.contains('hidden');
    if (hidden) {
        memoryMapPanel.classList.remove('hidden');
        await renderMemoryMap();
    } else {
        memoryMapPanel.classList.add('hidden');
    }
}

async function renderMemoryMap() {
    if (!memoryMapSvg) return;
    try {
        const response = await fetch('/api/memory/graph');
        if (!response.ok) throw new Error(`Memory graph fetch failed (${response.status})`);
        const graph = await response.json();
        memoryGraphFocus = null;
        renderMemoryGraph(graph);
    } catch (error) {
        console.warn('Could not load memory map', error);
        memoryMapSvg.innerHTML = '<text x="500" y="360" text-anchor="middle" fill="#e2e8f0" font-size="18">Memory map unavailable</text>';
    }
}

function renderMemoryGraph(graph) {
    const svgNS = 'http://www.w3.org/2000/svg';
    memoryMapSvg.innerHTML = '';

    const defs = document.createElementNS(svgNS, 'defs');
    const filter = document.createElementNS(svgNS, 'filter');
    filter.setAttribute('id', 'mapGlow');
    filter.innerHTML = '<feGaussianBlur stdDeviation="2.5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>';
    defs.appendChild(filter);
    memoryMapSvg.appendChild(defs);

    let nodes = graph.nodes || [];
    const links = graph.links || [];
    const nodeMap = new Map(nodes.map(node => [node.id, node]));

    if (memoryGraphFocus) {
        const focusNode = nodeMap.get(memoryGraphFocus);
        if (focusNode && focusNode.kind === 'folder') {
            const visibleIds = new Set([focusNode.id]);
            const queue = [focusNode.id];
            while (queue.length) {
                const currentId = queue.shift();
                for (const link of links) {
                    if (link.source === currentId && link.kind !== 'reference') {
                        visibleIds.add(link.target);
                        queue.push(link.target);
                    }
                }
            }
            nodes = nodes.filter(node => visibleIds.has(node.id));
        }
    }

    for (const link of links) {
        const source = nodeMap.get(link.source);
        const target = nodeMap.get(link.target);
        if (!source || !target) continue;
        if (memoryGraphFocus && !nodes.some(node => node.id === source.id || node.id === target.id)) continue;

        const line = document.createElementNS(svgNS, 'line');
        line.setAttribute('x1', source.x);
        line.setAttribute('y1', source.y);
        line.setAttribute('x2', target.x);
        line.setAttribute('y2', target.y);
        line.setAttribute('stroke', link.kind === 'reference' ? 'rgba(96,165,250,0.6)' : 'rgba(125,211,252,0.45)');
        line.setAttribute('stroke-width', link.kind === 'reference' ? '1.8' : '1.4');
        line.setAttribute('stroke-linecap', 'round');
        line.setAttribute('filter', 'url(#mapGlow)');
        memoryMapSvg.appendChild(line);

        const label = document.createElementNS(svgNS, 'text');
        label.setAttribute('x', (source.x + target.x) / 2);
        label.setAttribute('y', (source.y + target.y) / 2 - 8);
        label.setAttribute('fill', '#a5b4fc');
        label.setAttribute('font-size', '10');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = link.label || '';
        memoryMapSvg.appendChild(label);
    }

    for (const node of nodes) {
        const group = document.createElementNS(svgNS, 'g');
        group.setAttribute('transform', `translate(${node.x}, ${node.y})`);
        group.style.cursor = node.kind === 'folder' ? 'pointer' : node.markdown_id ? 'pointer' : 'default';
        group.addEventListener('click', () => {
            if (node.kind === 'folder') {
                memoryGraphFocus = node.id;
                renderMemoryGraph(graph);
                return;
            }
            if (node.markdown_id) openMemoryNode(node);
        });

        const width = node.kind === 'folder' ? 160 : Math.max(120, Math.min(180, 80 + (node.label.length * 6.5)));
        const height = node.kind === 'folder' ? 68 : Math.max(56, 48 + Math.min((node.facts || []).length * 12, 32));

        const box = document.createElementNS(svgNS, 'rect');
        box.setAttribute('x', -(width / 2));
        box.setAttribute('y', -(height / 2));
        box.setAttribute('width', width);
        box.setAttribute('height', height);
        box.setAttribute('rx', '16');
        box.setAttribute('fill', node.kind === 'folder' ? 'rgba(59, 130, 246, 0.18)' : 'rgba(14, 16, 28, 0.9)');
        box.setAttribute('stroke', node.kind === 'folder' ? 'rgba(125, 211, 252, 0.9)' : 'rgba(96,165,250,0.9)');
        box.setAttribute('stroke-width', '1.4');
        box.setAttribute('filter', 'url(#mapGlow)');
        group.appendChild(box);

        const title = document.createElementNS(svgNS, 'text');
        title.setAttribute('x', '0');
        title.setAttribute('y', node.kind === 'folder' ? '-10' : '-6');
        title.setAttribute('text-anchor', 'middle');
        title.setAttribute('fill', '#e2e8f0');
        title.setAttribute('font-size', node.kind === 'folder' ? '12' : '13');
        title.setAttribute('font-weight', '700');
        title.textContent = node.label.length > 18 ? `${node.label.slice(0, 17)}…` : node.label;
        group.appendChild(title);

        const type = document.createElementNS(svgNS, 'text');
        type.setAttribute('x', '0');
        type.setAttribute('y', node.kind === 'folder' ? '12' : '12');
        type.setAttribute('text-anchor', 'middle');
        type.setAttribute('fill', '#7dd3fc');
        type.setAttribute('font-size', '10');
        type.textContent = node.kind === 'folder' ? 'container' : (node.type || 'concept');
        group.appendChild(type);

        memoryMapSvg.appendChild(group);
    }
}

function escapeHtml(value = '') {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function convertMarkdownInline(text = '') {
    let html = escapeHtml(text);
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return html;
}

function renderMarkdownContent(markdown = '') {
    const lines = (markdown || '').replace(/\r\n/g, '\n').split('\n');
    const blocks = [];
    let listItems = [];
    let paragraph = [];
    let codeBlock = [];
    let inCode = false;

    const flushParagraph = () => {
        if (paragraph.length) {
            blocks.push(`<p>${convertMarkdownInline(paragraph.join(' ').trim())}</p>`);
            paragraph = [];
        }
    };

    const flushList = () => {
        if (listItems.length) {
            blocks.push(`<ul>${listItems.map(item => `<li>${convertMarkdownInline(item)}</li>`).join('')}</ul>`);
            listItems = [];
        }
    };

    const flushCode = () => {
        if (codeBlock.length) {
            blocks.push(`<pre><code>${escapeHtml(codeBlock.join('\n'))}</code></pre>`);
            codeBlock = [];
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line.trim()) {
            flushParagraph();
            flushList();
            continue;
        }

        if (line.startsWith('```')) {
            flushParagraph();
            flushList();
            if (inCode) {
                flushCode();
                inCode = false;
            } else {
                inCode = true;
            }
            continue;
        }

        if (inCode) {
            codeBlock.push(rawLine);
            continue;
        }

        const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
            flushParagraph();
            flushList();
            const level = Math.min(headingMatch[1].length, 6);
            blocks.push(`<h${level}>${convertMarkdownInline(headingMatch[2])}</h${level}>`);
            continue;
        }

        if (/^[-*]\s+/.test(line)) {
            flushParagraph();
            listItems.push(line.replace(/^[-*]\s+/, '').trim());
            continue;
        }

        if (/^>\s+/.test(line)) {
            flushParagraph();
            flushList();
            blocks.push(`<blockquote>${convertMarkdownInline(line.replace(/^>\s+/, ''))}</blockquote>`);
            continue;
        }

        paragraph.push(line);
    }

    flushParagraph();
    flushList();
    flushCode();
    return blocks.join('') || '<p>No content available.</p>';
}

async function openMemoryNode(node) {
    if (!memoryNoteViewer || !node || !node.markdown_id) return;
    memoryNoteViewer.classList.remove('hidden');
    memoryNoteViewer.innerHTML = `
        <div class="memory-note-header">
            <div>
                <div class="memory-note-kicker">MARKDOWN NODE</div>
                <h3>${escapeHtml(node.label || 'Document')}</h3>
            </div>
            <button class="hud-btn small" onclick="closeMemoryNote()">Close</button>
        </div>
        <div class="memory-note-body loading">Loading document…</div>
    `;

    try {
        const response = await fetch(`/api/memory/markdown/${encodeURIComponent(node.markdown_id)}`);
        if (!response.ok) throw new Error(`Markdown fetch failed (${response.status})`);
        const documentData = await response.json();
        const updated = documentData.updated_at ? new Date(documentData.updated_at).toLocaleString() : '';
        memoryNoteViewer.innerHTML = `
            <div class="memory-note-header">
                <div>
                    <div class="memory-note-kicker">MARKDOWN NODE</div>
                    <h3>${escapeHtml(documentData.title || node.label || 'Document')}</h3>
                    ${updated ? `<div class="memory-note-meta">Updated ${escapeHtml(updated)}</div>` : ''}
                </div>
                <button class="hud-btn small" onclick="closeMemoryNote()">Close</button>
            </div>
            <article class="memory-note-body">${renderMarkdownContent(documentData.content || '')}</article>
        `;
    } catch (error) {
        console.warn('Could not load markdown note', error);
        memoryNoteViewer.innerHTML = `
            <div class="memory-note-header">
                <div>
                    <div class="memory-note-kicker">MARKDOWN NODE</div>
                    <h3>${escapeHtml(node.label || 'Document')}</h3>
                </div>
                <button class="hud-btn small" onclick="closeMemoryNote()">Close</button>
            </div>
            <div class="memory-note-body error">Unable to load the document.</div>
        `;
    }
}

function closeMemoryNote() {
    if (!memoryNoteViewer) return;
    memoryNoteViewer.classList.add('hidden');
    memoryNoteViewer.innerHTML = '';
}

function sendMsg() {
    const input = document.getElementById('userInput');
    if (!input.value) return;
    appendMessage('You', input.value, 'user');
    ws.send(JSON.stringify({ type: 'text', text: input.value, stream: true }));
    input.value = '';
    // Safety net only - the real exit from 'thinking' happens when audio
    // actually starts playing (see playAudioBase64), not here.
    setUiState('thinking', 20000);
}

function showLlmSpeed(metrics) {
    const speed = Number(metrics.tokens_per_second || 0);
    const tokenCount = Number(metrics.completion_tokens || 0);
    const ttft = Number(metrics.ttft_seconds || 0);
    document.getElementById('statTokensPerSecond').innerText =
        tokenCount > 0 ? `${speed.toFixed(1)} tok/s (TTFT ${ttft.toFixed(2)}s)` : '0.0 tok/s';
}

function appendMessage(sender, text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${className}`;
    msgDiv.innerText = `${sender}: ${text}`;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msgDiv;
}

function playAudioChunk(b64, sampleRate) {
    audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const samples = new Int16Array(bytes.buffer);
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = samples[i] / 32768;
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    nextAudioStart = Math.max(nextAudioStart, audioContext.currentTime);
    source.start(nextAudioStart);
    nextAudioStart += buffer.duration;
    setUiState('nova');
}

function playAudioBase64(b64) {
    const pcmBytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const wavBuffer = pcmToWav(pcmBytes, 22050);
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    const audio = new Audio(URL.createObjectURL(blob));

    // Audio is actually about to play now - this is the real transition
    // out of 'thinking' into 'nova' (speaking).
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
    // Safety net only - the real exit from 'thinking' happens when audio
    // actually starts playing (see playAudioBase64), not here.
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
    // uiState is already 'thinking' (set by stopRecording) - leave it running.
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

    let strokeColor = 'rgba(88, 80, 236, ';   // Idle Violet/Indigo
    if (uiState === 'nova') strokeColor = 'rgba(0, 210, 255, ';       // NOVA Cyan/Blue
    if (uiState === 'user') strokeColor = 'rgba(99, 102, 241, ';      // User Violet
    if (uiState === 'thinking') strokeColor = 'rgba(245, 158, 11, ';  // Processing Amber

    const thinking = uiState === 'thinking';

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
    if (thinking) {
        ctx.setLineDash([]);
        ctx.arc(0, 0, 150, 0, Math.PI * 0.55);
        ctx.strokeStyle = strokeColor + '0.9)';
        ctx.lineWidth = 3;
    } else {
        ctx.setLineDash([40, 20, 10, 20]);
        ctx.arc(0, 0, 150, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor + '0.4)';
        ctx.lineWidth = 1.5;
    }
    ctx.stroke();
    ctx.restore();

    // 3. Counter-Rotating Inner Ring
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-angle * 1.6);
    ctx.beginPath();
    if (thinking) {
        ctx.setLineDash([]);
        ctx.arc(0, 0, 110, 0, Math.PI * 0.35);
        ctx.strokeStyle = strokeColor + '0.65)';
        ctx.lineWidth = 2;
    } else {
        ctx.setLineDash([90, 40]);
        ctx.arc(0, 0, 110, 0, Math.PI * 2);
        ctx.strokeStyle = strokeColor + '0.25)';
        ctx.lineWidth = 1;
    }
    ctx.stroke();
    ctx.restore();

    // 4. Dynamic Audio Wave Core
    ctx.beginPath();
    const points = 120;
    const baseRadius = 70;
    let amp = 3; // idle
    if (uiState === 'user') amp = 16;
    if (uiState === 'nova') amp = 28;
    if (thinking) amp = 6 + Math.sin(waveOffset * 2) * 3;

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

    let angleStep = 0.003;
    if (uiState === 'user' || uiState === 'nova') angleStep = 0.012;
    if (thinking) angleStep = 0.02;
    angle += angleStep;

    let waveStep = 0.04;
    if (uiState === 'user' || uiState === 'nova') waveStep = 0.15;
    if (thinking) waveStep = 0.08;
    waveOffset += waveStep;

    requestAnimationFrame(drawHUDCore);
}

drawHUDCore();