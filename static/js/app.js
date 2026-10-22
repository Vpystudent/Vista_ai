// VISTA Live Drone Target Acquisition Controller
// Real-time Camera Stream + Manual Target Filtering + Square Bounding Boxes with Confidence %

let activeTargets = ['person', 'cell phone']; // Initial manual targets
let latestDetections = [];
let isStreaming = false;
let socket = null;
let videoStream = null;
let currentFacingMode = 'user';
let lastInferenceTime = 0;
let isProcessingFrame = false;

// DOM Elements
const video = document.getElementById('webcam-video');
const canvas = document.getElementById('hud-canvas');
const ctx = canvas.getContext('2d');
const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d');

const targetInput = document.getElementById('target-input');
const addTargetBtn = document.getElementById('add-target-btn');
const activeTargetsList = document.getElementById('active-targets-list');
const targetModeBadge = document.getElementById('target-mode-badge');

const confSlider = document.getElementById('conf-slider');
const confVal = document.getElementById('conf-val');

const statFps = document.getElementById('stat-fps');
const statLatency = document.getElementById('stat-latency');
const statTargetsCount = document.getElementById('stat-targets-count');
const detectedCounterBadge = document.getElementById('detected-counter-badge');
const liveTargetsList = document.getElementById('live-targets-list');
const camStatusPill = document.getElementById('cam-status-pill');

document.addEventListener('DOMContentLoaded', () => {
  renderTargetChips();
  setupEventListeners();
  initWebSocket();
});

// 1. Initialize WebSocket Connection
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;
  
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('[VISTA WS] Connected to detection engine.');
  };

  socket.onmessage = (event) => {
    isProcessingFrame = false;
    try {
      const data = JSON.parse(event.data);
      if (data.success) {
        latestDetections = data.detections || [];
        updateTelemetry(data);
        renderTargetFeed(latestDetections);
      }
    } catch (e) {
      console.error('[VISTA WS Error]:', e);
    }
  };

  socket.onerror = (err) => {
    console.warn('[VISTA WS Warning]: WebSocket error, fallback active.');
    isProcessingFrame = false;
  };

  socket.onclose = () => {
    console.log('[VISTA WS] Disconnected. Reconnecting in 3s...');
    setTimeout(initWebSocket, 3000);
  };
}

// 2. Setup Event Listeners
function setupEventListeners() {
  // Connect / Toggle Camera
  document.getElementById('connect-camera-btn').addEventListener('click', startCamera);
  document.getElementById('toggle-feed-btn').addEventListener('click', toggleCamera);
  document.getElementById('switch-camera-btn').addEventListener('click', switchCameraDevice);

  // Manual Target Input (Enter key or Add button)
  addTargetBtn.addEventListener('click', handleAddManualTarget);
  targetInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleAddManualTarget();
  });

  // Confidence Slider
  confSlider.addEventListener('input', (e) => {
    confVal.innerText = `${e.target.value}%`;
  });
}

// 3. Camera Controls (HTML5 WebRTC)
async function startCamera() {
  try {
    const constraints = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: currentFacingMode
      },
      audio: false
    };

    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = videoStream;

    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      offscreenCanvas.width = 640;
      offscreenCanvas.height = Math.round((canvas.height / canvas.width) * 640);

      document.getElementById('camera-prompt-overlay').classList.add('hidden');
      isStreaming = true;
      updateFeedButtonState(true);

      camStatusPill.className = 'text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-800 mono flex items-center space-x-1.5 font-medium whitespace-nowrap';
      camStatusPill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span><span>LIVE FEED ACTIVE</span>';

      // Start render & inference loops
      requestAnimationFrame(renderLoop);
    };

  } catch (err) {
    alert('Camera access denied or unavailable: ' + err.message + '\nPlease allow camera permission in your browser.');
    console.error('Camera error:', err);
  }
}

function stopCamera() {
  if (videoStream) {
    videoStream.getTracks().forEach(t => t.stop());
    videoStream = null;
  }
  isStreaming = false;
  latestDetections = [];
  updateFeedButtonState(false);

  camStatusPill.className = 'text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 rounded-full bg-amber-950 text-amber-300 border border-amber-800 mono flex items-center space-x-1.5 font-medium whitespace-nowrap';
  camStatusPill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse"></span><span>STANDBY</span>';

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('camera-prompt-overlay').classList.remove('hidden');
}

function toggleCamera() {
  if (isStreaming) {
    stopCamera();
  } else {
    startCamera();
  }
}

async function switchCameraDevice() {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
  if (isStreaming) {
    stopCamera();
    await startCamera();
  }
}

function updateFeedButtonState(active) {
  const label = document.getElementById('toggle-feed-label');
  const btn = document.getElementById('toggle-feed-btn');
  if (active) {
    label.innerText = 'Stop Feed';
    btn.className = 'px-4 py-2 bg-rose-500/90 hover:bg-rose-500 text-white font-bold text-xs rounded-xl transition-all flex items-center space-x-2';
  } else {
    label.innerText = 'Start Feed';
    btn.className = 'px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-xs rounded-xl transition-all flex items-center space-x-2';
  }
}

// 4. Manual Target Filtering Management
function handleAddManualTarget() {
  const val = targetInput.value.trim().toLowerCase();
  if (!val) return;

  const parts = val.split(',').map(s => s.trim()).filter(Boolean);
  parts.forEach(p => {
    if (!activeTargets.includes(p)) {
      activeTargets.push(p);
    }
  });

  targetInput.value = '';
  renderTargetChips();
}

function addTargetPreset(targetName) {
  if (!activeTargets.includes(targetName)) {
    activeTargets.push(targetName);
    renderTargetChips();
  }
}

function removeTarget(targetName) {
  activeTargets = activeTargets.filter(t => t !== targetName);
  renderTargetChips();
}

function clearAllTargets() {
  activeTargets = [];
  renderTargetChips();
}

function renderTargetChips() {
  if (activeTargets.length === 0) {
    activeTargetsList.innerHTML = `
      <span class="text-[11px] text-amber-400/90 italic py-0.5 px-2">
        <i class="fa-solid fa-triangle-exclamation mr-1"></i> No manual targets selected (Detecting ALL objects)
      </span>
    `;
    targetModeBadge.innerText = 'TARGET FILTER: DETECT ALL';
    targetModeBadge.className = 'bg-slate-950 px-2.5 sm:px-3 py-1 rounded-lg border border-amber-500/50 text-[10px] sm:text-xs text-amber-300 mono font-medium text-center truncate max-w-[210px] sm:max-w-none';
    return;
  }

  targetModeBadge.innerText = `TARGET FILTER: MANUAL (${activeTargets.length})`;
  targetModeBadge.className = 'bg-slate-950 px-2.5 sm:px-3 py-1 rounded-lg border border-cyan-500/50 text-[10px] sm:text-xs text-cyan-300 mono font-medium text-center truncate max-w-[210px] sm:max-w-none';

  activeTargetsList.innerHTML = activeTargets.map(t => `
    <span class="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-850 bg-slate-800/90 text-slate-200 border border-slate-700 text-xs font-medium">
      <span class="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
      <span class="capitalize">${t}</span>
      <button onclick="removeTarget('${t}')" class="text-slate-400 hover:text-white transition-colors ml-1">
        <i class="fa-solid fa-xmark text-[11px]"></i>
      </button>
    </span>
  `).join('');
}

// 5. Main Canvas Render Loop (Draws Video + Squares + Confidence %)
function renderLoop() {
  if (!isStreaming) return;

  // 1. Draw live camera frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // 2. Draw tactical squares & confidence percentages for detected targets
  if (latestDetections.length > 0) {
    const scaleX = canvas.width / (offscreenCanvas.width || 640);
    const scaleY = canvas.height / (offscreenCanvas.height || 360);

    latestDetections.forEach((d) => {
      const [origX1, origY1, origX2, origY2] = d.box;
      const x1 = origX1 * scaleX;
      const y1 = origY1 * scaleY;
      const x2 = origX2 * scaleX;
      const y2 = origY2 * scaleY;
      const w = x2 - x1;
      const h = y2 - y1;

      // Draw Tactical Square / Bounding Box
      drawTargetSquare(x1, y1, w, h, d.class_name, d.confidence_pct);
    });
  }

  // 3. Trigger frame inference if ready (throttled to ~15 FPS to ensure responsiveness)
  const now = performance.now();
  if (!isProcessingFrame && (now - lastInferenceTime > 70)) {
    sendFrameForInference();
    lastInferenceTime = now;
  }

  requestAnimationFrame(renderLoop);
}

// Draw crisp tactical square with confidence percentage
function drawTargetSquare(x, y, w, h, className, confPct) {
  const color = '#00f0ff'; // Neon Tactical Cyan

  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.shadowColor = 'rgba(0, 240, 255, 0.5)';
  ctx.shadowBlur = 8;

  // 1. Draw outer square box
  ctx.strokeRect(x, y, w, h);

  // 2. Draw tactical corner brackets
  const bracketLen = Math.min(18, Math.max(6, w * 0.25));
  ctx.lineWidth = 3;
  ctx.beginPath();
  // Top-Left
  ctx.moveTo(x, y + bracketLen); ctx.lineTo(x, y); ctx.lineTo(x + bracketLen, y);
  // Top-Right
  ctx.moveTo(x + w - bracketLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bracketLen);
  // Bottom-Left
  ctx.moveTo(x, y + h - bracketLen); ctx.lineTo(x, y + h); ctx.lineTo(x + bracketLen, y + h);
  // Bottom-Right
  ctx.moveTo(x + w - bracketLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bracketLen);
  ctx.stroke();

  // 3. Center crosshair reticle inside the target square
  const cx = x + w / 2;
  const cy = y + h / 2;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
  ctx.beginPath();
  ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
  ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
  ctx.stroke();

  // 4. Prominent Tactical Label: Class Name + Confidence Percentage
  const labelText = `🎯 ${className.toUpperCase()} : ${confPct}%`;
  ctx.font = 'bold 13px "JetBrains Mono", monospace';
  const textWidth = ctx.measureText(labelText).width;
  const tagH = 22;
  const tagY = y - tagH - 4 > 0 ? y - tagH - 4 : y + 4;

  // Label dark backdrop
  ctx.fillStyle = 'rgba(3, 7, 18, 0.9)';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.fillRect(x, tagY, textWidth + 14, tagH);
  ctx.strokeRect(x, tagY, textWidth + 14, tagH);

  // Label text
  ctx.fillStyle = '#00f0ff';
  ctx.fillText(labelText, x + 7, tagY + 16);

  ctx.restore();
}

// 6. Send Frame to Backend YOLO11 Engine
function sendFrameForInference() {
  if (!isStreaming) return;

  isProcessingFrame = true;

  // Render to offscreen canvas for high-speed lightweight transmission
  offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
  const dataUrl = offscreenCanvas.toDataURL('image/jpeg', 0.65);

  const payload = {
    image: dataUrl,
    targets: activeTargets,
    conf: parseFloat(confSlider.value) / 100.0,
    iou: 0.45
  };

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  } else {
    // HTTP Fallback
    fetch('/api/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
      isProcessingFrame = false;
      if (data.success) {
        latestDetections = data.detections || [];
        updateTelemetry(data);
        renderTargetFeed(latestDetections);
      }
    })
    .catch(() => { isProcessingFrame = false; });
  }
}

// 7. Telemetry & Target Cards Feed Updates
function updateTelemetry(data) {
  statFps.innerText = data.fps ? data.fps.toFixed(1) : '30.0';
  statLatency.innerText = `${data.latency_ms || 25} ms`;
  statTargetsCount.innerText = latestDetections.length;
  detectedCounterBadge.innerText = `${latestDetections.length} Found`;
}

function renderTargetFeed(detections) {
  if (detections.length === 0) {
    liveTargetsList.innerHTML = `
      <div class="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500 text-xs">
        <i class="fa-solid fa-crosshairs text-2xl mb-2 text-slate-600"></i>
        <p>No objects matching "${activeTargets.join(', ') || 'ALL'}" detected above ${confSlider.value}% confidence.</p>
      </div>
    `;
    return;
  }

  liveTargetsList.innerHTML = detections.map(d => `
    <div class="p-2.5 bg-[#0d131f] border border-slate-800 hover:border-slate-700 rounded-xl flex items-center justify-between transition-all">
      <div class="flex items-center space-x-2.5">
        <div class="w-7 h-7 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center text-cyan-400 font-bold text-xs mono">
          #${d.id}
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <span class="text-xs font-semibold text-white capitalize">${d.class_name}</span>
            <span class="text-[10px] mono text-emerald-400 bg-emerald-950/70 px-1.5 py-0.5 rounded border border-emerald-800/60 font-medium">
              ${d.confidence_pct}%
            </span>
          </div>
          <p class="text-[10px] mono text-slate-500 mt-0.5">${d.width}x${d.height}px</p>
        </div>
      </div>
      
      <!-- Verification Feedback Actions -->
      <div id="verify-box-${d.id}" class="flex items-center space-x-1">
        <button onclick="recordFeedback(${d.id}, '${d.class_name}', ${d.confidence}, 'verified')" title="Confirm Object Found" class="p-1.5 rounded-lg bg-slate-800 hover:bg-emerald-600 hover:text-black text-slate-300 hover:border-emerald-600 border border-slate-700 text-xs transition-all">
          <i class="fa-solid fa-check text-[10px]"></i>
        </button>
        <button onclick="recordFeedback(${d.id}, '${d.class_name}', ${d.confidence}, 'flagged')" title="Flag False Alarm" class="p-1.5 rounded-lg bg-slate-800 hover:bg-rose-600 hover:text-white text-slate-300 hover:border-rose-600 border border-slate-700 text-xs transition-all">
          <i class="fa-solid fa-xmark text-[10px]"></i>
        </button>
      </div>
    </div>
  `).join('');
}

// Human-in-the-loop feedback
async function recordFeedback(id, className, confidence, status) {
  try {
    await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_id: id,
        class_name: className,
        confidence: confidence,
        status: status
      })
    });

    const box = document.getElementById(`verify-box-${id}`);
    if (box) {
      box.innerHTML = status === 'verified'
        ? `<span class="text-[10px] mono text-emerald-400 bg-emerald-950 px-1.5 py-0.5 rounded border border-emerald-800">Confirmed</span>`
        : `<span class="text-[10px] mono text-rose-400 bg-rose-950 px-1.5 py-0.5 rounded border border-rose-800">Flagged</span>`;
    }
  } catch (e) {
    console.error('Feedback failed:', e);
  }
}
