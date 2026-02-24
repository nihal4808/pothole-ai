// Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const cameraControls = document.getElementById('camera-controls');
const uploadControls = document.getElementById('upload-controls');
const currentModeStr = "camera";

const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
const imageUpload = document.getElementById('image-upload');
const statusIndicator = document.getElementById('connection-status');
const outputFrame = document.getElementById('output-frame');

const valFps = document.getElementById('val-fps');
const valCount = document.getElementById('val-count');
const valLatency = document.getElementById('val-latency');
const valSeverity = document.getElementById('val-severity');

const videoObj = document.getElementById('webcam');
const hiddenCanvas = document.getElementById('hidden-canvas');
const ctx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

let ws = null;
let stream = null;
let captureInterval = null;
let isStreaming = false;

// Tabs Logic
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = btn.dataset.mode;

        if (mode === 'camera') {
            cameraControls.style.display = 'block';
            uploadControls.style.display = 'none';
        } else {
            cameraControls.style.display = 'none';
            uploadControls.style.display = 'block';
            stopCamera();
        }
    });
});

// Update Metrics Display
function updateMetrics(metrics) {
    valFps.textContent = metrics.fps;
    valCount.textContent = metrics.count;
    valLatency.textContent = metrics.latency_ms + 'ms';

    valSeverity.textContent = metrics.max_severity;
    valSeverity.className = 'metric-value sev-' + metrics.max_severity;
}

// REST Upload Logic
imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading
    outputFrame.style.opacity = '0.5';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze/image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.image_b64) {
            outputFrame.src = data.image_b64;
            updateMetrics(data.metrics);
        } else {
            alert(data.error || "Failed to process image.");
        }
    } catch (err) {
        console.error("Upload error", err);
        alert("Error connecting to server.");
    } finally {
        outputFrame.style.opacity = '1';
    }
});

// WebSocket Camera Logic
function getWebSocketUrl() {
    const loc = window.location;
    const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${loc.host}/ws/camera`;
}

function processFrame() {
    if (!isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

    // Draw video frame to canvas
    if (videoObj.videoWidth > 0 && videoObj.videoHeight > 0) {
        hiddenCanvas.width = videoObj.videoWidth;
        hiddenCanvas.height = videoObj.videoHeight;
        ctx.drawImage(videoObj, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

        // Get jpeg base64
        const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.7);
        // Send to server
        ws.send(dataUrl);
    }
}

async function startCamera() {
    try {
        // Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
        });
        videoObj.srcObject = stream;

        videoObj.onloadedmetadata = () => {
            // Establish WebSockets
            ws = new WebSocket(getWebSocketUrl());

            ws.onopen = () => {
                statusIndicator.textContent = "Online";
                statusIndicator.className = "status-indicator online";
                isStreaming = true;
                startCameraBtn.disabled = true;
                stopCameraBtn.disabled = false;

                // Start capture loop (request animation frame for smooth pacing? 
                // Delaying to wait for server response prevents flooding)
                processFrame();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.image_b64) {
                    outputFrame.src = data.image_b64;
                    updateMetrics(data.metrics);
                }

                // Wait for response before sending next to prevent flooding
                if (isStreaming) {
                    requestAnimationFrame(processFrame);
                }
            };

            ws.onerror = (e) => {
                console.error("WebSocket Error", e);
                stopCamera();
            };

            ws.onclose = () => {
                console.log("WebSocket Closed");
                stopCamera();
            };
        };

    } catch (err) {
        console.error("Camera access denied or error:", err);
        alert("Cannot access webcam. Ensure permissions are granted.");
    }
}

function stopCamera() {
    isStreaming = false;

    if (ws) {
        ws.close();
        ws = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoObj.srcObject = null;
        stream = null;
    }

    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;
    statusIndicator.textContent = "Offline";
    statusIndicator.className = "status-indicator offline";
}

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
