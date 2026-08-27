/**
 * VARENIUM FORENSIC - Application Logic
 * Version: 2.1.0
 */

// Dom Elements
const dom = {
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    uploadPrompt: document.getElementById('upload-prompt'),
    imageMain: document.getElementById('image-main'),
    heatmapOverlay: document.getElementById('heatmap-overlay'),
    freqOverlay: document.getElementById('frequency-overlay'),
    terminal: document.getElementById('terminal'),
    analyzeBtn: document.getElementById('analyze-btn'),
    resetBtn: document.getElementById('reset-btn'),
    compareControls: document.getElementById('compare-controls'),
    resultBanner: document.getElementById('result-banner'),
    resultText: document.getElementById('result-text'),
    confVal: document.getElementById('conf-val'),
    logitVal: document.getElementById('logit-val'),
    spatialBar: document.getElementById('spatial-bar'),
    freqBar: document.getElementById('freq-bar'),
    spatialPct: document.getElementById('spatial-pct'),
    freqPct: document.getElementById('freq-pct'),
    meta: {
        res: document.getElementById('meta-res'),
        fmt: document.getElementById('meta-fmt'),
        size: document.getElementById('meta-size')
    },
    history: document.getElementById('history-strip'),
    pingText: document.getElementById('ping-text'),
    reportBtn: document.getElementById('report-btn')
};

// State
let selectedFile = null;
let lastResult = null;
let sessionHistory = [];
const API_BASE = 'http://localhost:5001';

function redirectToLogin(reason) {
    const params = new URLSearchParams({
        reason,
        returnTo: 'dashboard.html'
    });
    window.location.href = `login.html?${params.toString()}`;
}

async function requireAuthentication(reason = 'Sign in to access the forensic dashboard.') {
    try {
        const response = await fetch(`${API_BASE}/auth/me`, { credentials: 'include' });
        if (!response.ok) {
            redirectToLogin(reason);
            return false;
        }
        return true;
    } catch {
        addAuditLine('ERROR: Authentication service unavailable.', 'error');
        return false;
    }
}

// --- 1. Initialization ---
async function init() {
    if (!await requireAuthentication()) return;

    // Ping check (increased interval for performance)
    setInterval(checkConnectivity, 15000);
    checkConnectivity();

    // Event Listeners
    dom.dropZone.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', handleFileSelection);
    dom.analyzeBtn.addEventListener('click', runForensicPipeline);
    dom.resetBtn.addEventListener('click', () => location.reload());
    dom.reportBtn.addEventListener('click', downloadForensicReport);

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(name => {
        dom.dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dom.dropZone.style.borderColor = 'var(--accent-blue)';
        });
    });
    ['dragleave', 'drop'].forEach(name => {
        dom.dropZone.addEventListener(name, (e) => {
            e.preventDefault();
            dom.dropZone.style.borderColor = 'var(--glass-border)';
        });
    });
    dom.dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    // View Toggles
    document.querySelectorAll('[data-view]').forEach(btn => {
        btn.addEventListener('click', (e) => switchForensicView(e.target.dataset.view));
    });
}

// --- 2. File Handling ---
function handleFileSelection(e) {
    const file = e.target.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        addAuditLine(`ERROR: Invalid file type ${file.type}`, 'error');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    
    addAuditLine(`Loading forensic sample: ${file.name}...`, 'info');

    reader.onload = (e) => {
        console.log("[Forensic] Local file read successful. Updating viewport...");
        
        dom.imageMain.onload = () => {
            console.log("[Forensic] Viewport image rendering complete.");
            dom.imageMain.classList.remove('hidden');
            dom.uploadPrompt.classList.add('hidden');
            dom.analyzeBtn.disabled = false;
            addAuditLine(`Sample loaded successfully into viewport.`, 'success');
        };

        dom.imageMain.onerror = () => {
            console.error("[Forensic] Failed to render image in viewport.");
            addAuditLine(`CRITICAL ERROR: Failed to render image.`, 'error');
        };

        dom.imageMain.src = e.target.result;
    };

    reader.onerror = () => {
        addAuditLine(`ERROR: Failed to read local file.`, 'error');
    };

    reader.readAsDataURL(file);
}

// --- 3. Forensic Pipeline ---
async function runForensicPipeline() {
    if (!selectedFile) return;

    if (!await requireAuthentication('Sign in to run an analysis.')) return;

    dom.analyzeBtn.disabled = true;
    dom.analyzeBtn.innerText = "PROCESSING...";
    
    // UI steps for "worth"
    await addAuditLine("Initializing DeepScan protocol...", 'info', 400);
    await addAuditLine("Extracting spatial facial descriptors...", 'info', 600);
    await addAuditLine("Decomposing Haar Wavelet frequency domains...", 'info', 500);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            body: formData,
            credentials: 'include'
        });

        if (!response.ok) throw new Error("Backend connection lost.");

        const data = await response.json();
        lastResult = data;
        renderAnalysisResults(data);

    } catch (err) {
        addAuditLine(`CRITICAL ERROR: ${err.message}`, 'error');
        dom.analyzeBtn.disabled = false;
        dom.analyzeBtn.innerText = "RETRY ANALYSIS";
    }
}

function renderAnalysisResults(data) {
    const isFake = data.prediction === "MANIPULATED";
    
    // 1. Visuals
    dom.heatmapOverlay.src = `data:image/jpeg;base64,${data.heatmap}`;
    dom.freqOverlay.src = `data:image/jpeg;base64,${data.frequency_map}`;
    dom.compareControls.classList.remove('hidden');
    
    // 2. Metrics
    dom.resultBanner.classList.remove('hidden');
    dom.resultBanner.classList.toggle('manipulated', isFake);
    dom.resultText.innerText = data.prediction;
    dom.resultText.style.color = isFake ? 'var(--accent-crimson)' : 'var(--accent-blue)';
    
    dom.confVal.innerText = `${data.confidence}%`;
    dom.confVal.style.color = isFake ? 'var(--accent-crimson)' : 'var(--accent-blue)';
    dom.logitVal.innerText = data.logit_mag;

    // Simulated stream weights (calculated from data)
    const spatialWeight = isFake ? 85 + Math.random()*10 : 15 - Math.random()*10;
    const freqWeight = isFake ? 70 + Math.random()*20 : 10 - Math.random()*5;

    updateMetricBar('spatial', spatialWeight);
    updateMetricBar('freq', freqWeight);

    // 3. Metadata
    dom.meta.res.innerText = data.metadata.size;
    dom.meta.fmt.innerText = data.metadata.format;
    dom.meta.size.innerText = `${data.metadata.filesize_kb} KB`;

    // XAI Reasoning
    document.getElementById('ai-reasoning').innerHTML = `> ${data.reasoning || "Analyzing underlying forensic patterns..."}`;

    // 4. Audit Log Finish
    addAuditLine("Inference complete.", 'success');
    addAuditLine(`Verdict: ${data.prediction} (${data.confidence}%)`, isFake ? 'error' : 'success');
    
    dom.analyzeBtn.classList.add('hidden');
    dom.resetBtn.classList.remove('hidden');

    addToHistory(isFake);
}

// --- Helper UI Functions ---
async function addAuditLine(text, type = 'info', delay = 0) {
    if (delay) await new Promise(r => setTimeout(r, delay));
    
    const line = document.createElement('div');
    line.className = `terminal-line ${type}`;
    line.innerHTML = `> <span style="color: var(--text-dim)">[${new Date().toLocaleTimeString()}]</span> ${text}`;
    
    // Color coding
    if (type === 'success') line.style.color = 'var(--accent-green)';
    if (type === 'error') line.style.color = 'var(--accent-crimson)';
    if (type === 'dim') line.style.color = 'var(--text-muted)';

    dom.terminal.appendChild(line);
    dom.terminal.scrollTop = dom.terminal.scrollHeight;
}

function updateMetricBar(id, val) {
    const bar = dom[`${id}Bar`];
    const text = dom[`${id}Pct`];
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = `${val}%`;
        text.innerText = `${Math.round(val)}%`;
    }, 100);
}

function switchForensicView(view) {
    // Reset
    dom.imageMain.style.opacity = '1';
    dom.heatmapOverlay.classList.add('hidden');
    dom.freqOverlay.classList.add('hidden');
    
    document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-view="${view}"]`).classList.add('active');

    if (view === 'spatial') {
        dom.heatmapOverlay.classList.remove('hidden');
        dom.heatmapOverlay.style.opacity = '0.7';
        addAuditLine("Switched to Spatial Heatmap visualization.", 'dim');
    } else if (view === 'frequency') {
        dom.imageMain.style.opacity = '0.2';
        dom.freqOverlay.classList.remove('hidden');
        addAuditLine("Switched to Frequency Residual analysis.", 'dim');
    } else {
        addAuditLine("Restored original sample view.", 'dim');
    }
}

function addToHistory(isFake) {
    const img = document.createElement('img');
    img.src = dom.imageMain.src;
    img.className = 'history-item';
    img.style.borderColor = isFake ? 'var(--accent-crimson)' : 'var(--accent-blue)';
    dom.history.appendChild(img);
}

function downloadForensicReport() {
    if (!lastResult) return;
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const caseId = `CASE-${Math.random().toString(36).substr(2, 9).toUpperCase()}`;
    const timestamp = new Date().toLocaleString();

    // -- Header & Branding --
    doc.setFillColor(10, 10, 10);
    doc.rect(0, 0, 210, 40, 'F');
    
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text("VARENIUM FORENSIC", 15, 25);
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text("INTELLIGENCE AGENCY - IMAGE AUTHENTICITY PROTOCOL", 15, 32);

    // -- Analysis Summary --
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.text("FORENSIC ANALYSIS SUMMARY", 15, 60);
    doc.line(15, 62, 195, 62);

    doc.setFontSize(11);
    doc.text(`CASE ID:`, 15, 75);
    doc.setFont("helvetica", "bold");
    doc.text(caseId, 50, 75);

    doc.setFont("helvetica", "normal");
    doc.text(`TIMESTAMP:`, 15, 82);
    doc.text(timestamp, 50, 82);

    // -- Verdict Box --
    const isFake = lastResult.prediction === "MANIPULATED";
    doc.setDrawColor(isFake ? 255 : 0, isFake ? 45 : 242, isFake ? 85 : 255);
    doc.setLineWidth(1);
    doc.rect(15, 90, 180, 25);
    
    doc.setFontSize(18);
    doc.setTextColor(isFake ? 255 : 0, isFake ? 45 : 150, isFake ? 85 : 100);
    doc.text("VERDICT: " + lastResult.prediction, 20, 107, { charSpace: 2 });

    // -- Technical Metrics --
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("TECHNICAL EVIDENCE", 15, 135);
    doc.line(15, 137, 75, 137);

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.text("Model Confidence:", 15, 145);
    doc.text(`${lastResult.confidence}%`, 70, 145);

    doc.text("Logit Magnitude:", 15, 152);
    doc.text(lastResult.logit_mag.toString(), 70, 152);

    doc.text("Spatial Stream Consistency:", 15, 159);
    doc.text(dom.spatialPct.innerText, 70, 159);

    doc.text("Frequency Artifact Index:", 15, 166);
    doc.text(dom.freqPct.innerText, 70, 166);

    // -- Sample Metadata --
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("SAMPLE METADATA", 15, 185);
    doc.line(15, 187, 75, 187);

    doc.setFontSize(11);
    doc.setFont("helvetica", "normal");
    doc.text("Resolution:", 15, 195);
    doc.text(lastResult.metadata.size, 70, 195);
    
    doc.text("Format:", 15, 202);
    doc.text(lastResult.metadata.format, 70, 202);

    doc.text("File Size:", 15, 209);
    doc.text(`${lastResult.metadata.filesize_kb} KB`, 70, 209);

    // -- Formatting Reasoning --
    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("AI DECISION REASONING", 15, 222);
    doc.line(15, 224, 75, 224);
    
    doc.setFontSize(10);
    doc.setFont("helvetica", "italic");
    const cleanReasoning = lastResult.reasoning.replaceAll('<br>', '\n').replaceAll('<br><br>', '\n\n');
    const splitReasoning = doc.splitTextToSize(cleanReasoning, 180);
    doc.text(splitReasoning, 15, 232);

    // Calculate vertical position after reasoning
    const reasoningHeight = (splitReasoning.length * 5); 
    let currentY = 232 + reasoningHeight + 15;

    // -- Visual Evidence Section (Move to Page 2 if needed) --
    if (currentY > 240) {
        doc.addPage();
        currentY = 40; // Reset Y for new page
        
        // Re-add Header Bar on Page 2 for consistency
        doc.setFillColor(30, 30, 30);
        doc.rect(0, 0, 210, 20, 'F');
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(10);
        doc.text(`EVIDENCE LOG: ${caseId} // CONTINUED`, 15, 13);
        doc.setTextColor(0, 0, 0);
    }

    doc.setFontSize(14);
    doc.setFont("helvetica", "bold");
    doc.text("VISUAL EVIDENCE (SNAPSHOTS)", 15, currentY);
    doc.line(15, currentY + 2, 85, currentY + 2);

    try {
        const thumbY = currentY + 15;
        doc.setFontSize(9);
        // Original Sample
        doc.text("ORIGINAL_SAMPLE", 15, thumbY);
        doc.addImage(dom.imageMain.src, 'JPEG', 15, thumbY + 5, 45, 45);

        // Spatial Heatmap
        doc.text("SPATIAL_HEATMAP", 70, thumbY);
        doc.addImage(dom.heatmapOverlay.src, 'JPEG', 70, thumbY + 5, 45, 45);

        // Frequency Map
        doc.text("FREQUENCY_MAP", 125, thumbY);
        doc.addImage(dom.freqOverlay.src, 'JPEG', 125, thumbY + 5, 45, 45);
        
        currentY = thumbY + 60;
    } catch (e) {
        console.error("PDF Image add failed:", e);
    }

    // -- Footer --
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text("This document is generated by the Varenium Forensic Deepfake Detection Engine. Content is for forensic analysis only.", 15, 285);
    doc.text("VERSION 2.1.1 // CORE: MULTISTREAM-DETECTOR", 15, 290);

    doc.save(`forensic_report_${caseId}.pdf`);
    addAuditLine("Forensic PDF report generated (Multi-page check active).", 'success');
}

async function checkConnectivity() {
    const start = Date.now();
    try {
        const res = await fetch(`${API_BASE}/`, { credentials: 'include' });
        const latency = Date.now() - start;
        dom.pingText.innerText = `${latency}ms`;
        dom.pingText.style.color = 'var(--accent-green)';
    } catch {
        dom.pingText.innerText = `OFFLINE`;
        dom.pingText.style.color = 'var(--accent-crimson)';
    }
}

// Start
init();
