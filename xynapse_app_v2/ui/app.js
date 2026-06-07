/**
 * Xynapse Model Comparison — Frontend Logic
 * Handles file upload, API calls, and rendering results.
 */

// ── DOM References ──────────────────────────────────────────────────────────
const uploadZone    = document.getElementById('upload-zone');
const fileInput     = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const uploadPreview = document.getElementById('upload-preview');
const previewImg    = document.getElementById('preview-img');
const removeBtn     = document.getElementById('remove-btn');
const fileNameEl    = document.getElementById('file-name');
const reportText    = document.getElementById('report-text');
const charCount     = document.getElementById('char-count');
const runBtn        = document.getElementById('run-btn');
const btnContent    = document.querySelector('.run-btn-content');
const btnLoading    = document.querySelector('.run-btn-loading');
const resultsSection = document.getElementById('results-section');
const errorToast    = document.getElementById('error-toast');
const errorMsg      = document.getElementById('error-msg');

let selectedFile = null;

// ── File Upload ─────────────────────────────────────────────────────────────

uploadZone.addEventListener('click', (e) => {
    if (e.target.closest('.remove-btn')) return;
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file (PNG, JPEG, etc.)');
        return;
    }
    selectedFile = file;
    fileNameEl.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadContent.hidden = true;
        uploadPreview.hidden = false;
    };
    reader.readAsDataURL(file);

    updateRunBtn();
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    uploadContent.hidden = false;
    uploadPreview.hidden = true;
    previewImg.src = '';
    updateRunBtn();
});

// ── Report Text ─────────────────────────────────────────────────────────────

reportText.addEventListener('input', () => {
    const len = reportText.value.length;
    charCount.textContent = `${len} character${len !== 1 ? 's' : ''}`;
});

// ── Run Button State ────────────────────────────────────────────────────────

function updateRunBtn() {
    runBtn.disabled = !selectedFile;
}

// ── Run Inference ───────────────────────────────────────────────────────────

runBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Enter loading state
    runBtn.disabled = true;
    btnContent.hidden = true;
    btnLoading.hidden = false;
    hideError();

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('report', reportText.value);

    try {
        const resp = await fetch('/api/predict', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.error || `Server error: ${resp.status}`);
        }

        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        showError(err.message || 'Something went wrong. Check the server console.');
    } finally {
        runBtn.disabled = false;
        btnContent.hidden = false;
        btnLoading.hidden = true;
    }
});

// ── Render Results ──────────────────────────────────────────────────────────

function renderResults(data) {
    resultsSection.hidden = false;

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Meta info
    const mode = data.new_model?.result?.modality || '?';
    document.getElementById('results-mode').textContent = mode === 'multimodal' ? '🔗 Multimodal' : '🖼️ Image Only';
    document.getElementById('results-device').textContent = `Device: ${data.device.toUpperCase()}`;

    // Render new model panel
    if (data.new_model?.result) {
        renderPanel('new', data.new_model);
    }

    // Render old model panel only if data exists
    const oldPanel = document.getElementById('panel-old');
    if (data.old_model?.result) {
        oldPanel.hidden = false;
        renderPanel('old', data.old_model);
    } else {
        oldPanel.hidden = true;
    }

    // Diff table — only show if both models returned results
    const diffSection = document.getElementById('diff-section');
    if (data.new_model?.result && data.old_model?.result) {
        diffSection.hidden = false;
        renderDiffTable(data.new_model.result, data.old_model.result);
    } else {
        diffSection.hidden = true;
    }

    // Logs
    document.getElementById('log-new').textContent = data.new_model?.log || '';
    const oldLogPanel = document.getElementById('log-panel-old');
    if (data.old_model?.log) {
        oldLogPanel.hidden = false;
        document.getElementById('log-old').textContent = data.old_model.log;
    } else {
        oldLogPanel.hidden = true;
    }
}

const LABELS = ['Cardiomegaly', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation'];

function renderPanel(id, model) {
    const result = model.result;
    const timeEl = document.getElementById(`time-${id}`);
    timeEl.textContent = `${model.time}s`;

    // Probability bars
    const probsEl = document.getElementById(`probs-${id}`);
    probsEl.innerHTML = '';

    for (const label of LABELS) {
        const prob = result.probs[label];
        const isPositive = result.predictions[label] === 1;
        const threshold = result.thresholds[label];

        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-label-row">
                <span class="prob-name">${label}</span>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span class="prob-status ${isPositive ? 'positive' : 'negative'}">${isPositive ? '✓ Positive' : 'Negative'}</span>
                    <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                </div>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isPositive ? 'positive' : 'negative'}" style="width: 0%"></div>
                <div class="prob-bar-threshold" style="left: ${threshold * 100}%" title="Threshold: ${threshold}"></div>
            </div>
        `;
        probsEl.appendChild(row);

        // Animate bar fill
        requestAnimationFrame(() => {
            setTimeout(() => {
                row.querySelector('.prob-bar-fill').style.width = `${prob * 100}%`;
            }, 50);
        });
    }

    // Detected summary
    const detectedEl = document.getElementById(`detected-${id}`);
    if (result.detected.length > 0) {
        detectedEl.className = 'detected-summary has-findings';
        detectedEl.innerHTML = `
            <div class="detected-label">⚠ Findings Detected</div>
            <div>${result.detected.join(', ')}</div>
        `;
    } else {
        detectedEl.className = 'detected-summary no-findings';
        detectedEl.innerHTML = `<div>✓ No acute findings detected</div>`;
    }

    // Thresholds
    const thresholdsEl = document.getElementById(`thresholds-${id}`);
    thresholdsEl.innerHTML = `
        <div class="threshold-title">Classification Thresholds</div>
        <div class="threshold-chips">
            ${LABELS.map(l => `<span class="threshold-chip">${l.split(' ')[0]}: ${result.thresholds[l]}</span>`).join('')}
        </div>
    `;
}

function renderDiffTable(newResult, oldResult) {
    const tbody = document.getElementById('diff-tbody');
    tbody.innerHTML = '';

    for (const label of LABELS) {
        const pNew = newResult.probs[label];
        const pOld = oldResult.probs[label];
        const diff = pNew - pOld;
        const predNew = newResult.predictions[label];
        const predOld = oldResult.predictions[label];
        const agree = predNew === predOld;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-family:var(--font-sans);font-weight:500;color:var(--text-secondary)">${label}</td>
            <td>${(pNew * 100).toFixed(1)}%</td>
            <td>${(pOld * 100).toFixed(1)}%</td>
            <td class="${diff > 0 ? 'diff-positive' : diff < 0 ? 'diff-negative' : 'diff-neutral'}">
                ${diff > 0 ? '+' : ''}${(diff * 100).toFixed(1)}%
            </td>
            <td class="${predNew ? 'pred-pos' : 'pred-neg'}">${predNew ? 'POSITIVE' : 'negative'}</td>
            <td class="${predOld ? 'pred-pos' : 'pred-neg'}">${predOld ? 'POSITIVE' : 'negative'}</td>
            <td class="${agree ? 'agree-yes' : 'agree-no'}">${agree ? '✓ Agree' : '✗ Disagree'}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ── Error Handling ──────────────────────────────────────────────────────────

function showError(msg) {
    errorMsg.textContent = msg;
    errorToast.hidden = false;
    setTimeout(() => hideError(), 8000);
}

function hideError() {
    errorToast.hidden = true;
}

// ── Init ────────────────────────────────────────────────────────────────────
updateRunBtn();
