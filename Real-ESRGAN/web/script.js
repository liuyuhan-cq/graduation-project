const imageInput = document.getElementById('imageInput');
const modelSelect = document.getElementById('modelSelect');
const processBtn = document.getElementById('processBtn');
const batchBtn = document.getElementById('batchBtn');
const statusDiv = document.getElementById('status');

const inputPreview = document.getElementById('inputPreview');
const outputPreview = document.getElementById('outputPreview');
const inputInfo = document.getElementById('inputInfo');
const outputInfo = document.getElementById('outputInfo');

const downloadInputBtn = document.getElementById('downloadInputBtn');
const downloadOutputBtn = document.getElementById('downloadOutputBtn');
const downloadCompareBtn = document.getElementById('downloadCompareBtn');

const batchImageInput = document.getElementById('batchImageInput');
const batchProcessBtn = document.getElementById('batchProcessBtn');
const batchStatus = document.getElementById('batchStatus');
const batchResults = document.getElementById('batchResults');
const batchSummaryActions = document.getElementById('batchSummaryActions');
const allDownloadOutputBtn = document.getElementById('allDownloadOutputBtn');
const allDownloadCompareBtn = document.getElementById('allDownloadCompareBtn');

const modal = document.getElementById('imageModal');
const modalImg = document.getElementById('modalImg');
const modalClose = document.getElementById('modalClose');

let selectedFile = null;
let selectedFiles = [];
let outputBase64 = null;
let compareBase64 = null;
let inputDataUrl = null;
let currentInputSize = null;
let currentOutputSize = null;
let currentModelName = null;
let batchDownloadItems = [];
let currentBatchId = null;

function showError(message) {
    statusDiv.textContent = message;
    statusDiv.classList.add('error');
}

function clearError() {
    statusDiv.classList.remove('error');
}

function openModal(src) {
    modalImg.src = src;
    modal.classList.add('show');
}

function closeModal() {
    modal.classList.remove('show');
    modalImg.src = '';
}

function base64ToBlob(base64, mimeType = 'image/png') {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function downloadDataUrl(dataUrl, filename) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
}

function safeStem(name) {
    const base = String(name || 'image').replace(/\.[^.]+$/, '');
    return base.replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, '_');
}

async function downloadZipByBatch(batchId, kind) {
    if (!batchId) {
        batchStatus.textContent = '请先完成一次批量处理';
        return;
    }

    const url = `http://localhost:8000/batch_download_zip/${encodeURIComponent(batchId)}/${kind}`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`ZIP 下载失败（${res.status}）`);
    }

    const blob = await res.blob();
    const filename = kind === 'output'
        ? `batch_${batchId}_outputs.zip`
        : `batch_${batchId}_compares.zip`;
    downloadBlob(blob, filename);
}

function buildCompareCanvas() {
    return new Promise((resolve, reject) => {
        if (!inputPreview.src || !outputPreview.src) {
            reject(new Error('缺少原图或结果图，无法生成对比图'));
            return;
        }

        const img1 = new Image();
        const img2 = new Image();
        let loaded = 0;

        function onLoaded() {
            loaded += 1;
            if (loaded < 2) return;

            const padding = 24;
            const titleH = 92;
            const labelH = 42;
            const maxH = Math.max(img1.height, img2.height);

            const canvas = document.createElement('canvas');
            canvas.width = Math.max(img1.width, img2.width) * 2 + padding * 3;
            canvas.height = maxH + titleH + labelH + padding * 2;

            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#111111';
            ctx.font = 'bold 26px Arial';
            ctx.fillText('Real-ESRGAN 对比图', padding, 44);

            ctx.font = '20px Arial';
            ctx.fillText('原图', padding, 80);
            ctx.fillText('超分结果', canvas.width / 2 + padding / 2, 80);

            const drawContain = (img, x, y, w, h) => {
                const ratio = Math.min(w / img.width, h / img.height);
                const dw = img.width * ratio;
                const dh = img.height * ratio;
                const dx = x + (w - dw) / 2;
                const dy = y + (h - dh) / 2;
                ctx.drawImage(img, dx, dy, dw, dh);
            };

            const halfW = (canvas.width - padding * 3) / 2;
            drawContain(img1, padding, titleH + 10, halfW, maxH);
            drawContain(img2, halfW + padding * 2, titleH + 10, halfW, maxH);

            ctx.fillStyle = '#444444';
            ctx.font = '18px Arial';
            ctx.fillText(`输入尺寸：${currentInputSize ? currentInputSize.join(' × ') : ''}`, padding, canvas.height - 16);
            ctx.fillText(`输出尺寸：${currentOutputSize ? currentOutputSize.join(' × ') : ''}`, halfW + padding * 2, canvas.height - 16);

            resolve(canvas);
        }

        img1.onload = onLoaded;
        img2.onload = onLoaded;
        img1.onerror = reject;
        img2.onerror = reject;

        img1.src = inputPreview.src;
        img2.src = outputPreview.src;
    });
}

function renderBatchSummaryButtons() {
    batchSummaryActions.style.display = batchDownloadItems.length > 0 ? 'flex' : 'none';
}

imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        inputPreview.src = e.target.result;
        inputDataUrl = e.target.result;
        inputInfo.textContent = `文件名：${file.name}，大小：${(file.size / 1024).toFixed(2)} KB`;
    };
    reader.readAsDataURL(file);
});

batchImageInput.addEventListener('change', function () {
    selectedFiles = Array.from(this.files || []);

    if (!selectedFiles.length) {
        batchStatus.textContent = '未选择批量图片';
        return;
    }

    batchStatus.textContent = `已选择 ${selectedFiles.length} 张图片，点击“开始批量超分”执行`;
});

processBtn.addEventListener('click', async function () {
    if (!selectedFile) {
        showError('请先选择一张图片');
        return;
    }

    clearError();
    statusDiv.textContent = '正在处理，请稍候...';
    processBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', modelSelect.value);

        const response = await fetch('http://localhost:8000/process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || '服务器处理失败');
        }

        outputBase64 = data.output;
        compareBase64 = data.compare || null;
        currentInputSize = data.input_size;
        currentOutputSize = data.output_size;
        currentModelName = data.model;

        outputPreview.src = `data:image/png;base64,${data.output}`;

        const [inW, inH] = data.input_size;
        const [outW, outH] = data.output_size;

        inputInfo.textContent = `输入尺寸：${inW} × ${inH}`;
        outputInfo.textContent = `输出尺寸：${outW} × ${outH}`;

        statusDiv.textContent = `处理完成，使用模型：${data.model}`;
    } catch (err) {
        console.error(err);
        showError(`处理失败：${err.message}`);
    } finally {
        processBtn.disabled = false;
    }
});

downloadInputBtn.addEventListener('click', function () {
    if (!inputDataUrl) {
        showError('暂无原图可下载');
        return;
    }

    downloadDataUrl(inputDataUrl, selectedFile ? selectedFile.name : 'input.png');
});

downloadOutputBtn.addEventListener('click', function () {
    if (!outputBase64) {
        showError('暂无超分结果可下载');
        return;
    }

    const blob = base64ToBlob(outputBase64);
    const modelName = currentModelName || modelSelect.value || 'RealESRGAN';
    downloadBlob(blob, `${modelName}_output.png`);
});

downloadCompareBtn.addEventListener('click', async function () {
    if (compareBase64) {
        const blob = base64ToBlob(compareBase64);
        const modelName = currentModelName || modelSelect.value || 'RealESRGAN';
        downloadBlob(blob, `${modelName}_compare.png`);
        return;
    }

    try {
        const canvas = await buildCompareCanvas();
        canvas.toBlob((blob) => {
            if (!blob) {
                showError('对比图生成失败');
                return;
            }
            downloadBlob(blob, `compare_${currentModelName || modelSelect.value || 'RealESRGAN'}.png`);
        }, 'image/png');
    } catch (err) {
        showError(err.message);
    }
});

batchBtn.addEventListener('click', function () {
    batchImageInput.click();
});

batchProcessBtn.addEventListener('click', async function () {
    if (!selectedFiles.length) {
        batchStatus.textContent = '请先选择至少一张图片';
        return;
    }

    batchStatus.textContent = '批量处理中，请稍候...';
    batchProcessBtn.disabled = true;
    batchResults.innerHTML = '';
    batchDownloadItems = [];
    currentBatchId = null;
    renderBatchSummaryButtons();

    try {
        const formData = new FormData();
        selectedFiles.forEach((file) => formData.append('files', file));
        formData.append('model', modelSelect.value);

        const response = await fetch('http://localhost:8000/batch_process', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || '批量处理失败');
        }

        currentBatchId = data.batch_id || null;
        batchStatus.textContent = `批量处理完成：共 ${data.count} 张，模型：${data.model}`;

        data.results.forEach((item, index) => {
            const div = document.createElement('div');
            div.className = 'batch-item';

            if (item.error) {
                div.innerHTML = `<strong>${index + 1}. ${item.filename}</strong><div class="line">错误：${item.error}</div>`;
            } else {
                batchDownloadItems.push(item);

                div.innerHTML = `
                    <strong>${index + 1}. ${item.filename}</strong>
                    <div class="line">输入尺寸：${item.input_size[0]} × ${item.input_size[1]}</div>
                    <div class="line">输出尺寸：${item.output_size[0]} × ${item.output_size[1]}</div>
                    <div class="actions">
                        <button class="secondary-btn" type="button" data-kind="single-output">下载超分结果</button>
                        <button class="secondary-btn" type="button" data-kind="single-compare">下载对比图</button>
                    </div>
                `;

                const outputBtn = div.querySelector('[data-kind="single-output"]');
                const compareBtn = div.querySelector('[data-kind="single-compare"]');

                outputBtn.addEventListener('click', () => {
                    const blob = base64ToBlob(item.output_base64);
                    downloadBlob(blob, `${safeStem(item.filename)}_sr.png`);
                });

                compareBtn.addEventListener('click', () => {
                    const blob = base64ToBlob(item.compare_base64);
                    downloadBlob(blob, `${safeStem(item.filename)}_compare.png`);
                });
            }

            batchResults.appendChild(div);
        });

        renderBatchSummaryButtons();
    } catch (err) {
        console.error(err);
        batchStatus.textContent = `批量处理失败：${err.message}`;
    } finally {
        batchProcessBtn.disabled = false;
    }
});

allDownloadOutputBtn.addEventListener('click', function () {
    downloadZipByBatch(currentBatchId, 'output')
        .then(() => {
            batchStatus.textContent = '超分结果 ZIP 已开始下载';
        })
        .catch((err) => {
            batchStatus.textContent = `下载失败：${err.message}`;
        });
});

allDownloadCompareBtn.addEventListener('click', function () {
    downloadZipByBatch(currentBatchId, 'compare')
        .then(() => {
            batchStatus.textContent = '对比图 ZIP 已开始下载';
        })
        .catch((err) => {
            batchStatus.textContent = `下载失败：${err.message}`;
        });
});

inputPreview.addEventListener('click', function () {
    if (inputPreview.src) {
        openModal(inputPreview.src);
    }
});

outputPreview.addEventListener('click', function () {
    if (outputPreview.src) {
        openModal(outputPreview.src);
    }
});

modalClose.addEventListener('click', closeModal);

modal.addEventListener('click', function (e) {
    if (e.target === modal) {
        closeModal();
    }
});

document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});
