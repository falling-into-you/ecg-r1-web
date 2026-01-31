document.addEventListener('DOMContentLoaded', function() {
    // --- Elements ---
    const form = document.getElementById('inferenceForm');
    const imageInput = document.getElementById('imageInput');
    const ecgInput = document.getElementById('ecgInput');
    const imageDropZone = document.getElementById('imageDropZone');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const imageFileName = document.getElementById('imageFileName');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const ecgFileName = document.getElementById('ecgFileName');
    const submitBtn = document.getElementById('submitBtn');
    
    // Result panels
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultContent = document.getElementById('resultContent');
    const diagnosisText = document.getElementById('diagnosisText');
    const reasoningText = document.getElementById('reasoningText');
    const reportDate = document.getElementById('reportDate');
    const toggleReasoningBtn = document.getElementById('toggleReasoning');
    // const reasoningContent = document.getElementById('reasoningContent'); // Removed toggle content var as we handle it differently now
    
    // Feedback elements
    const feedbackSection = document.getElementById('feedbackSection');
    const likeBtn = document.getElementById('likeBtn');
    const dislikeBtn = document.getElementById('dislikeBtn');
    const feedbackMessage = document.getElementById('feedbackMessage');
    let currentRequestId = null;

    const statusBadge = document.querySelector('.status-badge');
    let isSystemOnline = false;

    // --- Feedback Handling ---
    async function sendFeedback(type) {
        if (!currentRequestId) return;
        
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    request_id: currentRequestId,
                    feedback: type
                })
            });
            
            if (response.ok) {
                feedbackMessage.classList.remove('hidden');
                
                // Update UI state
                if (type === 'like') {
                    likeBtn.classList.add('active', 'like');
                    dislikeBtn.classList.remove('active', 'dislike');
                } else {
                    dislikeBtn.classList.add('active', 'dislike');
                    likeBtn.classList.remove('active', 'like');
                }
                
                // Disable buttons after selection (optional, here we allow changing mind)
                // likeBtn.disabled = true;
                // dislikeBtn.disabled = true;
            }
        } catch (error) {
            console.error('Error sending feedback:', error);
            alert('Failed to submit feedback');
        }
    }

    likeBtn.addEventListener('click', () => sendFeedback('like'));
    dislikeBtn.addEventListener('click', () => sendFeedback('dislike'));

    // --- Check System Status ---
    async function checkSystemStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            if (data.status === 'online') {
                isSystemOnline = true;
                statusBadge.classList.remove('offline');
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> System Online';
            } else {
                isSystemOnline = false;
                statusBadge.classList.add('offline');
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> System Offline';
            }
        } catch (error) {
            console.error('Failed to check status:', error);
            isSystemOnline = false;
            statusBadge.classList.add('offline');
            statusBadge.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> Connection Error';
        }
    }
    
    // Check status immediately
    checkSystemStatus();

    // --- Helper Function to Parse Result ---
    function parseResult(text) {
        if (!text) return { thinking: "", diagnosis: "" };

        const raw = String(text);
        const thinkMatch = raw.match(/<think>([\s\S]*?)<\/think>/i);
        if (thinkMatch) {
            const thinking = (thinkMatch[1] || "").trim();
            const diagnosis = raw.replace(/<think>[\s\S]*?<\/think>/i, "").trim();
            return { thinking, diagnosis };
        }

        const markerReasoning = /(?:^|\n)\s*(?:thinking|reasoning|analysis|思考过程|思考|推理过程|推理|分析)\s*[:：]\s*/i;
        const markerFinal = /(?:^|\n)\s*(?:final|final answer|answer|diagnosis|结论|最终诊断|最终结论|诊断)\s*[:：]\s*/i;

        const reasoningMatch = raw.match(markerReasoning);
        const finalMatch = raw.match(markerFinal);
        if (finalMatch) {
            const finalIdx = finalMatch.index ?? -1;
            const finalLabelLen = finalMatch[0].length;
            const before = finalIdx >= 0 ? raw.slice(0, finalIdx).trim() : "";
            const after = finalIdx >= 0 ? raw.slice(finalIdx + finalLabelLen).trim() : raw.trim();

            let thinking = before;
            if (reasoningMatch) {
                const rIdx = reasoningMatch.index ?? -1;
                const rLen = reasoningMatch[0].length;
                if (rIdx >= 0 && rIdx + rLen <= finalIdx) {
                    thinking = raw.slice(rIdx + rLen, finalIdx).trim();
                }
            }
            return { thinking, diagnosis: after };
        }

        if (reasoningMatch) {
            const rIdx = reasoningMatch.index ?? -1;
            const rLen = reasoningMatch[0].length;
            if (rIdx >= 0) {
                const thinking = raw.slice(rIdx + rLen).trim();
                return { thinking, diagnosis: "" };
            }
        }

        return { thinking: "", diagnosis: raw.trim() };
    }

    // --- Image Upload Handling ---
    imageInput.addEventListener('change', function(e) {
        handleImageSelect(this.files[0]);
    });

    // Drag and Drop visual feedback
    imageDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageDropZone.classList.add('dragover');
    });

    imageDropZone.addEventListener('dragleave', () => {
        imageDropZone.classList.remove('dragover');
    });

    imageDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageInput.files = e.dataTransfer.files; // Update input files
            handleImageSelect(file);
        }
    });

    function handleImageSelect(file) {
        if (file) {
            imageFileName.textContent = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropzone click
        imageInput.value = '';
        imagePreview.src = '';
        imagePreviewContainer.classList.add('hidden');
        imageFileName.textContent = '';
    });

    // --- ECG File Handling ---
    ecgInput.addEventListener('change', function() {
        if (this.files[0]) {
            ecgFileName.textContent = this.files[0].name;
            ecgFileName.style.color = '#0f172a'; // Darker text
        } else {
            ecgFileName.textContent = 'Select .dat/.hea signal file...';
            ecgFileName.style.color = ''; // Reset
        }
    });

    // --- Form Submission ---
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate inputs: At least one must be provided
        if (!imageInput.files[0] && !ecgInput.files[0]) {
            alert('Please provide at least one input (Image or ECG Signal File) to proceed.');
            return;
        }

        // UI Updates
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        
        emptyState.classList.add('hidden');
        resultContent.classList.add('hidden');
        loadingState.classList.remove('hidden');
        
        const formData = new FormData(this);
        
        try {
            diagnosisText.textContent = '';
            reasoningText.textContent = '';

            const controller = new AbortController();
            const response = await fetch('/predict_stream', {
                method: 'POST',
                body: formData,
                signal: controller.signal,
                cache: 'no-store',
                headers: { 'Accept': 'text/event-stream' }
            });
            if (!response.ok) {
                let detail = 'Analysis failed';
                try {
                    const data = await response.json();
                    detail = data.detail || detail;
                } catch (e) {}
                throw new Error(detail);
            }
            const requestIdHeader = response.headers.get('x-request-id');
            let revealed = true;
            currentRequestId = null;
            submitBtn.innerHTML = '<span>Generating...</span>';
            const startedAt = Date.now();

            reportDate.textContent = new Date().toLocaleDateString('en-US', {
                year: 'numeric', month: 'long', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });

            const contentType = response.headers.get('content-type') || '';
            if (!contentType.includes('text/event-stream') || !response.body) {
                const fallbackResp = await fetch('/predict', { method: 'POST', body: new FormData(this) });
                const data = await fallbackResp.json();
                if (!fallbackResp.ok) {
                    throw new Error(data.detail || 'Analysis failed');
                }
                const parsed = parseResult(data.result);
                diagnosisText.textContent = parsed.diagnosis;
                reasoningText.textContent = parsed.thinking || "No detailed reasoning process provided by the model.";
                currentRequestId = data.request_id;
                return;
            }

            loadingState.classList.add('hidden');
            resultContent.classList.remove('hidden');
            feedbackSection.classList.remove('hidden');
            likeBtn.classList.remove('active', 'like');
            dislikeBtn.classList.remove('active', 'dislike');
            feedbackMessage.classList.add('hidden');
            reasoningText.textContent = '生成中...';

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let contentBuf = '';
            let reasoningBuf = '';
            let rawProcessedLen = 0;
            const streamState = { inThink: false, carry: '' };
            let placeholderShown = true;
            let gotAnyChunk = false;

            function revealIfNeeded() {
                if (revealed) return;
                loadingState.classList.add('hidden');
                resultContent.classList.remove('hidden');
                feedbackSection.classList.remove('hidden');
                likeBtn.classList.remove('active', 'like');
                dislikeBtn.classList.remove('active', 'dislike');
                feedbackMessage.classList.add('hidden');
                revealed = true;
                placeholderShown = true;
                reasoningText.textContent = '生成中...';
            }

            function takeCarryPrefix(text) {
                const candidates = ['<think>', '</think>'];
                const lower = text.toLowerCase();
                let bestLen = 0;
                for (const cand of candidates) {
                    for (let i = 1; i < cand.length; i++) {
                        const suffix = lower.slice(-i);
                        if (cand.startsWith(suffix) && i > bestLen) {
                            bestLen = i;
                        }
                    }
                }
                if (bestLen > 0) {
                    return { body: text.slice(0, -bestLen), carry: text.slice(-bestLen) };
                }
                return { body: text, carry: '' };
            }

            function routeContentChunk(chunk) {
                let text = streamState.carry + String(chunk);
                streamState.carry = '';
                const split = takeCarryPrefix(text);
                text = split.body;
                streamState.carry = split.carry;

                while (text.length > 0) {
                    const lower = text.toLowerCase();
                    const openIdx = lower.indexOf('<think>');
                    const closeIdx = lower.indexOf('</think>');

                    if (!streamState.inThink) {
                        if (openIdx === -1) {
                            contentBuf += text;
                            diagnosisText.textContent = contentBuf;
                            return;
                        }
                        const before = text.slice(0, openIdx);
                        if (before) {
                            contentBuf += before;
                            diagnosisText.textContent = contentBuf;
                        }
                        reasoningBuf += '<think>';
                        reasoningText.textContent = reasoningBuf;
                        text = text.slice(openIdx + '<think>'.length);
                        streamState.inThink = true;
                    } else {
                        if (closeIdx === -1) {
                            reasoningBuf += text;
                            reasoningText.textContent = reasoningBuf;
                            return;
                        }
                        const before = text.slice(0, closeIdx);
                        if (before) {
                            reasoningBuf += before;
                            reasoningText.textContent = reasoningBuf;
                        }
                        reasoningBuf += '</think>';
                        reasoningText.textContent = reasoningBuf;
                        text = text.slice(closeIdx + '</think>'.length);
                        streamState.inThink = false;
                    }
                }
            }

            async function pollProgress(requestId) {
                while (true) {
                    await new Promise(r => setTimeout(r, 1000));
                    let resp;
                    let data;
                    try {
                        resp = await fetch(`/predict_progress/${requestId}`, { cache: 'no-store' });
                        data = await resp.json();
                        if (!resp.ok) throw new Error(data.detail || 'Polling failed');
                    } catch (e) {
                        continue;
                    }

                    const full = data.content || '';
                    if (full.length > rawProcessedLen) {
                        const delta = full.slice(rawProcessedLen);
                        rawProcessedLen = full.length;
                        routeContentChunk(delta);
                        gotAnyChunk = true;
                    }
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    if (data.done) {
                        return;
                    }
                }
            }

            let pollPromise = null;
            if (requestIdHeader) {
                setTimeout(() => {
                    if (!gotAnyChunk) {
                        try { controller.abort(); } catch (e) {}
                        pollPromise = pollProgress(requestIdHeader);
                    }
                }, 5000);
            }

            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const parts = buffer.split(/\\r?\\n\\r?\\n/);
                    buffer = parts.pop();

                    for (const part of parts) {
                        const lines = part.split(/\\r?\\n/);
                        let eventType = 'message';
                        const dataLines = [];
                        for (const line of lines) {
                            if (line.startsWith('event:')) eventType = line.slice(6).trim();
                            if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
                        }
                        const dataStr = dataLines.join('\\n');
                        let payload = dataStr;
                        try { payload = JSON.parse(dataStr); } catch (e) {}

                        if (eventType === 'reasoning') {
                            revealIfNeeded();
                            if (placeholderShown) {
                                reasoningBuf = '';
                                reasoningText.textContent = '';
                                placeholderShown = false;
                            }
                            reasoningBuf += payload;
                            reasoningText.textContent = reasoningBuf;
                            gotAnyChunk = true;
                        } else if (eventType === 'content') {
                            revealIfNeeded();
                            if (placeholderShown) {
                                reasoningText.textContent = '';
                                placeholderShown = false;
                            }
                            const s = String(payload);
                            rawProcessedLen += s.length;
                            routeContentChunk(s);
                            gotAnyChunk = true;
                        } else if (eventType === 'ready') {
                            revealIfNeeded();
                        } else if (eventType === 'ping') {
                            if (placeholderShown) {
                                const sec = Math.floor((Date.now() - startedAt) / 1000);
                                reasoningText.textContent = `生成中... (${sec}s)`;
                            }
                        } else if (eventType === 'done') {
                            if (payload && payload.request_id) currentRequestId = payload.request_id;
                            revealIfNeeded();
                            if (!diagnosisText.textContent && !reasoningText.textContent) {
                                diagnosisText.textContent = '模型未返回内容';
                            }
                            if (pollPromise) await pollPromise;
                        } else if (eventType === 'error') {
                            const msg = (payload && payload.detail) ? payload.detail : 'Streaming error';
                            throw new Error(msg);
                        }
                    }
                }
            } catch (err) {
                const msg = String(err && (err.message || err));
                const isAbort = (err && err.name === 'AbortError') || msg.includes('BodyStreamBuffer was aborted');
                if (pollPromise && isAbort) {
                    await pollPromise;
                } else {
                    throw err;
                }
            }
            
        } catch (error) {
            loadingState.classList.add('hidden');
            emptyState.classList.remove('hidden'); // Go back to empty state or show error
            alert('Error: ' + error.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span>Start Analysis</span><i class="fa-solid fa-arrow-right"></i>';
        }
    });
});
