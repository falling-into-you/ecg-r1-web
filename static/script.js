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
    const answerText = document.getElementById('answerText');
    const answerSection = document.getElementById('answerSection');
    const reasoningText = document.getElementById('reasoningText');
    const reportDate = document.getElementById('reportDate');
    const requestIdValue = document.getElementById('requestIdValue');
    const copyRequestIdBtn = document.getElementById('copyRequestIdBtn');
    const toggleReasoningBtn = document.getElementById('toggleReasoning');
    const reasoningSection = document.querySelector('.reasoning-section');
    const reasoningContent = document.getElementById('reasoningContent');
    
    // Feedback elements
    const feedbackSection = document.getElementById('feedbackSection');
    const likeBtn = document.getElementById('likeBtn');
    const dislikeBtn = document.getElementById('dislikeBtn');
    const feedbackMessage = document.getElementById('feedbackMessage');
    const feedbackDetail = document.getElementById('feedbackDetail');
    const feedbackDetailTitle = document.getElementById('feedbackDetailTitle');
    const feedbackComment = document.getElementById('feedbackComment');
    const feedbackSubmitBtn = document.getElementById('feedbackSubmitBtn');
    let currentRequestId = null;
    let isAnalysisComplete = false;
    let pendingFeedbackType = null;

    async function copyToClipboard(text) {
        const value = String(text || '').trim();
        if (!value) return false;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(value);
            return true;
        }
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(textarea);
        return ok;
    }

    if (copyRequestIdBtn) {
        copyRequestIdBtn.addEventListener('click', async () => {
            try {
                const ok = await copyToClipboard(currentRequestId || (requestIdValue && requestIdValue.textContent));
                if (ok) {
                    copyRequestIdBtn.title = 'Copied';
                    setTimeout(() => { copyRequestIdBtn.title = 'Copy request id'; }, 1000);
                }
            } catch (e) {}
        });
    }

    if (toggleReasoningBtn && reasoningSection && reasoningContent) {
        toggleReasoningBtn.setAttribute('aria-expanded', 'true');
        toggleReasoningBtn.addEventListener('click', () => {
            const collapsed = reasoningSection.classList.toggle('collapsed');
            toggleReasoningBtn.setAttribute('aria-expanded', String(!collapsed));
        });
    }

    const statusBadge = document.querySelector('.status-badge');
    let isSystemOnline = false;
    const apiBaseMeta = document.querySelector('meta[name="ecg-api-base"]');
    const apiBase = apiBaseMeta && apiBaseMeta.content ? apiBaseMeta.content.trim().replace(/\/$/, '') : '';
    const apiUrl = (path) => apiBase ? `${apiBase}${path}` : path;

    function openFeedbackDetail(type) {
        pendingFeedbackType = type;
        feedbackMessage.classList.add('hidden');
        feedbackSubmitBtn.disabled = false;

        if (type === 'like') {
            likeBtn.classList.add('active', 'like');
            dislikeBtn.classList.remove('active', 'dislike');
            feedbackDetailTitle.textContent = 'You marked this interpretation as Helpful.';
        } else {
            dislikeBtn.classList.add('active', 'dislike');
            likeBtn.classList.remove('active', 'like');
            feedbackDetailTitle.textContent = 'You marked this interpretation as Not Helpful.';
        }

        feedbackDetail.classList.remove('hidden');
        if (feedbackComment) feedbackComment.focus();
    }

    async function submitFeedback() {
        if (!currentRequestId || !pendingFeedbackType) return;

        feedbackSubmitBtn.disabled = true;
        try {
            const text = (feedbackComment && feedbackComment.value) ? String(feedbackComment.value).trim() : '';
            const response = await fetch(apiUrl('/feedback'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_id: currentRequestId,
                    feedback: pendingFeedbackType,
                    comment: text || null
                })
            });

            if (!response.ok) {
                let detail = 'Failed to submit feedback';
                try {
                    const data = await response.json();
                    detail = data.detail || detail;
                } catch (e) {}
                throw new Error(detail);
            }

            feedbackMessage.classList.remove('hidden');
            if (feedbackComment) feedbackComment.value = '';
            feedbackDetail.classList.add('hidden');
        } catch (error) {
            alert('Error: ' + (error && error.message ? error.message : String(error)));
        } finally {
            feedbackSubmitBtn.disabled = false;
        }
    }

    likeBtn.addEventListener('click', () => {
        if (!currentRequestId) {
            // Try to recover from DOM if missing (e.g. after refresh if we persisted state, though here we don't)
            const val = requestIdValue.textContent.trim();
            if (val && val !== '--') currentRequestId = val;
        }

        if (!currentRequestId) {
            alert('Waiting for analysis to start...');
            return;
        }
        openFeedbackDetail('like');
    });

    dislikeBtn.addEventListener('click', () => {
        if (!currentRequestId) {
             const val = requestIdValue.textContent.trim();
             if (val && val !== '--') currentRequestId = val;
        }

        if (!currentRequestId) {
            alert('Waiting for analysis to start...');
            return;
        }
        openFeedbackDetail('dislike');
    });

    if (feedbackSubmitBtn) {
        feedbackSubmitBtn.addEventListener('click', submitFeedback);
    }

    // --- Check System Status ---
    async function checkSystemStatus() {
        try {
            const response = await fetch(apiUrl('/status'));
            const data = await response.json();
            
            if (data.status === 'online') {
                isSystemOnline = true;
                statusBadge.classList.remove('offline');
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> System Online';
            } else if (data.status === 'loading') {
                isSystemOnline = false;
                statusBadge.classList.remove('offline');
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> System Loading';
            } else {
                isSystemOnline = false;
                statusBadge.classList.add('offline');
                statusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> System Offline';
            }
            statusBadge.title = data.detail ? String(data.detail) : '';
        } catch (error) {
            console.error('Failed to check status:', error);
            isSystemOnline = false;
            statusBadge.classList.add('offline');
            statusBadge.innerHTML = '<i class="fa-solid fa-circle-exclamation"></i> Connection Error';
            statusBadge.title = '';
        }
    }
    
    // Check status immediately
    checkSystemStatus();
    setInterval(checkSystemStatus, 10000);

    // --- Helper Function to Parse Result ---
    function parseResult(text) {
        if (!text) return { thinking: "", diagnosis: "", answer: "" };

        const raw = String(text);
        
        // Extract Answer first <answer>...</answer>
        let answer = "";
        let content = raw;
        const answerMatch = raw.match(/<answer>([\s\S]*?)<\/answer>/i);
        if (answerMatch) {
            answer = (answerMatch[1] || "").trim();
            content = raw.replace(/<answer>[\s\S]*?<\/answer>/i, "").trim();
        }

        const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
        if (thinkMatch) {
            const thinking = (thinkMatch[1] || "").trim();
            const diagnosis = content.replace(/<think>[\s\S]*?<\/think>/i, "").trim();
            return { thinking, diagnosis, answer };
        }

        const markerReasoning = /(?:^|\n)\s*(?:thinking|reasoning|analysis|思考过程|思考|推理过程|推理|分析)\s*[:：]\s*/i;
        const markerFinal = /(?:^|\n)\s*(?:final|final answer|answer|diagnosis|结论|最终诊断|最终结论|诊断)\s*[:：]\s*/i;

        const reasoningMatch = content.match(markerReasoning);
        const finalMatch = content.match(markerFinal);
        if (finalMatch) {
            const finalIdx = finalMatch.index ?? -1;
            const finalLabelLen = finalMatch[0].length;
            const before = finalIdx >= 0 ? content.slice(0, finalIdx).trim() : "";
            const after = finalIdx >= 0 ? content.slice(finalIdx + finalLabelLen).trim() : content.trim();

            let thinking = before;
            if (reasoningMatch) {
                const rIdx = reasoningMatch.index ?? -1;
                const rLen = reasoningMatch[0].length;
                if (rIdx >= 0 && rIdx + rLen <= finalIdx) {
                    thinking = content.slice(rIdx + rLen, finalIdx).trim();
                }
            }
            return { thinking, diagnosis: after, answer };
        }

        if (reasoningMatch) {
            const rIdx = reasoningMatch.index ?? -1;
            const rLen = reasoningMatch[0].length;
            if (rIdx >= 0) {
                const thinking = content.slice(rIdx + rLen).trim();
                return { thinking, diagnosis: "", answer };
            }
        }

        return { thinking: "", diagnosis: content.trim(), answer };
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
            // Validate extension
            const ext = file.name.split('.').pop().toLowerCase();
            if (!['png', 'jpg', 'jpeg'].includes(ext)) {
                alert('Only .png, .jpg, or .jpeg images are allowed.');
                imageInput.value = '';
                imageFileName.textContent = '';
                imagePreview.src = '';
                imagePreviewContainer.classList.add('hidden');
                return;
            }

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
        if (this.files.length > 0) {
            // Requirement: Must select exactly 2 files (.dat and .hea)
            if (this.files.length !== 2) {
                alert('Please select exactly 2 files (.dat and .hea) for the ECG signal.');
                this.value = ''; // Clear selection
                ecgFileName.textContent = 'Select .dat and .hea files...';
                ecgFileName.style.color = '';
                return;
            }

            const files = Array.from(this.files);
            const exts = files.map(f => f.name.split('.').pop().toLowerCase());
            
            const hasDat = exts.includes('dat');
            const hasHea = exts.includes('hea');

            if (!hasDat || !hasHea) {
                alert('You must select one .dat file and one .hea file.');
                this.value = ''; // Clear selection
                ecgFileName.textContent = 'Select .dat and .hea files...';
                ecgFileName.style.color = '';
                return;
            }

            // Success
            ecgFileName.textContent = files.map(f => f.name).join(', ');
            ecgFileName.style.color = '#0f172a'; // Darker text
        } else {
            ecgFileName.textContent = 'Select .dat and .hea files...';
            ecgFileName.style.color = ''; // Reset
        }
    });

    // --- Form Submission ---
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        isAnalysisComplete = false;
        
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
            const response = await fetch(apiUrl('/predict_stream'), {
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
            currentRequestId = requestIdHeader; // Set immediately
            submitBtn.innerHTML = '<span>Generating...</span>';
            const startedAt = Date.now();
            if (requestIdValue) requestIdValue.textContent = currentRequestId || '--';

            reportDate.textContent = new Date().toLocaleDateString('en-US', {
                year: 'numeric', month: 'long', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });

            const contentType = response.headers.get('content-type') || '';
            if (!contentType.includes('text/event-stream') || !response.body) {
                const fallbackResp = await fetch(apiUrl('/predict'), { method: 'POST', body: new FormData(this) });
                const data = await fallbackResp.json();
                if (!fallbackResp.ok) {
                    throw new Error(data.detail || 'Analysis failed');
                }
                const parsed = parseResult(data.result);
                diagnosisText.textContent = parsed.diagnosis;
                reasoningText.textContent = parsed.thinking || "No detailed reasoning process provided by the model.";
                
                if (parsed.answer) {
                    answerText.textContent = parsed.answer;
                    answerSection.classList.remove('hidden');
                } else {
                    answerSection.classList.add('hidden');
                }

                currentRequestId = data.request_id;
                if (requestIdValue) requestIdValue.textContent = currentRequestId || '--';
                isAnalysisComplete = true;
                return;
            }

            loadingState.classList.add('hidden');
            resultContent.classList.remove('hidden');
            feedbackSection.classList.remove('hidden');
            likeBtn.classList.remove('active', 'like');
            dislikeBtn.classList.remove('active', 'dislike');
            feedbackMessage.classList.add('hidden');
            reasoningText.textContent = 'Generating...';
            reasoningText.classList.add('streaming');
            diagnosisText.classList.remove('streaming');
            answerText.classList.remove('streaming');
            answerSection.classList.remove('hidden');
            answerText.textContent = '';

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let contentBuf = '';
            let reasoningBuf = '';
            let answerBuf = '';
            let rawProcessedLen = 0;
            const streamState = { inThink: false, inAnswer: false, carry: '' };
            let sawAnswer = false;
            let answerPlaceholderShown = true;
            let placeholderShown = true;
            let gotAnyChunk = false;
            let usePolling = false;
            let pendingDiagnosis = '';
            let pendingReasoning = '';
            let pendingAnswer = '';
            let flushRaf = null;
            let streamDone = false;
            let diagnosisStarted = false;

            function setCursorTarget(target) {
                if (target === 'reasoning') {
                    reasoningText.classList.add('streaming');
                    diagnosisText.classList.remove('streaming');
                    answerText.classList.remove('streaming');
                } else if (target === 'diagnosis') {
                    reasoningText.classList.remove('streaming');
                    diagnosisText.classList.add('streaming');
                    answerText.classList.remove('streaming');
                } else if (target === 'answer') {
                    reasoningText.classList.remove('streaming');
                    diagnosisText.classList.remove('streaming');
                    answerText.classList.add('streaming');
                } else {
                    reasoningText.classList.remove('streaming');
                    diagnosisText.classList.remove('streaming');
                    answerText.classList.remove('streaming');
                }
            }

            function maybeFinishTyping() {
                if (!streamDone) return;
                if (pendingReasoning || pendingDiagnosis || pendingAnswer) return;
                if (flushRaf) return;
                setCursorTarget('none');
                diagnosisText.textContent = String(diagnosisText.textContent || '').replace(/^\s+/, '');
                answerText.textContent = String(answerText.textContent || '').replace(/^\s+/, '');
                if (!sawAnswer) {
                    answerSection.classList.add('hidden');
                    answerText.textContent = '';
                }
            }

            function flushStep() {
                const step = 48;
                let didWork = false;

                if (pendingReasoning) {
                    setCursorTarget('reasoning');
                    const take = pendingReasoning.slice(0, step);
                    pendingReasoning = pendingReasoning.slice(step);
                    reasoningBuf += take;
                    reasoningText.textContent = reasoningBuf;
                    didWork = true;
                } else if (pendingAnswer) {
                    setCursorTarget('answer');
                    if (answerSection.classList.contains('hidden')) {
                         answerSection.classList.remove('hidden');
                    }
                    if (answerPlaceholderShown) {
                        answerPlaceholderShown = false;
                        answerBuf = '';
                        answerText.textContent = '';
                    }
                    const take = pendingAnswer.slice(0, step);
                    pendingAnswer = pendingAnswer.slice(step);
                    answerBuf += take;
                    answerText.textContent = answerBuf;
                    didWork = true;
                } else if (!streamState.inThink && !streamState.inAnswer && pendingDiagnosis) {
                    setCursorTarget('diagnosis');
                    const take = pendingDiagnosis.slice(0, step);
                    pendingDiagnosis = pendingDiagnosis.slice(step);
                    contentBuf += take;
                    diagnosisText.textContent = contentBuf.replace(/^\\s+/, '');
                    diagnosisStarted = diagnosisStarted || Boolean(diagnosisText.textContent.trim());
                    didWork = true;
                } else if (streamState.inThink) {
                    setCursorTarget('reasoning');
                } else if (streamState.inAnswer) {
                    setCursorTarget('answer');
                } else if (diagnosisStarted) {
                    setCursorTarget('diagnosis');
                }

                if (pendingReasoning || pendingDiagnosis || pendingAnswer) {
                    flushRaf = requestAnimationFrame(flushStep);
                    return;
                }

                flushRaf = null;
                maybeFinishTyping();
            }

            function scheduleFlush() {
                if (flushRaf) return;
                flushRaf = requestAnimationFrame(flushStep);
            }

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
                reasoningText.textContent = 'Generating...';
            }

            function takeCarryPrefix(text) {
                const candidates = ['<think>', '</think>', '<answer>', '</answer>'];
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
                    const thinkOpen = lower.indexOf('<think>');
                    const thinkClose = lower.indexOf('</think>');
                    const ansOpen = lower.indexOf('<answer>');
                    const ansClose = lower.indexOf('</answer>');

                    // Find the earliest significant tag
                    let tag = null;
                    let tagIdx = -1;
                    
                    // Helper to check tags
                    const check = (t, idx) => {
                         if (idx !== -1 && (tagIdx === -1 || idx < tagIdx)) {
                             tag = t;
                             tagIdx = idx;
                         }
                    };

                    if (!streamState.inThink && !streamState.inAnswer) {
                        check('<think>', thinkOpen);
                        check('<answer>', ansOpen);
                    } else if (streamState.inThink) {
                        check('</think>', thinkClose);
                    } else if (streamState.inAnswer) {
                        check('</answer>', ansClose);
                    }

                    if (tag === null) {
                        // No tag found, append all to current state target
                        if (streamState.inThink) {
                            pendingReasoning += text;
                        } else if (streamState.inAnswer) {
                            pendingAnswer += text;
                        } else {
                            pendingDiagnosis += text;
                        }
                        scheduleFlush();
                        return;
                    }

                    // Process up to tag
                    const before = text.slice(0, tagIdx);
                    if (before) {
                        if (streamState.inThink) {
                            pendingReasoning += before;
                        } else if (streamState.inAnswer) {
                            pendingAnswer += before;
                        } else {
                            pendingDiagnosis += before;
                        }
                        scheduleFlush();
                    }

                    // Switch state
                    if (tag === '<think>') {
                        streamState.inThink = true;
                        pendingReasoning += '<think>'; // Keep tag visible in reasoning? Usually yes or we hide it.
                                                       // Per requirement, we show <think> tags in reasoning area? 
                                                       // Current impl was appending raw text. 
                                                       // Let's keep consistency.
                    } else if (tag === '</think>') {
                        streamState.inThink = false;
                        pendingReasoning += '</think>';
                    } else if (tag === '<answer>') {
                        streamState.inAnswer = true;
                        sawAnswer = true;
                        // Don't show <answer> tag in UI, just switch buffer
                    } else if (tag === '</answer>') {
                        streamState.inAnswer = false;
                        // Don't show </answer> tag in UI
                    }

                    scheduleFlush();
                    text = text.slice(tagIdx + tag.length);
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
                        streamDone = true;
                        scheduleFlush();
                        maybeFinishTyping();
                        return;
                    }
                }
            }

            let pollPromise = null;
            if (requestIdHeader) {
                setTimeout(() => {
                    if (!gotAnyChunk) {
                        usePolling = true;
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

                        if (usePolling && (eventType === 'reasoning' || eventType === 'content')) {
                            continue;
                        }

                        if (eventType === 'reasoning') {
                            revealIfNeeded();
                            if (placeholderShown) {
                                reasoningBuf = '';
                                pendingReasoning = '';
                                reasoningText.textContent = '';
                                placeholderShown = false;
                            }
                            pendingReasoning += String(payload);
                            scheduleFlush();
                            gotAnyChunk = true;
                        } else if (eventType === 'content') {
                            revealIfNeeded();
                            if (placeholderShown) {
                                reasoningBuf = '';
                                pendingReasoning = '';
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
                                reasoningText.textContent = `Generating... (${sec}s)`;
                            }
                        } else if (eventType === 'done') {
                            if (payload && payload.request_id) currentRequestId = payload.request_id;
                            if (!currentRequestId && requestIdHeader) currentRequestId = requestIdHeader;
                            if (!currentRequestId && requestIdValue && requestIdValue.textContent) {
                                const value = String(requestIdValue.textContent || '').trim();
                                if (value && value !== '--') currentRequestId = value;
                            }
                            if (requestIdValue) requestIdValue.textContent = currentRequestId || (requestIdHeader || '--');
                            revealIfNeeded();
                            if (!diagnosisText.textContent && !reasoningText.textContent) {
                                diagnosisText.textContent = 'Model returned no content';
                            }
                            isAnalysisComplete = true;
                            streamDone = true;
                            scheduleFlush();
                            maybeFinishTyping();
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
