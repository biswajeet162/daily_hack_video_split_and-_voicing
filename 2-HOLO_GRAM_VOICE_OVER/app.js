const processBtn = document.getElementById("processBtn");
const videoFileInput = document.getElementById("videoFileInput");
const chunkFilesInput = document.getElementById("chunkFilesInput");
const chunkSizeInput = document.getElementById("chunkSizeInput");
const loadChunksBtn = document.getElementById("loadChunksBtn");
const selectedChunksLabel = document.getElementById("selectedChunksLabel");
const modeSplitBtn = document.getElementById("modeSplitBtn");
const modeChunksBtn = document.getElementById("modeChunksBtn");
const splitModeControls = document.getElementById("splitModeControls");
const chunksModeControls = document.getElementById("chunksModeControls");
const chunkListTitle = document.getElementById("chunkListTitle");
const micSelect = document.getElementById("micSelect");
const refreshMicBtn = document.getElementById("refreshMicBtn");
const selectedMicLabel = document.getElementById("selectedMicLabel");
const mergeBtn = document.getElementById("mergeBtn");
const startRecordingBtn = document.getElementById("startRecordingBtn");
const stopRecordingBtn = document.getElementById("stopRecordingBtn");
const playRecordingBtn = document.getElementById("playRecordingBtn");
const downloadRecordingLink = document.getElementById("downloadRecordingLink");
const downloadFinalBtn = document.getElementById("downloadFinalBtn");
const bgMusicCheckbox = document.getElementById("bgMusicCheckbox");
const bgMusicPickLabel = document.getElementById("bgMusicPickLabel");
const bgMusicFileInput = document.getElementById("bgMusicFileInput");
const bgMusicName = document.getElementById("bgMusicName");
const bgMusicLevelWrap = document.getElementById("bgMusicLevelWrap");
const bgMusicLevelInput = document.getElementById("bgMusicLevelInput");
const storyVideo = document.getElementById("storyVideo");

const statusText = document.getElementById("statusText");
const chunkList = document.getElementById("chunkList");

const activeChunkPanel = document.getElementById("activeChunkPanel");
const activeChunkId = document.getElementById("activeChunkId");
const activeChunkTime = document.getElementById("activeChunkTime");
const activeRemaining = document.getElementById("activeRemaining");
const activeChunkText = document.getElementById("activeChunkText");
const activeProgress = document.getElementById("activeProgress");

let transcript = [];
let currentChunkIndex = -1;
let mediaRecorder = null;
let recordingStream = null;
let recordingChunks = [];
let recordedByChunk = new Map();
let recordedBlobByChunk = new Map();
let timerInterval = null;
let autoStopTimer = null;
let recordingStartTime = 0;
let previewAudio = null;
let selectedVideoFile = null;
let selectedChunkFiles = [];
let appMode = "split"; // "split" | "chunks"
let transcribingChunkId = null;
let selectedBgMusicFile = null;
let bgMusicUploaded = false;
let finalReady = false;

const MIC_STORAGE_KEY = "selectedMicId";
const PREFERRED_MIC_PATTERNS = [/pd200xs/i, /pd200x/i, /pd200/i, /maono/i];
let micAccessReady = false;
let micInitPromise = null;

function setStatus(message) {
    statusText.textContent = message;
}

function updateSelectedMicLabel() {
    if (!selectedMicLabel) {
        return;
    }
    const option = micSelect.selectedOptions[0];
    if (option && option.value) {
        selectedMicLabel.textContent = `Using: ${option.textContent}`;
    } else {
        selectedMicLabel.textContent = "";
    }
}

function isLikelyBuiltInMic(label) {
    const text = (label || "").toLowerCase();
    return /array|realtek|built-?in|internal|webcam|laptop|default -|communications -/i.test(text);
}

function isLikelyExternalMic(label) {
    const text = (label || "").toLowerCase();
    if (isLikelyBuiltInMic(label)) {
        return false;
    }
    return /usb|external|blue|yeti|rode|shure|audio interface|headset|xm|fifine|maono|pd200|hyperx|elgato|podcast|condenser/i.test(text);
}

function findPreferredMicByPattern(mics) {
    for (const pattern of PREFERRED_MIC_PATTERNS) {
        const match = mics.find((mic) => pattern.test(mic.label || ""));
        if (match) {
            return match;
        }
    }
    return null;
}

function micDisplayLabel(mic, index) {
    if (mic.label) {
        return mic.label;
    }
    const suffix = mic.deviceId ? ` (${mic.deviceId.slice(0, 8)}...)` : "";
    return `Microphone ${index + 1}${suffix}`;
}

function listMicrophones(mics) {
    const seen = new Set();
    const listed = [];

    mics.forEach((mic) => {
        if (!mic.deviceId || seen.has(mic.deviceId)) {
            return;
        }
        seen.add(mic.deviceId);
        listed.push(mic);
    });

    listed.sort((a, b) => {
        const aPreferred = findPreferredMicByPattern([a]) ? 0 : 1;
        const bPreferred = findPreferredMicByPattern([b]) ? 0 : 1;
        if (aPreferred !== bPreferred) {
            return aPreferred - bPreferred;
        }

        const aBuiltIn = isLikelyBuiltInMic(a.label) ? 1 : 0;
        const bBuiltIn = isLikelyBuiltInMic(b.label) ? 1 : 0;
        if (aBuiltIn !== bBuiltIn) {
            return aBuiltIn - bBuiltIn;
        }

        return micDisplayLabel(a, 0).localeCompare(micDisplayLabel(b, 0));
    });

    return listed;
}

function pickPreferredMic(mics) {
    if (!mics.length) {
        return null;
    }

    const brandedMic = findPreferredMicByPattern(mics);
    if (brandedMic) {
        return brandedMic;
    }

    const savedId = localStorage.getItem(MIC_STORAGE_KEY);
    if (savedId) {
        const saved = mics.find((mic) => mic.deviceId === savedId);
        if (saved && !isLikelyBuiltInMic(saved.label)) {
            return saved;
        }
    }

    const externalMics = mics.filter((mic) => isLikelyExternalMic(mic.label));
    if (externalMics.length === 1) {
        return externalMics[0];
    }
    if (externalMics.length > 1) {
        return externalMics.find((mic) => !isLikelyBuiltInMic(mic.label)) || externalMics[0];
    }

    const nonBuiltIn = mics.filter((mic) => !isLikelyBuiltInMic(mic.label));
    if (nonBuiltIn.length === 1) {
        return nonBuiltIn[0];
    }
    if (nonBuiltIn.length > 1) {
        return nonBuiltIn[0];
    }

    if (mics.length >= 2) {
        const builtIn = mics.find((mic) => isLikelyBuiltInMic(mic.label));
        if (builtIn) {
            const alternate = mics.find((mic) => mic.deviceId !== builtIn.deviceId);
            if (alternate) {
                return alternate;
            }
        }
    }

    return mics[0];
}

function populateMicSelect(mics, preferredMic) {
    micSelect.innerHTML = "";

    if (!mics.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No microphone found";
        micSelect.appendChild(option);
        updateSelectedMicLabel();
        return false;
    }

    mics.forEach((mic, index) => {
        const option = document.createElement("option");
        option.value = mic.deviceId;
        option.textContent = micDisplayLabel(mic, index);
        micSelect.appendChild(option);
    });

    const preferred = preferredMic || pickPreferredMic(mics);
    if (preferred) {
        micSelect.value = preferred.deviceId;
        localStorage.setItem(MIC_STORAGE_KEY, preferred.deviceId);
    }

    updateSelectedMicLabel();
    return true;
}

async function enumerateAudioInputs() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === "audioinput");
}

async function discoverMicrophones() {
    let mics = await enumerateAudioInputs();

    if (!mics.some((mic) => mic.label)) {
        await new Promise((resolve) => setTimeout(resolve, 250));
        mics = await enumerateAudioInputs();
    }

    // Probe each input once so Chrome/Windows exposes every connected mic in the list.
    for (const mic of mics) {
        if (!mic.deviceId) {
            continue;
        }
        try {
            const probeStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: { exact: mic.deviceId },
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });
            probeStream.getTracks().forEach((track) => track.stop());
        } catch {
            // Device may be busy or unavailable; keep it in the list anyway.
        }
    }

    await new Promise((resolve) => setTimeout(resolve, 150));
    mics = await enumerateAudioInputs();
    return listMicrophones(mics);
}

async function ensureMicrophones(forcePermission = false) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        setStatus("This browser does not support microphone selection.");
        return false;
    }

    if (micAccessReady && !forcePermission && micSelect.options.length > 1 && micSelect.value) {
        return true;
    }

    if (micInitPromise && !forcePermission) {
        return micInitPromise;
    }

    micInitPromise = (async () => {
        try {
            if (!micAccessReady || forcePermission) {
                // Ask once for mic permission so Windows exposes device names.
                const tempStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                    },
                });
                tempStream.getTracks().forEach((track) => track.stop());
                micAccessReady = true;
            }

            const mics = await discoverMicrophones();
            const preferredMic = pickPreferredMic(mics);
            const populated = populateMicSelect(mics, preferredMic);

            if (!populated) {
                setStatus("No microphone found. Check Windows Sound settings.");
                return false;
            }

            const label = preferredMic?.label || micSelect.selectedOptions[0]?.textContent || "Microphone";
            const branded = findPreferredMicByPattern(mics);
            const micNames = mics.map((mic, index) => micDisplayLabel(mic, index)).join(" | ");
            if (branded) {
                setStatus(`Found ${mics.length} mic(s): ${micNames}. Using ${label}.`);
            } else {
                setStatus(`Found ${mics.length} mic(s): ${micNames}. Select your PD200XS if needed.`);
            }
            return true;
        } catch (error) {
            micAccessReady = false;
            setStatus(`Allow microphone access in browser popup. ${error.message}`);
            return false;
        } finally {
            micInitPromise = null;
        }
    })();

    return micInitPromise;
}

function getAudioConstraints() {
    const deviceId = micSelect.value;
    const audio = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
    };

    if (deviceId) {
        audio.deviceId = { exact: deviceId };
    }

    return { audio };
}

async function openRecordingStream() {
    try {
        return await navigator.mediaDevices.getUserMedia(getAudioConstraints());
    } catch (error) {
        const deviceId = micSelect.value;
        if (!deviceId || error.name !== "OverconstrainedError") {
            throw error;
        }

        return navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: { ideal: deviceId },
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            },
        });
    }
}

function formatChunkVideoPath(chunkId) {
    const padded = String(chunkId).padStart(3, "0");
    return `./split_videos/chunk_${padded}.mp4`;
}

function stopPlayback() {
    if (previewAudio) {
        previewAudio.pause();
        previewAudio.currentTime = 0;
        previewAudio = null;
    }
    storyVideo.pause();
}

function startTimer(durationSeconds, autoStopAfterSeconds = durationSeconds) {
    clearInterval(timerInterval);
    clearTimeout(autoStopTimer);

    recordingStartTime = performance.now();
    activeProgress.style.width = "0%";
    activeRemaining.textContent = `Remaining: ${durationSeconds.toFixed(1)}s`;

    timerInterval = setInterval(() => {
        const elapsed = (performance.now() - recordingStartTime) / 1000;
        const progress = Math.min(100, (elapsed / durationSeconds) * 100);
        const remaining = Math.max(0, durationSeconds - elapsed);

        activeProgress.style.width = `${progress}%`;
        activeRemaining.textContent = `Remaining: ${remaining.toFixed(1)}s`;

        if (progress >= 100) {
            clearInterval(timerInterval);
        }
    }, 80);

    autoStopTimer = setTimeout(() => {
        stopCurrentRecording();
    }, autoStopAfterSeconds * 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
    clearTimeout(autoStopTimer);
}

function markChunkAsRecorded(card) {
    if (!card) {
        return;
    }
    card.classList.add("recorded");
    const actions = card.querySelector(".chunk-actions");
    if (!actions || actions.querySelector(".done-badge")) {
        return;
    }
    const badge = document.createElement("span");
    badge.className = "done-badge";
    badge.textContent = "Done";
    actions.appendChild(badge);
}

function updateTopActionsForChunk(index) {
    const hasRecording = recordedByChunk.has(index);
    startRecordingBtn.disabled = index < 0;
    if (hasRecording) {
        playRecordingBtn.classList.remove("hidden");
        playRecordingBtn.disabled = false;
    } else {
        playRecordingBtn.classList.add("hidden");
        playRecordingBtn.disabled = true;
    }
}

function updateMergeState() {
    mergeBtn.disabled = !transcript.length;
}

function updateDownloadFinalState() {
    if (!downloadFinalBtn) {
        return;
    }
    downloadFinalBtn.disabled = !finalReady;
}

function resetFinalDownloadUi() {
    finalReady = false;
    updateDownloadFinalState();
}

function updateBgMusicUi() {
    const enabled = Boolean(bgMusicCheckbox?.checked);
    bgMusicPickLabel?.classList.toggle("hidden", !enabled);
    bgMusicLevelWrap?.classList.toggle("hidden", !enabled);
    if (!enabled) {
        if (bgMusicName) {
            bgMusicName.textContent = "";
        }
        return;
    }
    if (bgMusicName) {
        bgMusicName.textContent = selectedBgMusicFile ? selectedBgMusicFile.name : "Select a music file";
    }
}

function getBgMusicLevelPercent() {
    const raw = Number.parseFloat(bgMusicLevelInput?.value || "20");
    if (!Number.isFinite(raw)) {
        return 20;
    }
    return Math.max(1, Math.min(100, Math.round(raw)));
}

function updateActiveCardUI() {
    const cards = chunkList.querySelectorAll(".chunk-card");
    cards.forEach((card, idx) => {
        card.classList.toggle("active", idx === currentChunkIndex);
    });
}

function resetSessionUi() {
    recordedByChunk.clear();
    recordedBlobByChunk.clear();
    chunkList.innerHTML = "";
    activeChunkPanel.classList.add("hidden");
    currentChunkIndex = -1;
    transcript = [];
    resetFinalDownloadUi();
    updateMergeState();
}

function setAppMode(mode) {
    appMode = mode === "chunks" ? "chunks" : "split";
    document.body.dataset.mode = appMode;
    modeSplitBtn.classList.toggle("active", appMode === "split");
    modeChunksBtn.classList.toggle("active", appMode === "chunks");
    splitModeControls.classList.toggle("hidden", appMode !== "split");
    chunksModeControls.classList.toggle("hidden", appMode !== "chunks");
    chunkListTitle.textContent = appMode === "chunks" ? "Video Parts" : "Chunk Tabs";

    if (appMode === "split") {
        setStatus("Option 1: select a video, set chunk size, then Process Video.");
    } else {
        setStatus("Option 2: select multiple videos (Ctrl+Click), they will appear on the left.");
    }
}

function renderChunkCards() {
    chunkList.innerHTML = "";

    transcript.forEach((chunk, index) => {
        const card = document.createElement("article");
        card.className = "chunk-card";
        card.dataset.index = String(index);
        card.draggable = true;
        const duration = Math.max(0, Number(chunk.end) - Number(chunk.start));
        const label = appMode === "chunks" ? "Part" : "Chunk";
        const sourceName = chunk.source_name ? ` · ${chunk.source_name}` : "";
        const previewText = (chunk.text || "").trim() || (appMode === "chunks" ? "Click to transcribe..." : "");

        card.innerHTML = `
            <div class="chunk-head">
                <strong><span class="drag-handle" title="Drag to reorder">⋮⋮</span> ${label} ${chunk.chunk_id}${sourceName}</strong>
                <span>${Number(chunk.start).toFixed(1)}s - ${Number(chunk.end).toFixed(1)}s (${duration.toFixed(1)}s)</span>
            </div>
            <p class="chunk-text">${previewText}</p>
            <div class="chunk-actions">
                <div class="chunk-reorder">
                    <button type="button" class="btn-icon move-up-btn" title="Move up" ${index === 0 ? "disabled" : ""}>↑</button>
                    <button type="button" class="btn-icon move-down-btn" title="Move down" ${index === transcript.length - 1 ? "disabled" : ""}>↓</button>
                </div>
                <span class="chunk-hint">Drag to reorder · Click to open</span>
            </div>
        `;

        card.addEventListener("click", (event) => {
            if (event.target.closest(".chunk-reorder") || dragMoved) {
                dragMoved = false;
                return;
            }
            selectChunk(index);
        });

        const upBtn = card.querySelector(".move-up-btn");
        const downBtn = card.querySelector(".move-down-btn");
        upBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            moveChunk(index, -1);
        });
        downBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            moveChunk(index, 1);
        });

        card.addEventListener("dragstart", (event) => {
            if (reorderBusy || mediaRecorder?.state === "recording") {
                event.preventDefault();
                return;
            }
            dragFromIndex = index;
            dragMoved = false;
            card.classList.add("dragging");
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("text/plain", String(index));
        });

        card.addEventListener("dragend", () => {
            card.classList.remove("dragging");
            chunkList.querySelectorAll(".chunk-card").forEach((el) => el.classList.remove("drag-over"));
            dragFromIndex = -1;
        });

        card.addEventListener("dragover", (event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            card.classList.add("drag-over");
        });

        card.addEventListener("dragleave", () => {
            card.classList.remove("drag-over");
        });

        card.addEventListener("drop", async (event) => {
            event.preventDefault();
            event.stopPropagation();
            card.classList.remove("drag-over");
            const from = dragFromIndex >= 0 ? dragFromIndex : Number(event.dataTransfer.getData("text/plain"));
            const to = index;
            if (!Number.isFinite(from) || from === to) {
                return;
            }
            dragMoved = true;
            await reorderToIndex(from, to);
        });

        if (recordedByChunk.has(index)) {
            markChunkAsRecorded(card);
        }

        chunkList.appendChild(card);
    });
}

let dragFromIndex = -1;
let dragMoved = false;

function remapRecordingsAfterReorder(oldIdsInNewOrder) {
    const nextRecorded = new Map();
    const nextBlobs = new Map();

    oldIdsInNewOrder.forEach((oldChunkId, newIndex) => {
        const oldIndex = transcript.findIndex((item) => Number(item.chunk_id) === Number(oldChunkId));
        if (oldIndex < 0) {
            return;
        }
        if (recordedBlobByChunk.has(oldIndex)) {
            nextBlobs.set(newIndex, recordedBlobByChunk.get(oldIndex));
        }
        if (recordedByChunk.has(oldIndex)) {
            const prev = recordedByChunk.get(oldIndex);
            const newChunkId = newIndex + 1;
            if (typeof prev === "string" && prev.includes("/recordings_fixed/")) {
                nextRecorded.set(
                    newIndex,
                    `/recordings_fixed/chunk_${String(newChunkId).padStart(3, "0")}.wav`,
                );
            } else {
                nextRecorded.set(newIndex, prev);
            }
        }
    });

    recordedByChunk = nextRecorded;
    recordedBlobByChunk = nextBlobs;
}

async function applyChunkOrder(newOrder, nextSelectedIndex) {
    if (reorderBusy || mediaRecorder?.state === "recording") {
        setStatus(mediaRecorder?.state === "recording" ? "Stop recording before reordering." : "Reorder already in progress...");
        return;
    }

    const currentOrder = transcript.map((item) => Number(item.chunk_id));
    if (newOrder.length !== currentOrder.length || newOrder.every((id, i) => id === currentOrder[i])) {
        return;
    }

    reorderBusy = true;
    try {
        setStatus("Reordering video parts...");
        const response = await fetch("/api/reorder-chunks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ order: newOrder }),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "reorder failed");
        }

        remapRecordingsAfterReorder(newOrder);
        transcript = data.transcript;
        renderChunkCards();
        updateMergeState();

        if (nextSelectedIndex >= 0 && nextSelectedIndex < transcript.length) {
            currentChunkIndex = nextSelectedIndex;
            await selectChunk(currentChunkIndex, { skipTranscribeIfCached: true });
        }

        setStatus("Video parts reordered.");
    } catch (error) {
        setStatus(`Reorder failed: ${error.message}`);
    } finally {
        reorderBusy = false;
    }
}

async function moveChunk(index, delta) {
    const target = index + delta;
    if (target < 0 || target >= transcript.length) {
        return;
    }

    const order = transcript.map((item) => Number(item.chunk_id));
    const swapped = order.slice();
    const tmp = swapped[index];
    swapped[index] = swapped[target];
    swapped[target] = tmp;

    let nextSelected = currentChunkIndex;
    if (currentChunkIndex === index) {
        nextSelected = target;
    } else if (currentChunkIndex === target) {
        nextSelected = index;
    }

    await applyChunkOrder(swapped, nextSelected);
}

async function reorderToIndex(fromIndex, toIndex) {
    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) {
        return;
    }
    const order = transcript.map((item) => Number(item.chunk_id));
    const next = order.slice();
    const [moved] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, moved);

    let nextSelected = currentChunkIndex;
    if (currentChunkIndex === fromIndex) {
        nextSelected = toIndex;
    } else if (fromIndex < currentChunkIndex && toIndex >= currentChunkIndex) {
        nextSelected = currentChunkIndex - 1;
    } else if (fromIndex > currentChunkIndex && toIndex <= currentChunkIndex) {
        nextSelected = currentChunkIndex + 1;
    }

    await applyChunkOrder(next, nextSelected);
}

async function loadTranscriptFromFile() {
    setStatus("Loading transcript.json...");
    const response = await fetch("./transcript.json");
    if (!response.ok) {
        throw new Error("Unable to read transcript.json");
    }
    const data = await response.json();
    if (!Array.isArray(data) || data.length === 0) {
        throw new Error("Transcript JSON is empty or invalid.");
    }
    transcript = data;
    renderChunkCards();
    updateMergeState();
}

async function waitForProcessDone() {
    while (true) {
        const progressResp = await fetch("/api/process-progress");
        const progress = await progressResp.json();
        if (progress.error) {
            throw new Error(progress.error);
        }

        if (progress.running) {
            setStatus(progress.message || "Processing...");
            await new Promise((resolve) => setTimeout(resolve, 350));
            continue;
        }

        if (progress.done) {
            setStatus(progress.message || "Pipeline complete.");
            return progress;
        }

        await new Promise((resolve) => setTimeout(resolve, 200));
    }
}

async function processFromBeginning() {
    try {
        if (!selectedVideoFile) {
            setStatus("Select a video first.");
            return;
        }

        const chunkSize = Number.parseInt(chunkSizeInput.value, 10);
        if (!Number.isFinite(chunkSize) || chunkSize <= 0) {
            setStatus("Enter a valid chunk size (seconds).");
            return;
        }

        processBtn.disabled = true;
        mergeBtn.disabled = true;
        resetSessionUi();
        setStatus(`Uploading ${selectedVideoFile.name}...`);

        const uploadResp = await fetch(`/api/upload-video?name=${encodeURIComponent(selectedVideoFile.name)}`, {
            method: "POST",
            headers: {
                "Content-Type": selectedVideoFile.type || "application/octet-stream",
            },
            body: selectedVideoFile,
        });
        if (!uploadResp.ok) {
            const errData = await uploadResp.json().catch(() => ({}));
            throw new Error(errData.error || "video upload failed");
        }

        setStatus(`Starting pipeline with ${chunkSize}s chunks...`);
        const processResp = await fetch(`/api/process-video?chunk_size=${chunkSize}`, { method: "POST" });
        if (!processResp.ok) {
            const errData = await processResp.json().catch(() => ({}));
            throw new Error(errData.error || "pipeline start failed");
        }

        await waitForProcessDone();
        appMode = "split";
        await loadTranscriptFromFile();
        setStatus(`Loaded ${transcript.length} chunks and prepared split videos.`);
    } catch (error) {
        setStatus(`Error: ${error.message}. Make sure python server.py is running.`);
    } finally {
        processBtn.disabled = false;
    }
}

async function loadSelectedChunks() {
    try {
        if (!selectedChunkFiles.length) {
            setStatus("Select one or more video chunks first.");
            return;
        }

        loadChunksBtn.disabled = true;
        mergeBtn.disabled = true;
        resetSessionUi();

        setStatus("Preparing chunk upload...");
        const clearResp = await fetch("/api/clear-chunk-staging", { method: "POST" });
        if (!clearResp.ok) {
            throw new Error("Unable to clear staging folder");
        }

        const stagedFiles = [];
        for (let i = 0; i < selectedChunkFiles.length; i += 1) {
            const file = selectedChunkFiles[i];
            setStatus(`Uploading chunk ${i + 1}/${selectedChunkFiles.length}: ${file.name}`);
            const uploadResp = await fetch(
                `/api/upload-chunk-file?index=${i}&name=${encodeURIComponent(file.name)}`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": file.type || "application/octet-stream",
                    },
                    body: file,
                },
            );
            const uploadData = await uploadResp.json().catch(() => ({}));
            if (!uploadResp.ok || !uploadData.ok) {
                throw new Error(uploadData.error || `upload failed for ${file.name}`);
            }
            stagedFiles.push({ name: uploadData.name, staged: uploadData.staged });
        }

        setStatus("Finalizing loaded chunks...");
        const loadResp = await fetch("/api/load-chunks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ files: stagedFiles }),
        });
        const loadData = await loadResp.json().catch(() => ({}));
        if (!loadResp.ok || !loadData.ok) {
            throw new Error(loadData.error || "load chunks failed");
        }

        await waitForProcessDone();
        appMode = "chunks";
        document.body.dataset.mode = "chunks";
        await loadTranscriptFromFile();
        setStatus(`Loaded ${transcript.length} videos on the left. Click one to transcribe and record.`);
    } catch (error) {
        setStatus(`Error: ${error.message}`);
    } finally {
        loadChunksBtn.disabled = false;
    }
}

function loadChunkVideo(chunk) {
    const videoPath = formatChunkVideoPath(chunk.chunk_id);
    storyVideo.src = `${videoPath}?t=${Date.now()}`;
    storyVideo.currentTime = 0;
    storyVideo.load();
    return videoPath;
}

function fillActiveChunkPanel(chunk) {
    const duration = Math.max(0.1, Number(chunk.end) - Number(chunk.start));
    activeChunkPanel.classList.remove("hidden");
    activeChunkId.textContent = `Part ${chunk.chunk_id}${chunk.source_name ? ` · ${chunk.source_name}` : ""}`;
    activeChunkTime.textContent = `${Number(chunk.start).toFixed(1)}s - ${Number(chunk.end).toFixed(1)}s`;
    activeChunkText.textContent = chunk.text || (appMode === "chunks" ? "Transcribing..." : "");
    activeProgress.style.width = "0%";
    activeRemaining.textContent = `Remaining: ${duration.toFixed(1)}s`;
}

async function ensureChunkTranscription(chunk, index) {
    if (appMode !== "chunks") {
        return chunk;
    }
    if ((chunk.text || "").trim()) {
        return chunk;
    }
    if (transcribingChunkId === chunk.chunk_id) {
        return chunk;
    }

    transcribingChunkId = chunk.chunk_id;
    activeChunkText.textContent = "Transcribing this video part...";
    setStatus(`Transcribing part ${chunk.chunk_id}...`);
    startRecordingBtn.disabled = true;

    try {
        const response = await fetch(`/api/transcribe-chunk?chunk_id=${chunk.chunk_id}`, {
            method: "POST",
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "transcription failed");
        }

        transcript[index] = data.chunk;
        renderChunkCards();
        updateActiveCardUI();
        fillActiveChunkPanel(data.chunk);
        setStatus(`Transcript ready for part ${data.chunk.chunk_id}.`);
        return data.chunk;
    } catch (error) {
        activeChunkText.textContent = `Transcription failed: ${error.message}`;
        setStatus(`Transcription failed for part ${chunk.chunk_id}: ${error.message}`);
        throw error;
    } finally {
        transcribingChunkId = null;
        updateTopActionsForChunk(index);
    }
}

async function selectChunk(index, options = {}) {
    currentChunkIndex = index;
    updateActiveCardUI();
    stopPlayback();

    let chunk = transcript[index];
    const duration = Math.max(0.1, Number(chunk.end) - Number(chunk.start));

    fillActiveChunkPanel(chunk);
    stopRecordingBtn.disabled = true;
    downloadRecordingLink.classList.add("disabled-link");
    downloadRecordingLink.removeAttribute("href");
    downloadRecordingLink.removeAttribute("download");
    updateTopActionsForChunk(index);

    const path = loadChunkVideo(chunk);
    storyVideo.muted = true;
    setStatus(`Selected part ${chunk.chunk_id}. Loaded ${path}`);

    if (!options.skipTranscribeIfCached || !(chunk.text || "").trim()) {
        try {
            chunk = await ensureChunkTranscription(chunk, index);
        } catch {
            return;
        }
    }

    activeRemaining.textContent = `Remaining: ${Math.max(0.1, Number(chunk.end) - Number(chunk.start)).toFixed(1)}s`;
    if (recordedByChunk.has(index)) {
        downloadRecordingLink.href = recordedByChunk.get(index);
        downloadRecordingLink.download = `chunk_${chunk.chunk_id}.wav`;
        downloadRecordingLink.classList.remove("disabled-link");
    }
}

async function startChunkRecording(index) {
    try {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            await stopCurrentRecording();
        }

        const micReady = await ensureMicrophones(true);
        if (!micReady || !micSelect.value) {
            setStatus("Microphone not ready. Allow mic access, then try again.");
            return;
        }

        let chunk = transcript[index];
        if (appMode === "chunks" && !(chunk.text || "").trim()) {
            chunk = await ensureChunkTranscription(chunk, index);
        }

        const duration = Math.max(0.1, Number(chunk.end) - Number(chunk.start));

        stopPlayback();
        loadChunkVideo(chunk);

        recordingStream = await openRecordingStream();
        mediaRecorder = new MediaRecorder(recordingStream);
        recordingChunks = [];

        mediaRecorder.onstart = () => {
            const audioTrack = recordingStream?.getAudioTracks?.()[0];
            const activeMicLabel = audioTrack?.label || micSelect.selectedOptions[0]?.textContent || "Microphone";
            // Start timer + muted video only when the browser confirms recording started.
            storyVideo.muted = true;
            const playPromise = storyVideo.play();
            if (playPromise && typeof playPromise.catch === "function") {
                playPromise.catch(() => {
                    setStatus("Video play blocked, recording continues.");
                });
            }
            setStatus(`Recording part ${chunk.chunk_id} with ${activeMicLabel}...`);
            // Record a bit extra after the timer ends to avoid cutting early.
            startTimer(duration, duration + 0.2);
        };

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordingChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordingChunks, { type: "audio/webm" });
            const url = URL.createObjectURL(blob);
            recordedBlobByChunk.set(index, blob);

            const card = chunkList.querySelector(`.chunk-card[data-index="${index}"]`);
            markChunkAsRecorded(card);
            updateMergeState();

            // Disable play/download until server normalizes to exact chunk duration.
            playRecordingBtn.classList.add("hidden");
            playRecordingBtn.disabled = true;
            downloadRecordingLink.classList.add("disabled-link");
            downloadRecordingLink.removeAttribute("href");
            downloadRecordingLink.removeAttribute("download");
            setStatus(`Uploading + fixing audio for part ${chunk.chunk_id}...`);

            fetch(`/api/upload-recording?chunk_id=${chunk.chunk_id}`, {
                method: "POST",
                headers: {
                    "Content-Type": "audio/webm",
                },
                body: blob,
            })
                .then((r) => r.json())
                .then((data) => {
                    if (data && data.fixed_url) {
                        // Use fixed-duration WAV from server for play/download.
                        recordedByChunk.set(index, data.fixed_url);
                        downloadRecordingLink.href = data.fixed_url;
                        downloadRecordingLink.download = `chunk_${chunk.chunk_id}.wav`;
                        downloadRecordingLink.classList.remove("disabled-link");
                        playRecordingBtn.classList.remove("hidden");
                        playRecordingBtn.disabled = false;
                        const durText = data.fixed_duration ? `${data.fixed_duration.toFixed(2)}s` : `${duration.toFixed(1)}s`;
                        setStatus(`Saved part ${chunk.chunk_id} audio (fixed duration: ${durText}).`);
                    } else if (data && data.normalize_error) {
                        // Fallback to raw WebM if normalize fails.
                        recordedByChunk.set(index, url);
                        downloadRecordingLink.href = url;
                        downloadRecordingLink.download = `chunk_${chunk.chunk_id}.webm`;
                        downloadRecordingLink.classList.remove("disabled-link");
                        playRecordingBtn.classList.remove("hidden");
                        playRecordingBtn.disabled = false;
                        setStatus(`Normalize failed for part ${chunk.chunk_id}: ${data.normalize_error}`);
                    }
                })
                .catch(() => {
                    recordedByChunk.set(index, url);
                    downloadRecordingLink.href = url;
                    downloadRecordingLink.download = `chunk_${chunk.chunk_id}.webm`;
                    downloadRecordingLink.classList.remove("disabled-link");
                    playRecordingBtn.classList.remove("hidden");
                    playRecordingBtn.disabled = false;
                    setStatus(`Upload failed for part ${chunk.chunk_id}. Using raw recording.`);
                });

            stopRecordingBtn.disabled = true;
            startRecordingBtn.disabled = false;

            if (recordingStream) {
                recordingStream.getTracks().forEach((track) => track.stop());
                recordingStream = null;
            }
        };

        mediaRecorder.start();
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
    } catch (error) {
        setStatus(`Recording failed: ${error.message}`);
        stopTimer();
        stopRecordingBtn.disabled = true;
        startRecordingBtn.disabled = false;
    }
}

async function startSelectedChunkRecording() {
    if (currentChunkIndex < 0) {
        setStatus("Select a chunk first.");
        return;
    }

    await startChunkRecording(currentChunkIndex);
}

async function stopCurrentRecording() {
    stopTimer();
    stopPlayback();
    activeRemaining.textContent = "Remaining: 0.0s";
    activeProgress.style.width = "100%";

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    } else {
        stopRecordingBtn.disabled = true;
        startRecordingBtn.disabled = currentChunkIndex < 0;
    }
}

async function playCurrentRecording() {
    if (currentChunkIndex < 0) {
        return;
    }
    const chunk = transcript[currentChunkIndex];
    const url = recordedByChunk.get(currentChunkIndex);
    if (!url) {
        return;
    }

    stopPlayback();
    loadChunkVideo(chunk);
    storyVideo.muted = true;

    previewAudio = new Audio(url);
    previewAudio.currentTime = 0;

    const videoPlayPromise = storyVideo.play();
    if (videoPlayPromise && typeof videoPlayPromise.catch === "function") {
        videoPlayPromise.catch(() => {
            setStatus("Video autoplay blocked; playing audio only.");
        });
    }

    const audioPlayPromise = previewAudio.play();
    if (audioPlayPromise && typeof audioPlayPromise.catch === "function") {
        audioPlayPromise.catch(() => {
            setStatus("Unable to play recorded audio.");
        });
    }

    setStatus(`Playing part ${chunk.chunk_id} preview.`);
}

async function mergeFinalVideo() {
    if (!transcript.length) {
        setStatus("Load videos first.");
        return;
    }

    const doneCount = transcript.reduce((acc, _, idx) => acc + (recordedBlobByChunk.has(idx) ? 1 : 0), 0);
    const keepOriginal = transcript.length - doneCount;

    mergeBtn.disabled = true;
    resetFinalDownloadUi();
    setStatus(
        `Merging... (${doneCount} with your voice, ${keepOriginal} keeping original audio)`,
    );

    try {
        const response = await fetch("/api/merge-final", { method: "POST" });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "merge failed");
        }

        finalReady = true;
        updateDownloadFinalState();
        setStatus(
            `Merged successfully (${doneCount} VO / ${keepOriginal} original). Click Download Final.`,
        );
    } catch (error) {
        setStatus(`Merge failed: ${error.message}`);
    } finally {
        updateMergeState();
    }
}

async function uploadBgMusicIfNeeded() {
    if (!selectedBgMusicFile) {
        throw new Error("Select a background music file first.");
    }
    if (bgMusicUploaded) {
        return;
    }
    setStatus(`Uploading BG music: ${selectedBgMusicFile.name}...`);
    const response = await fetch(
        `/api/upload-bg-music?name=${encodeURIComponent(selectedBgMusicFile.name)}`,
        {
            method: "POST",
            headers: {
                "Content-Type": selectedBgMusicFile.type || "application/octet-stream",
            },
            body: selectedBgMusicFile,
        },
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
        throw new Error(data.error || "BG music upload failed");
    }
    bgMusicUploaded = true;
}

function triggerBrowserDownload(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename || "final_output.mp4";
    document.body.appendChild(link);
    link.click();
    link.remove();
}

async function downloadFinalVideo() {
    if (!finalReady) {
        setStatus("Click Merge first.");
        return;
    }

    const useBg = Boolean(bgMusicCheckbox?.checked);
    downloadFinalBtn.disabled = true;

    try {
        if (useBg) {
            if (!selectedBgMusicFile) {
                setStatus("BG Music is on — choose a music file first.");
                bgMusicFileInput?.click();
                return;
            }
            await uploadBgMusicIfNeeded();
            const level = getBgMusicLevelPercent();
            if (bgMusicLevelInput) {
                bgMusicLevelInput.value = String(level);
            }
            setStatus(`Mixing BG music at ${level}% level (looping to video length)...`);
        } else {
            setStatus("Preparing final download...");
        }

        const level = getBgMusicLevelPercent();
        const response = await fetch(
            `/api/prepare-final-download?bg=${useBg ? "1" : "0"}&level=${level}`,
            { method: "POST" },
        );
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || "prepare download failed");
        }

        triggerBrowserDownload(data.output_url, data.filename || "final_output.mp4");
        setStatus(
            useBg
                ? `Downloaded final video with BG music at ${data.level || level}%.`
                : "Downloaded final video.",
        );
    } catch (error) {
        setStatus(`Download failed: ${error.message}`);
    } finally {
        updateDownloadFinalState();
    }
}

function updateSelectedChunksLabel() {
    if (!selectedChunkFiles.length) {
        selectedChunksLabel.textContent = "Use Ctrl+Click to pick many";
        return;
    }
    if (selectedChunkFiles.length === 1) {
        selectedChunksLabel.textContent = `1 video: ${selectedChunkFiles[0].name}`;
        return;
    }
    selectedChunksLabel.textContent = `${selectedChunkFiles.length} videos selected`;
}

modeSplitBtn.addEventListener("click", () => setAppMode("split"));
modeChunksBtn.addEventListener("click", () => setAppMode("chunks"));

videoFileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    selectedVideoFile = file || null;
    if (selectedVideoFile) {
        setStatus(`Selected video: ${selectedVideoFile.name}`);
    }
});

chunkFilesInput.addEventListener("change", async (event) => {
    const files = event.target.files ? Array.from(event.target.files) : [];
    // Numeric-aware sort so part-01, part-02... stay in order
    selectedChunkFiles = files.slice().sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    updateSelectedChunksLabel();
    if (!selectedChunkFiles.length) {
        return;
    }
    setStatus(`Selected ${selectedChunkFiles.length} video(s). Loading to left panel...`);
    await loadSelectedChunks();
});

processBtn.addEventListener("click", processFromBeginning);
loadChunksBtn.addEventListener("click", loadSelectedChunks);
mergeBtn.addEventListener("click", mergeFinalVideo);
downloadFinalBtn?.addEventListener("click", downloadFinalVideo);

bgMusicCheckbox?.addEventListener("change", () => {
    updateBgMusicUi();
    if (bgMusicCheckbox.checked) {
        if (!selectedBgMusicFile) {
            bgMusicFileInput?.click();
        }
        setStatus("BG Music on — choose a track, then Download Final to mix it.");
    } else {
        setStatus("BG Music off — Download Final will use the merged video only.");
    }
});

bgMusicFileInput?.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    selectedBgMusicFile = file || null;
    bgMusicUploaded = false;
    updateBgMusicUi();
    if (selectedBgMusicFile) {
        setStatus(`BG music selected: ${selectedBgMusicFile.name}`);
    }
});

startRecordingBtn.addEventListener("click", startSelectedChunkRecording);
stopRecordingBtn.addEventListener("click", stopCurrentRecording);
playRecordingBtn.addEventListener("click", playCurrentRecording);

refreshMicBtn?.addEventListener("click", () => ensureMicrophones(true));
micSelect.addEventListener("mousedown", () => {
    if (!micAccessReady || micSelect.options.length <= 1) {
        ensureMicrophones(true);
    }
});
micSelect.addEventListener("change", () => {
    if (micSelect.value) {
        localStorage.setItem(MIC_STORAGE_KEY, micSelect.value);
    }
    updateSelectedMicLabel();
    setStatus(`Selected mic: ${micSelect.selectedOptions[0].textContent}`);
});

function setupMicAutoInit() {
    window.addEventListener("load", () => {
        ensureMicrophones(true);
    });

    document.addEventListener("click", () => ensureMicrophones(true), { once: true });
}

setAppMode(new URLSearchParams(window.location.search).get("mode") === "chunks" ? "chunks" : "split");
updateBgMusicUi();
updateDownloadFinalState();

if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    navigator.mediaDevices.addEventListener("devicechange", () => ensureMicrophones(true));
    setupMicAutoInit();
    setStatus("Detecting microphones... allow access when the browser asks.");
} else {
    setStatus("This browser does not support microphone device selection.");
}
