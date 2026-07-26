const micSelect = document.getElementById("micSelect");
const refreshMicBtn = document.getElementById("refreshMicBtn");
const selectedMicLabel = document.getElementById("selectedMicLabel");
const progressSummary = document.getElementById("progressSummary");
const statusText = document.getElementById("statusText");
const chunkList = document.getElementById("chunkList");

const activeChunkPanel = document.getElementById("activeChunkPanel");
const activeChunkId = document.getElementById("activeChunkId");
const activeChunkTime = document.getElementById("activeChunkTime");
const activeCategory = document.getElementById("activeCategory");
const activeRemaining = document.getElementById("activeRemaining");
const activeDisplayLabel = document.getElementById("activeDisplayLabel");
const activeChunkText = document.getElementById("activeChunkText");
const activeProgress = document.getElementById("activeProgress");
const storyVideo = document.getElementById("storyVideo");

const startRecordingBtn = document.getElementById("startRecordingBtn");
const stopRecordingBtn = document.getElementById("stopRecordingBtn");
const reRecordBtn = document.getElementById("reRecordBtn");
const playRecordingBtn = document.getElementById("playRecordingBtn");
const stopPlaybackBtn = document.getElementById("stopPlaybackBtn");
const doneBtn = document.getElementById("doneBtn");
const downloadOutputLink = document.getElementById("downloadOutputLink");

let clips = [];
let currentClipIndex = -1;
let mediaRecorder = null;
let recordingStream = null;
let recordingChunks = [];
let recordedUrlByClip = new Map();
let hasDraftRecording = new Map();
let timerInterval = null;
let autoStopTimer = null;
let recordingStartTime = 0;
let previewAudio = null;

const MIC_STORAGE_KEY = "videoSplitVoiceoverMicId";
const PREFERRED_MIC_PATTERNS = [/pd200xs/i, /pd200x/i, /pd200/i, /maono/i];
let micAccessReady = false;
let micInitPromise = null;

function setStatus(message) {
    statusText.textContent = message;
}

function updateProgressSummary(doneCount, total) {
    progressSummary.textContent = `${doneCount} / ${total} done`;
}

function updateSelectedMicLabel() {
    const option = micSelect.selectedOptions[0];
    selectedMicLabel.textContent = option && option.value ? `Using: ${option.textContent}` : "";
}

function isLikelyBuiltInMic(label) {
    return /array|realtek|built-?in|internal|webcam|laptop|default -|communications -/i.test(label || "");
}

function isLikelyExternalMic(label) {
    if (isLikelyBuiltInMic(label)) return false;
    return /usb|external|blue|yeti|rode|shure|audio interface|headset|xm|fifine|maono|pd200|hyperx|elgato|podcast|condenser/i.test(label || "");
}

function findPreferredMicByPattern(mics) {
    for (const pattern of PREFERRED_MIC_PATTERNS) {
        const match = mics.find((mic) => pattern.test(mic.label || ""));
        if (match) return match;
    }
    return null;
}

function micDisplayLabel(mic, index) {
    return mic.label || `Microphone ${index + 1}${mic.deviceId ? ` (${mic.deviceId.slice(0, 8)}...)` : ""}`;
}

function listMicrophones(mics) {
    const seen = new Set();
    const listed = [];
    mics.forEach((mic) => {
        if (!mic.deviceId || seen.has(mic.deviceId)) return;
        seen.add(mic.deviceId);
        listed.push(mic);
    });
    listed.sort((a, b) => {
        const aPreferred = findPreferredMicByPattern([a]) ? 0 : 1;
        const bPreferred = findPreferredMicByPattern([b]) ? 0 : 1;
        if (aPreferred !== bPreferred) return aPreferred - bPreferred;
        const aBuiltIn = isLikelyBuiltInMic(a.label) ? 1 : 0;
        const bBuiltIn = isLikelyBuiltInMic(b.label) ? 1 : 0;
        if (aBuiltIn !== bBuiltIn) return aBuiltIn - bBuiltIn;
        return micDisplayLabel(a, 0).localeCompare(micDisplayLabel(b, 0));
    });
    return listed;
}

function pickPreferredMic(mics) {
    if (!mics.length) return null;
    const branded = findPreferredMicByPattern(mics);
    if (branded) return branded;
    const savedId = localStorage.getItem(MIC_STORAGE_KEY);
    if (savedId) {
        const saved = mics.find((mic) => mic.deviceId === savedId);
        if (saved && !isLikelyBuiltInMic(saved.label)) return saved;
    }
    const external = mics.filter((mic) => isLikelyExternalMic(mic.label));
    if (external.length) return external[0];
    const nonBuiltIn = mics.filter((mic) => !isLikelyBuiltInMic(mic.label));
    if (nonBuiltIn.length) return nonBuiltIn[0];
    return mics[0];
}

function populateMicSelect(mics, preferredMic) {
    micSelect.innerHTML = "";
    if (!mics.length) {
        micSelect.innerHTML = '<option value="">No microphone found</option>';
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
    for (const mic of mics) {
        if (!mic.deviceId) continue;
        try {
            const probe = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: { exact: mic.deviceId },
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                },
            });
            probe.getTracks().forEach((track) => track.stop());
        } catch {
            // keep in list
        }
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    return listMicrophones(await enumerateAudioInputs());
}

async function ensureMicrophones(forcePermission = false) {
    if (!navigator.mediaDevices?.enumerateDevices) {
        setStatus("Browser does not support microphone selection.");
        return false;
    }
    if (micAccessReady && !forcePermission && micSelect.options.length > 1 && micSelect.value) {
        return true;
    }
    if (micInitPromise && !forcePermission) return micInitPromise;

    micInitPromise = (async () => {
        try {
            if (!micAccessReady || forcePermission) {
                const temp = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
                });
                temp.getTracks().forEach((track) => track.stop());
                micAccessReady = true;
            }
            const mics = await discoverMicrophones();
            populateMicSelect(mics, pickPreferredMic(mics));
            return true;
        } catch (error) {
            micAccessReady = false;
            setStatus(`Allow microphone access. ${error.message}`);
            return false;
        } finally {
            micInitPromise = null;
        }
    })();
    return micInitPromise;
}

function getAudioConstraints() {
    const audio = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
    if (micSelect.value) audio.deviceId = { exact: micSelect.value };
    return { audio };
}

async function openRecordingStream() {
    try {
        return await navigator.mediaDevices.getUserMedia(getAudioConstraints());
    } catch (error) {
        if (!micSelect.value || error.name !== "OverconstrainedError") throw error;
        return navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: { ideal: micSelect.value },
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            },
        });
    }
}

function stopPlayback() {
    if (previewAudio) {
        previewAudio.pause();
        previewAudio.currentTime = 0;
        previewAudio = null;
    }
    storyVideo.pause();
    stopPlaybackBtn.disabled = true;
}

function startTimer(durationSeconds) {
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
        if (progress >= 100) clearInterval(timerInterval);
    }, 80);

    autoStopTimer = setTimeout(() => stopCurrentRecording(), (durationSeconds + 0.2) * 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
    clearTimeout(autoStopTimer);
}

function markChunkCard(index) {
    const card = chunkList.querySelector(`.chunk-card[data-index="${index}"]`);
    if (!card) return;
    const clip = clips[index];
    card.classList.toggle("recorded", Boolean(clip.done || hasDraftRecording.get(index)));
    const actions = card.querySelector(".chunk-actions");
    if (!actions) return;
    actions.innerHTML = "";
    if (clip.done || hasDraftRecording.get(index)) {
        const badge = document.createElement("span");
        badge.className = clip.done ? "done-badge" : "draft-badge";
        badge.textContent = clip.done ? "Done" : "Recorded";
        actions.appendChild(badge);
    }
}

function updateActionButtons() {
    const clip = clips[currentClipIndex];
    const hasDraft = hasDraftRecording.get(currentClipIndex);
    const isDone = clip?.done;
    const canInteract = currentClipIndex >= 0;

    startRecordingBtn.disabled = !canInteract || mediaRecorder?.state === "recording";
    stopRecordingBtn.disabled = mediaRecorder?.state !== "recording";
    reRecordBtn.disabled = !canInteract || (!hasDraft && !isDone);
    playRecordingBtn.disabled = !canInteract || !recordedUrlByClip.has(currentClipIndex);
    stopPlaybackBtn.disabled = true;
    doneBtn.disabled = !canInteract || !hasDraft;
    downloadOutputLink.classList.toggle("disabled-link", !isDone);
    if (isDone) {
        downloadOutputLink.href = `/api/output-video?clip_id=${encodeURIComponent(clip.clip_id)}`;
        downloadOutputLink.download = clip.part_name;
    } else {
        downloadOutputLink.removeAttribute("href");
        downloadOutputLink.removeAttribute("download");
    }
}

function renderChunkList() {
    chunkList.innerHTML = "";
    let lastGroup = "";

    clips.forEach((clip, index) => {
        if (clip.video_name !== lastGroup) {
            lastGroup = clip.video_name;
            const group = document.createElement("div");
            group.className = "group-title";
            group.textContent = clip.video_name;
            chunkList.appendChild(group);
        }

        const card = document.createElement("article");
        card.className = "chunk-card";
        card.dataset.index = String(index);
        if (clip.done || hasDraftRecording.get(index)) card.classList.add("recorded");
        if (index === currentClipIndex) card.classList.add("active");

        const categories = (clip.category_names_en || clip.categories || []).slice(0, 2).join(", ");
        card.innerHTML = `
            <div class="chunk-head">
                <strong>${clip.part_name}</strong>
                <span>${clip.duration_sec}s</span>
            </div>
            <p class="chunk-meta">${clip.display_label}${categories ? ` · ${categories}` : ""}</p>
            <p class="chunk-text">${clip.text}</p>
            <div class="chunk-actions"></div>
        `;
        card.addEventListener("click", () => selectClip(index));
        chunkList.appendChild(card);
        markChunkCard(index);
    });
}

async function loadClips() {
    const response = await fetch("/api/clips");
    if (!response.ok) throw new Error("Unable to load clips");
    const data = await response.json();
    clips = data.clips || [];
    clips.forEach((clip, index) => {
        if (clip.has_recording) {
            hasDraftRecording.set(index, true);
            recordedUrlByClip.set(index, `/recordings_fixed/${clip.clip_id}.wav`);
        }
    });
    updateProgressSummary(data.done_count || 0, data.total || clips.length);
    renderChunkList();
    if (!clips.length) {
        setStatus("No categorized clips found. Run steps 4 and 5 first.");
    } else {
        setStatus(`Loaded ${clips.length} clips ready for voice-over.`);
    }
}

function loadClipVideo(clip) {
    storyVideo.src = `/api/video?clip_id=${encodeURIComponent(clip.clip_id)}`;
    storyVideo.currentTime = 0;
    storyVideo.load();
}

async function selectClip(index) {
    currentClipIndex = index;
    stopPlayback();
    stopTimer();

    document.querySelectorAll(".chunk-card").forEach((card, idx) => {
        card.classList.toggle("active", idx === index);
    });

    const clip = clips[index];
    activeChunkPanel.classList.remove("hidden");
    activeChunkId.textContent = clip.part_name;
    activeChunkTime.textContent = `${clip.duration_sec}s`;
    activeCategory.textContent = (clip.category_names_en || clip.categories || []).join(" · ") || "Uncategorized";
    activeDisplayLabel.textContent = clip.display_label || "";
    activeChunkText.textContent = clip.text || "";
    activeProgress.style.width = "0%";
    activeRemaining.textContent = `Remaining: ${clip.duration_sec.toFixed(1)}s`;

    loadClipVideo(clip);
    storyVideo.muted = true;
    updateActionButtons();
    setStatus(`Selected ${clip.part_name}. Read the Hindi text and record your voice.`);
}

async function startRecording() {
    if (currentClipIndex < 0) return;
    const clip = clips[currentClipIndex];

    try {
        if (mediaRecorder?.state === "recording") await stopCurrentRecording();
        const micReady = await ensureMicrophones(true);
        if (!micReady || !micSelect.value) {
            setStatus("Microphone not ready.");
            return;
        }

        stopPlayback();
        loadClipVideo(clip);
        recordingStream = await openRecordingStream();
        mediaRecorder = new MediaRecorder(recordingStream);
        recordingChunks = [];

        mediaRecorder.onstart = () => {
            const track = recordingStream?.getAudioTracks?.()[0];
            const micLabel = track?.label || micSelect.selectedOptions[0]?.textContent || "Microphone";
            storyVideo.muted = true;
            storyVideo.play().catch(() => {});
            setStatus(`Recording ${clip.part_name} with ${micLabel}...`);
            startTimer(clip.duration_sec);
        };

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) recordingChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            stopTimer();
            activeRemaining.textContent = "Remaining: 0.0s";
            activeProgress.style.width = "100%";

            const blob = new Blob(recordingChunks, { type: "audio/webm" });
            const response = await fetch(`/api/upload-recording?clip_id=${encodeURIComponent(clip.clip_id)}`, {
                method: "POST",
                headers: { "Content-Type": "audio/webm" },
                body: blob,
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
                setStatus(`Upload failed: ${data.error || "unknown error"}`);
            } else if (data.fixed_url) {
                recordedUrlByClip.set(currentClipIndex, data.fixed_url);
                hasDraftRecording.set(currentClipIndex, true);
                clips[currentClipIndex].has_recording = true;
                clips[currentClipIndex].done = false;
                markChunkCard(currentClipIndex);
                setStatus(`Recording saved for ${clip.part_name}. Click Done when it sounds good.`);
            } else {
                setStatus(`Normalize failed: ${data.normalize_error || "unknown error"}`);
            }

            startRecordingBtn.disabled = false;
            stopRecordingBtn.disabled = true;
            recordingStream?.getTracks().forEach((track) => track.stop());
            recordingStream = null;
            updateActionButtons();
        };

        mediaRecorder.start();
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
    } catch (error) {
        setStatus(`Recording failed: ${error.message}`);
        stopTimer();
        updateActionButtons();
    }
}

async function stopCurrentRecording() {
    stopTimer();
    if (mediaRecorder?.state === "recording") {
        mediaRecorder.stop();
    } else {
        stopRecordingBtn.disabled = true;
        startRecordingBtn.disabled = currentClipIndex >= 0;
    }
}

async function reRecordCurrentClip() {
    if (currentClipIndex < 0) return;
    const clip = clips[currentClipIndex];
    stopPlayback();
    await fetch(`/api/clear-recording?clip_id=${encodeURIComponent(clip.clip_id)}`, { method: "POST" });
    recordedUrlByClip.delete(currentClipIndex);
    hasDraftRecording.delete(currentClipIndex);
    clip.done = false;
    clip.has_recording = false;
    markChunkCard(currentClipIndex);
    updateActionButtons();
    setStatus(`Cleared recording for ${clip.part_name}. Record again.`);
}

async function playCurrentRecording() {
    if (currentClipIndex < 0) return;
    const clip = clips[currentClipIndex];
    const url = recordedUrlByClip.get(currentClipIndex);
    if (!url) return;

    stopPlayback();
    loadClipVideo(clip);
    storyVideo.muted = true;
    storyVideo.play().catch(() => {});

    previewAudio = new Audio(url);
    previewAudio.play().catch(() => setStatus("Unable to play recording."));
    stopPlaybackBtn.disabled = false;
    setStatus(`Playing preview for ${clip.part_name}.`);
}

async function finishCurrentClip() {
    if (currentClipIndex < 0) return;
    const clip = clips[currentClipIndex];
    doneBtn.disabled = true;
    setStatus(`Creating voice-over video for ${clip.part_name}...`);

    const response = await fetch(`/api/finish-clip?clip_id=${encodeURIComponent(clip.clip_id)}`, { method: "POST" });
    const data = await response.json();
    if (!response.ok || !data.ok) {
        setStatus(`Done failed: ${data.error || "unknown error"}`);
        doneBtn.disabled = false;
        return;
    }

    clip.done = true;
    clip.output_video = data.output_video;
    markChunkCard(currentClipIndex);
    updateProgressSummary(clips.filter((item) => item.done).length, clips.length);
    updateActionButtons();
    setStatus(`Saved voice-over clip to output_voiceover_videos/${data.output_video}`);
}

refreshMicBtn.addEventListener("click", () => ensureMicrophones(true));
micSelect.addEventListener("change", () => {
    if (micSelect.value) localStorage.setItem(MIC_STORAGE_KEY, micSelect.value);
    updateSelectedMicLabel();
});
startRecordingBtn.addEventListener("click", startRecording);
stopRecordingBtn.addEventListener("click", stopCurrentRecording);
reRecordBtn.addEventListener("click", reRecordCurrentClip);
playRecordingBtn.addEventListener("click", playCurrentRecording);
stopPlaybackBtn.addEventListener("click", stopPlayback);
doneBtn.addEventListener("click", finishCurrentClip);

if (navigator.mediaDevices?.enumerateDevices) {
    navigator.mediaDevices.addEventListener("devicechange", () => ensureMicrophones(true));
    window.addEventListener("load", async () => {
        await ensureMicrophones(true);
        try {
            await loadClips();
        } catch (error) {
            setStatus(`Error: ${error.message}. Is python 06_voiceover_server.py running?`);
        }
    });
} else {
    setStatus("Browser does not support microphone access.");
}
