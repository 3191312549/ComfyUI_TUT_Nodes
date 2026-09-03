import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_ID = "TUT_AdvancedAudioLoader";
const MIN_WIDTH = 520;
const EDITOR_HEIGHT = 232;
const HANDLE_HIT_RADIUS = 10;

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const finiteNumber = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

function inputViewUrl(value) {
    if (value && typeof value === "object" && value.filename) {
        const query = new URLSearchParams({
            filename: value.filename,
            type: value.type || "input",
            subfolder: value.subfolder || "",
        });
        return api.apiURL(`/view?${query.toString()}`);
    }

    let path = String(value || "").trim().replace(/\s+\[input\]\s*$/i, "");
    path = path.replace(/\\/g, "/");
    const separator = path.lastIndexOf("/");
    const filename = separator >= 0 ? path.slice(separator + 1) : path;
    const subfolder = separator >= 0 ? path.slice(0, separator) : "";
    if (!filename) return null;
    const query = new URLSearchParams({ filename, type: "input", subfolder });
    return api.apiURL(`/view?${query.toString()}`);
}

function formatTime(seconds) {
    const safe = Math.max(0, finiteNumber(seconds));
    const minutes = Math.floor(safe / 60);
    const remainder = safe - minutes * 60;
    return `${minutes}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function writeNativeWidget(node, widget, value) {
    if (!widget) return;
    const next = Math.max(0, finiteNumber(value));
    if (Math.abs(finiteNumber(widget.value) - next) < 1e-7) return;
    const previous = widget.value;
    app.graph?.beforeChange?.();
    try {
        widget.value = next;
        widget.callback?.(next, app.canvas, node);
        node.onWidgetChanged?.(widget.name, next, previous, widget);
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } finally {
        app.graph?.afterChange?.();
    }
}

function installAudioEditor(node) {
    if (node.__tutAudioEditor || typeof node.addDOMWidget !== "function") return;
    const fileWidget = node.widgets?.find((widget) => widget.name === "audio_file");
    const startWidget = node.widgets?.find((widget) => widget.name === "start_time");
    const endWidget = node.widgets?.find((widget) => widget.name === "end_time");
    if (!fileWidget || !startWidget || !endWidget) return;

    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "100%",
        height: `${EDITOR_HEIGHT}px`,
        padding: "5px 8px 7px",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        color: "#ddd",
        font: "12px sans-serif",
    });

    const canvas = document.createElement("canvas");
    canvas.height = 340;
    Object.assign(canvas.style, {
        width: "100%",
        height: "170px",
        border: "1px solid #555",
        borderRadius: "5px",
        background: "#15181d",
        boxSizing: "border-box",
        touchAction: "none",
        cursor: "default",
    });

    const controls = document.createElement("div");
    Object.assign(controls.style, { display: "flex", gap: "6px", height: "30px" });
    const makeButton = (label) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        Object.assign(button.style, {
            flex: "1 1 0",
            minWidth: "0",
            border: "1px solid #666",
            borderRadius: "4px",
            background: "#2b2f36",
            color: "#eee",
            cursor: "pointer",
        });
        return button;
    };
    const uploadButton = makeButton("选择音频文件");
    const playButton = makeButton("播放");
    const stopButton = makeButton("停止");
    const resetButton = makeButton("重置全长");
    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = "audio/*,.wav,.mp3,.flac,.ogg,.oga,.m4a,.aac,.opus";
    uploadInput.hidden = true;
    controls.append(uploadButton, playButton, stopButton, resetButton);
    container.append(canvas, controls, uploadInput);

    const context = canvas.getContext("2d");
    const state = {
        audio: null,
        audioContext: null,
        objectUrl: null,
        abortController: null,
        loadSerial: 0,
        selectedKey: null,
        duration: 0,
        peaks: null,
        message: "请选择或上传音频文件",
        drag: null,
        animationFrame: null,
        disposed: false,
    };

    function selection() {
        const duration = Math.max(0, state.duration);
        const start = clamp(finiteNumber(startWidget.value), 0, duration);
        const rawEnd = finiteNumber(endWidget.value);
        const end = clamp(rawEnd <= 0 ? duration : rawEnd, 0, duration);
        return { start, end };
    }

    function markDirty() {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        draw();
    }

    function selectUploadedFile(value) {
        const previous = fileWidget.value;
        if (previous === value) {
            void loadSelectedFile(true);
            return;
        }
        app.graph?.beforeChange?.();
        try {
            fileWidget.value = value;
            const values = fileWidget.options?.values;
            if (Array.isArray(values) && !values.includes(value)) values.push(value);
            fileWidget.callback?.(value, app.canvas, node);
            node.onWidgetChanged?.(fileWidget.name, value, previous, fileWidget);
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        } finally {
            app.graph?.afterChange?.();
        }
    }

    async function uploadAudioFile(file) {
        if (!file || state.disposed) return;
        uploadButton.disabled = true;
        uploadButton.textContent = "正在加入…";
        state.message = "正在上传音频…";
        markDirty();
        try {
            const body = new FormData();
            body.append("image", file);
            body.append("type", "input");
            const response = await api.fetchApi("/upload/image", { method: "POST", body });
            if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
            const result = await response.json();
            const filename = String(result?.name || "");
            if (!filename) throw new Error("服务器未返回文件名");
            const subfolder = String(result?.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
            selectUploadedFile(subfolder ? `${subfolder}/${filename}` : filename);
        } catch (error) {
            console.warn("TUT 高级音频加载：音频上传失败", error);
            state.message = `音频上传失败：${error?.message || error}`;
            markDirty();
        } finally {
            uploadButton.disabled = false;
            uploadButton.textContent = "选择音频文件";
            uploadInput.value = "";
        }
    }

    function cancelProgressAnimation() {
        if (state.animationFrame !== null) cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
    }

    function animateProgress() {
        cancelProgressAnimation();
        const tick = () => {
            if (state.disposed || !state.audio || state.audio.paused) {
                state.animationFrame = null;
                return;
            }
            draw();
            state.animationFrame = requestAnimationFrame(tick);
        };
        state.animationFrame = requestAnimationFrame(tick);
    }

    function stopPreview(resetToStart = true) {
        const audio = state.audio;
        if (audio) {
            audio.pause();
            if (resetToStart && state.duration > 0) {
                const { start } = selection();
                try { audio.currentTime = start; } catch { /* Metadata may not be ready yet. */ }
            }
        }
        playButton.textContent = "播放";
        markDirty();
    }

    function releaseMedia() {
        cancelProgressAnimation();
        state.loadSerial += 1;
        state.abortController?.abort();
        state.abortController = null;
        if (state.audio) {
            state.audio.pause();
            state.audio.removeAttribute("src");
            state.audio.load();
        }
        state.audio = null;
        if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
        state.objectUrl = null;
        if (state.audioContext && state.audioContext.state !== "closed") {
            state.audioContext.close().catch(() => {});
        }
        state.audioContext = null;
        state.duration = 0;
        state.peaks = null;
        playButton.textContent = "播放";
    }

    function buildPeaks(buffer, bucketCount = 1200) {
        const count = Math.max(1, Math.min(bucketCount, buffer.length));
        const peaks = new Float32Array(count);
        const step = buffer.length / count;
        for (let bucket = 0; bucket < count; bucket++) {
            const from = Math.floor(bucket * step);
            const to = Math.max(from + 1, Math.floor((bucket + 1) * step));
            let peak = 0;
            for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
                const samples = buffer.getChannelData(channel);
                for (let index = from; index < to && index < samples.length; index++) {
                    peak = Math.max(peak, Math.abs(samples[index]));
                }
            }
            peaks[bucket] = peak;
        }
        return peaks;
    }

    function attachAudioEvents(audio, serial) {
        audio.preload = "metadata";
        audio.addEventListener("loadedmetadata", () => {
            if (serial !== state.loadSerial || state.disposed) return;
            if (Number.isFinite(audio.duration) && audio.duration > 0) state.duration = audio.duration;
            markDirty();
        });
        audio.addEventListener("timeupdate", () => {
            if (serial !== state.loadSerial || state.disposed) return;
            const { end } = selection();
            if (!audio.paused && audio.currentTime >= end - 0.01) {
                audio.pause();
                try { audio.currentTime = end; } catch { /* Ignore seek errors. */ }
                playButton.textContent = "播放";
            }
            markDirty();
        });
        audio.addEventListener("play", () => {
            playButton.textContent = "暂停";
            animateProgress();
            markDirty();
        });
        audio.addEventListener("pause", () => {
            cancelProgressAnimation();
            playButton.textContent = "播放";
            markDirty();
        });
        audio.addEventListener("error", () => {
            if (serial !== state.loadSerial || state.disposed) return;
            if (!state.peaks) state.message = "浏览器无法播放或解析此音频；后端仍会尝试加载";
            markDirty();
        });
    }

    async function loadSelectedFile(force = false) {
        const selectedKey = typeof fileWidget.value === "object"
            ? JSON.stringify(fileWidget.value)
            : String(fileWidget.value || "");
        if (!force && selectedKey === state.selectedKey && (state.abortController || state.objectUrl)) return;
        releaseMedia();
        state.selectedKey = selectedKey;
        const url = inputViewUrl(fileWidget.value);
        if (!url) {
            state.message = "请选择或上传音频文件";
            markDirty();
            return;
        }

        const serial = state.loadSerial;
        const controller = new AbortController();
        state.abortController = controller;
        state.message = "正在加载音频…";
        markDirty();
        try {
            const response = await fetch(url, { signal: controller.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const blob = await response.blob();
            if (serial !== state.loadSerial || state.disposed) return;

            state.objectUrl = URL.createObjectURL(blob);
            const audio = new Audio(state.objectUrl);
            state.audio = audio;
            attachAudioEvents(audio, serial);
            audio.load();

            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                state.message = "浏览器不支持波形解析，但仍可尝试播放音频";
                markDirty();
                return;
            }
            try {
                const audioContext = new AudioContextClass();
                state.audioContext = audioContext;
                const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
                if (serial !== state.loadSerial || state.disposed) return;
                state.duration = decoded.duration;
                state.peaks = buildPeaks(decoded);
                state.message = "";
            } catch (error) {
                if (serial !== state.loadSerial || state.disposed) return;
                console.warn("TUT 高级音频加载：波形解析失败", error);
                state.message = "浏览器无法解析波形，但仍可尝试播放；后端会继续加载";
            }
            markDirty();
        } catch (error) {
            if (error?.name === "AbortError" || serial !== state.loadSerial || state.disposed) return;
            console.warn("TUT 高级音频加载：音频下载失败", error);
            state.message = "音频下载失败，请检查文件是否仍在 input 目录";
            markDirty();
        } finally {
            if (serial === state.loadSerial) state.abortController = null;
        }
    }

    function canvasMetrics() {
        const rect = canvas.getBoundingClientRect();
        const displayWidth = Math.max(1, rect.width);
        const displayHeight = Math.max(1, rect.height);
        const ratio = Math.max(1, window.devicePixelRatio || 1);
        const width = Math.round(displayWidth * ratio);
        const height = Math.round(displayHeight * ratio);
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        return { width, height, ratio };
    }

    function draw() {
        if (!context || state.disposed) return;
        const { width, height, ratio } = canvasMetrics();
        const padX = 14 * ratio;
        const top = 26 * ratio;
        const bottom = height - 25 * ratio;
        const graphWidth = Math.max(1, width - padX * 2);
        const graphHeight = Math.max(1, bottom - top);
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#15181d";
        context.fillRect(0, 0, width, height);

        context.strokeStyle = "rgba(255,255,255,.09)";
        context.lineWidth = ratio;
        for (let index = 0; index <= 4; index++) {
            const x = padX + graphWidth * index / 4;
            context.beginPath();
            context.moveTo(x, top);
            context.lineTo(x, bottom);
            context.stroke();
        }

        if (state.peaks?.length) {
            context.strokeStyle = "#65cbed";
            context.lineWidth = Math.max(ratio, graphWidth / state.peaks.length * 0.7);
            const middle = (top + bottom) / 2;
            const amplitude = graphHeight * 0.47;
            context.beginPath();
            for (let index = 0; index < state.peaks.length; index++) {
                const x = padX + graphWidth * index / Math.max(1, state.peaks.length - 1);
                const peak = state.peaks[index] * amplitude;
                context.moveTo(x, middle - peak);
                context.lineTo(x, middle + peak);
            }
            context.stroke();
        } else {
            context.fillStyle = "rgba(225,225,225,.72)";
            context.font = `${12 * ratio}px sans-serif`;
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.fillText(state.message || "正在读取波形…", width / 2, (top + bottom) / 2, graphWidth - 10 * ratio);
        }

        const duration = state.duration;
        if (duration > 0) {
            const { start, end } = selection();
            const startX = padX + graphWidth * start / duration;
            const endX = padX + graphWidth * end / duration;
            context.fillStyle = "rgba(0,0,0,.46)";
            context.fillRect(padX, top, Math.max(0, startX - padX), graphHeight);
            context.fillRect(endX, top, Math.max(0, padX + graphWidth - endX), graphHeight);
            context.fillStyle = "rgba(82,199,238,.10)";
            context.fillRect(startX, top, Math.max(0, endX - startX), graphHeight);

            for (const [x, color] of [[startX, "#7ee787"], [endX, "#ffcc66"]]) {
                context.strokeStyle = color;
                context.fillStyle = color;
                context.lineWidth = 2 * ratio;
                context.beginPath();
                context.moveTo(x, top - 4 * ratio);
                context.lineTo(x, bottom + 4 * ratio);
                context.stroke();
                context.beginPath();
                context.arc(x, top, 5 * ratio, 0, Math.PI * 2);
                context.fill();
            }

            const current = clamp(finiteNumber(state.audio?.currentTime, start), 0, duration);
            const currentX = padX + graphWidth * current / duration;
            context.strokeStyle = "#ffffff";
            context.lineWidth = ratio;
            context.beginPath();
            context.moveTo(currentX, top);
            context.lineTo(currentX, bottom);
            context.stroke();

            context.font = `${11 * ratio}px sans-serif`;
            context.textBaseline = "middle";
            context.fillStyle = "rgba(235,235,235,.88)";
            context.textAlign = "left";
            context.fillText(formatTime(start), padX, 12 * ratio);
            context.textAlign = "center";
            context.fillText(`${formatTime(current)} / ${formatTime(duration)}`, width / 2, 12 * ratio);
            context.textAlign = "right";
            context.fillText(formatTime(end), padX + graphWidth, 12 * ratio);
        }
    }

    function eventTime(event) {
        if (!(state.duration > 0)) return 0;
        const rect = canvas.getBoundingClientRect();
        const x = clamp(event.clientX - rect.left, 14, Math.max(14, rect.width - 14));
        return (x - 14) / Math.max(1, rect.width - 28) * state.duration;
    }

    function dragModeAt(event) {
        if (!(state.duration > 0)) return null;
        const rect = canvas.getBoundingClientRect();
        const graphWidth = Math.max(1, rect.width - 28);
        const pointerX = clamp(event.clientX - rect.left, 14, Math.max(14, rect.width - 14));
        const { start, end } = selection();
        const startX = 14 + graphWidth * start / state.duration;
        const endX = 14 + graphWidth * end / state.duration;

        const handleDistances = [
            ["start", Math.abs(pointerX - startX)],
            ["end", Math.abs(pointerX - endX)],
        ].sort((left, right) => left[1] - right[1]);
        if (handleDistances[0][1] <= HANDLE_HIT_RADIUS) return handleDistances[0][0];

        const time = eventTime(event);
        if (time >= start && time <= end) return "seek";
        return Math.abs(time - start) <= Math.abs(time - end) ? "start" : "end";
    }

    function updateCursor(event) {
        if (state.drag) {
            canvas.style.cursor = state.drag === "seek" ? "grabbing" : "ew-resize";
            return;
        }
        const mode = dragModeAt(event);
        canvas.style.cursor = mode === "seek" ? "grab" : mode ? "ew-resize" : "default";
    }

    function beginDrag(event) {
        if (event.button !== 0 || !(state.duration > 0)) return;
        event.preventDefault();
        state.drag = dragModeAt(event);
        if (!state.drag) return;
        canvas.setPointerCapture?.(event.pointerId);
        updateCursor(event);
        updateDrag(event);
    }

    function updateDrag(event) {
        if (!state.drag) return;
        event.preventDefault();
        const time = eventTime(event);
        const { start, end } = selection();
        if (state.drag === "seek") {
            if (state.audio) {
                try { state.audio.currentTime = clamp(time, start, end); } catch { /* Ignore seek errors. */ }
            }
        } else if (state.drag === "start") {
            writeNativeWidget(node, startWidget, clamp(time, 0, end));
        } else {
            writeNativeWidget(node, endWidget, clamp(time, start, state.duration));
        }
        if (state.drag !== "seek" && state.audio && !state.audio.paused) stopPreview(false);
        updateCursor(event);
        markDirty();
    }

    function finishDrag(event) {
        if (!state.drag) return;
        state.drag = null;
        canvas.releasePointerCapture?.(event.pointerId);
        const rect = canvas.getBoundingClientRect();
        const isInside = event.clientX >= rect.left && event.clientX <= rect.right
            && event.clientY >= rect.top && event.clientY <= rect.bottom;
        if (isInside) updateCursor(event);
        else canvas.style.cursor = "default";
        markDirty();
    }

    function leaveCanvas() {
        if (!state.drag) canvas.style.cursor = "default";
    }

    async function togglePlay() {
        const audio = state.audio;
        if (!audio || !(state.duration > 0)) return;
        if (!audio.paused) {
            audio.pause();
            return;
        }
        const { start, end } = selection();
        if (!(end > start)) {
            state.message = "试听选区不能为空";
            markDirty();
            return;
        }
        if (audio.currentTime < start || audio.currentTime >= end - 0.01) audio.currentTime = start;
        try {
            await audio.play();
        } catch (error) {
            console.warn("TUT 高级音频加载：播放失败", error);
            state.message = "浏览器无法播放此音频；后端仍会尝试加载";
            markDirty();
        }
    }

    const resetSelection = () => {
        stopPreview(false);
        writeNativeWidget(node, startWidget, 0);
        writeNativeWidget(node, endWidget, 0);
        if (state.audio) {
            try { state.audio.currentTime = 0; } catch { /* Ignore seek errors. */ }
        }
        markDirty();
    };
    const stopClick = () => stopPreview(true);
    const chooseFile = () => uploadInput.click();
    const uploadSelection = () => void uploadAudioFile(uploadInput.files?.[0]);

    const wrapWidgetCallback = (widget, callback) => {
        const previous = widget.callback;
        widget.callback = function (...args) {
            const result = previous?.apply(this, args);
            callback();
            return result;
        };
        return () => { widget.callback = previous; };
    };
    const restoreCallbacks = [
        wrapWidgetCallback(fileWidget, () => void loadSelectedFile(true)),
        wrapWidgetCallback(startWidget, markDirty),
        wrapWidgetCallback(endWidget, markDirty),
    ];

    canvas.addEventListener("pointerdown", beginDrag);
    canvas.addEventListener("pointermove", updateDrag);
    canvas.addEventListener("pointermove", updateCursor);
    canvas.addEventListener("pointerup", finishDrag);
    canvas.addEventListener("pointercancel", finishDrag);
    canvas.addEventListener("pointerleave", leaveCanvas);
    playButton.addEventListener("click", togglePlay);
    stopButton.addEventListener("click", stopClick);
    resetButton.addEventListener("click", resetSelection);
    uploadButton.addEventListener("click", chooseFile);
    uploadInput.addEventListener("change", uploadSelection);
    const resizeObserver = typeof ResizeObserver === "function" ? new ResizeObserver(draw) : null;
    resizeObserver?.observe(canvas);

    const domWidget = node.addDOMWidget("tut_audio_editor", "TUT_AUDIO_EDITOR", container, {
        serialize: false,
        hideOnZoom: false,
    });
    domWidget.serialize = false;
    domWidget.computeSize = (width) => [width, EDITOR_HEIGHT];

    const cleanup = () => {
        if (state.disposed) return;
        state.disposed = true;
        releaseMedia();
        resizeObserver?.disconnect();
        canvas.removeEventListener("pointerdown", beginDrag);
        canvas.removeEventListener("pointermove", updateDrag);
        canvas.removeEventListener("pointermove", updateCursor);
        canvas.removeEventListener("pointerup", finishDrag);
        canvas.removeEventListener("pointercancel", finishDrag);
        canvas.removeEventListener("pointerleave", leaveCanvas);
        playButton.removeEventListener("click", togglePlay);
        stopButton.removeEventListener("click", stopClick);
        resetButton.removeEventListener("click", resetSelection);
        uploadButton.removeEventListener("click", chooseFile);
        uploadInput.removeEventListener("change", uploadSelection);
        restoreCallbacks.forEach((restore) => restore());
    };

    node.__tutAudioEditor = { cleanup, reload: loadSelectedFile, draw };
    node.setSize?.([
        Math.max(node.size?.[0] || 0, MIN_WIDTH),
        Math.max(node.size?.[1] || 0, 410),
    ]);
    requestAnimationFrame(() => {
        draw();
        void loadSelectedFile();
    });
}

app.registerExtension({
    name: "TUT_Nodes.AdvancedAudioLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const originalRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installAudioEditor(this);
            return result;
        };

        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                this.__tutAudioEditor?.draw();
                void this.__tutAudioEditor?.reload();
            });
            return result;
        };

        nodeType.prototype.onRemoved = function () {
            this.__tutAudioEditor?.cleanup();
            this.__tutAudioEditor = null;
            return originalRemoved?.apply(this, arguments);
        };
    },
});
