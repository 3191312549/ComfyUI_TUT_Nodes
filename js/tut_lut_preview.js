import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_ID = "TUT_LUTLoaderPreview";
const MIN_WIDTH = 320;
const MIN_HEIGHT = 300;
const LABEL_HEIGHT = 22;
const LUT_ACCEPT = ".cube,.3dl,.1dlut,.png";
const LUT_SUBFOLDER = "TUT_Nodes/luts";

function previewUrl(data) {
    if (!data?.filename) return null;
    const query = new URLSearchParams({
        filename: data.filename,
        type: data.type ?? "temp",
        subfolder: data.subfolder ?? "",
    });
    const preview = app.getPreviewFormatParam?.() ?? "";
    const random = app.getRandParam?.() ?? `&rand=${Math.random()}`;
    return api.apiURL(`/view?${query.toString()}${preview}${random}`);
}

function firstValue(value) {
    return Array.isArray(value) ? value[0] : value;
}

function loadPreview(node, data) {
    const url = previewUrl(firstValue(data));
    if (!url) return { state: "missing", image: null };

    const entry = { state: "loading", image: new Image() };
    entry.image.onload = () => {
        entry.state = "ready";
        node.setDirtyCanvas?.(true, true);
    };
    entry.image.onerror = () => {
        entry.state = "error";
        node.setDirtyCanvas?.(true, true);
    };
    entry.image.src = url;
    return entry;
}

function outputMessage(output) {
    const message = firstValue(output?.message);
    return typeof message === "string" && message.trim()
        ? message.trim()
        : "未选择预览图片";
}

function drawMessage(ctx, width, y, height, text) {
    ctx.save();
    ctx.fillStyle = "rgba(20, 20, 20, 0.55)";
    ctx.fillRect(0, y, width, height);
    ctx.fillStyle = "rgba(235, 235, 235, 0.9)";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, width / 2, y + height / 2);
    ctx.restore();
}

function drawLabels(ctx, x, top, width) {
    ctx.save();
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "rgba(205, 205, 205, 0.78)";
    ctx.textBaseline = "bottom";
    ctx.textAlign = "left";
    ctx.fillText("原图", x, top - 5);
    ctx.textAlign = "right";
    ctx.fillText("LUT 效果", x + width, top - 5);
    ctx.restore();
}

function drawCompare(ctx, node, width, y) {
    const height = Math.max(90, node.size[1] - y - 4);
    const original = node.__tutLutOriginal;
    const graded = node.__tutLutGraded;
    node.__tutLutBounds = null;

    if (!original || !graded || original.state === "missing" || graded.state === "missing") {
        drawMessage(ctx, width, y, height, node.__tutLutMessage ?? "请执行工作流生成 LUT 预览");
        return;
    }
    if (original.state === "error" || graded.state === "error") {
        drawMessage(ctx, width, y, height, "预览图像加载失败");
        return;
    }
    if (original.state !== "ready" || graded.state !== "ready") {
        drawMessage(ctx, width, y, height, "正在加载预览图像…");
        return;
    }

    const imageAreaHeight = Math.max(68, height - LABEL_HEIGHT);
    const imageAreaTop = y + LABEL_HEIGHT;
    const aspect = original.image.naturalWidth / original.image.naturalHeight;
    const availableAspect = width / imageAreaHeight;
    let drawWidth = width;
    let drawHeight = imageAreaHeight;
    if (aspect > availableAspect) {
        drawHeight = width / aspect;
    } else {
        drawWidth = imageAreaHeight * aspect;
    }

    const x = (width - drawWidth) / 2;
    const top = imageAreaTop + (imageAreaHeight - drawHeight) / 2;
    const ratio = node.__tutLutRatio ?? 0.5;
    const split = x + drawWidth * ratio;
    node.__tutLutBounds = [x, top, drawWidth, drawHeight];

    drawLabels(ctx, x, top, drawWidth);

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, top, drawWidth, drawHeight);
    ctx.clip();
    ctx.drawImage(graded.image, x, top, drawWidth, drawHeight);
    ctx.beginPath();
    ctx.rect(x, top, Math.max(0, split - x), drawHeight);
    ctx.clip();
    ctx.drawImage(original.image, x, top, drawWidth, drawHeight);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(0, 0, 0, 0.65)";
    ctx.shadowBlur = 2;
    ctx.beginPath();
    ctx.moveTo(split, top);
    ctx.lineTo(split, top + drawHeight);
    ctx.stroke();

    ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
    ctx.beginPath();
    ctx.arc(split, top + drawHeight / 2, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
}

function isInsidePreview(node, pos) {
    const bounds = node.__tutLutBounds;
    if (!bounds || !pos) return false;
    const [x, y, width, height] = bounds;
    return pos[0] >= x && pos[0] <= x + width && pos[1] >= y && pos[1] <= y + height;
}

function updateRatio(node, pos) {
    const bounds = node.__tutLutBounds;
    if (!bounds || !pos) return false;
    const [x, , width] = bounds;
    node.__tutLutRatio = Math.max(0, Math.min(1, (pos[0] - x) / width));
    node.setDirtyCanvas?.(true, false);
    return true;
}

function setUploadStatus(node, button, text, isError = false) {
    button.name = text;
    if (isError) node.__tutLutMessage = text;
    node.setDirtyCanvas?.(true, true);
}

function addComboValue(widget, value) {
    const values = widget?.options?.values;
    if (!Array.isArray(values)) return;
    if (!values.includes(value)) values.push(value);
}

function setComboValue(node, widget, value) {
    const previous = widget.value;
    addComboValue(widget, value);
    widget.value = value;
    widget.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(widget.name, value, previous, widget);
    node.setDirtyCanvas?.(true, true);
}

function uploadToken(result, fallbackName) {
    const name = result?.name || fallbackName;
    const subfolder = result?.subfolder || LUT_SUBFOLDER;
    return [subfolder, name]
        .filter(Boolean)
        .join("/")
        .replaceAll("\\", "/")
        .replace(/^\/+/, "");
}

async function uploadLut(node, combo, button, file) {
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!["cube", "3dl", "1dlut", "png"].includes(extension)) {
        setUploadStatus(node, button, "上传失败：不支持的 LUT 格式", true);
        return;
    }

    setUploadStatus(node, button, "正在上传 LUT…");
    try {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        form.append("subfolder", LUT_SUBFOLDER);

        const response = await api.fetchApi("/upload/image", {
            method: "POST",
            body: form,
        });
        let result = null;
        try {
            result = await response.json();
        } catch {
            // The status text below is more useful than a JSON parsing exception.
        }
        if (!response.ok) {
            const detail = result?.error || result?.message || `${response.status} ${response.statusText}`;
            throw new Error(detail);
        }

        const token = uploadToken(result, file.name);
        setComboValue(node, combo, token);
        setUploadStatus(node, button, "上传 LUT");
        node.__tutLutMessage = `已上传：${token}`;
    } catch (error) {
        const detail = error instanceof Error ? error.message : String(error);
        setUploadStatus(node, button, `上传失败：${detail}`, true);
    }
}

function addLutUpload(node) {
    const combo = node.widgets?.find((widget) => widget.name === "lut_file");
    if (!combo || node.__tutLutFileInput) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = LUT_ACCEPT;
    input.style.display = "none";
    document.body.appendChild(input);
    node.__tutLutFileInput = input;

    const button = node.addWidget("button", "上传 LUT", null, () => input.click());
    input.addEventListener("change", async () => {
        const file = input.files?.[0];
        input.value = "";
        if (file) await uploadLut(node, combo, button, file);
    });
}

app.registerExtension({
    name: "TUT_Nodes.LUTLoaderPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalExecuted = nodeType.prototype.onExecuted;
        const originalMouseDown = nodeType.prototype.onMouseDown;
        const originalMouseMove = nodeType.prototype.onMouseMove;
        const originalMouseUp = nodeType.prototype.onMouseUp;
        const originalRemoved = nodeType.prototype.onRemoved;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.__tutLutRatio = 0.5;
            this.__tutLutDragging = false;
            this.__tutLutMessage = "请执行工作流生成 LUT 预览";
            addLutUpload(this);
            this.addCustomWidget({
                name: "tut_lut_preview",
                type: "custom",
                value: null,
                serializeValue: () => null,
                computeSize: (width) => [width, Math.max(220, width * 0.72)],
                draw: (ctx, node, width, y) => drawCompare(ctx, node, width, y),
            });
            this.setSize?.([
                Math.max(this.size?.[0] ?? 0, MIN_WIDTH),
                Math.max(this.size?.[1] ?? 0, MIN_HEIGHT),
            ]);
            return result;
        };

        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            this.__tutLutRatio = 0.5;
            this.__tutLutDragging = false;
            this.__tutLutMessage = outputMessage(output);
            this.__tutLutOriginal = loadPreview(this, output?.original_images);
            this.__tutLutGraded = loadPreview(this, output?.graded_images);
            this.setDirtyCanvas?.(true, true);
            return result;
        };

        nodeType.prototype.onMouseDown = function (_event, pos) {
            if (isInsidePreview(this, pos)) {
                this.__tutLutDragging = true;
                updateRatio(this, pos);
                return true;
            }
            return originalMouseDown?.apply(this, arguments);
        };

        nodeType.prototype.onMouseMove = function (_event, pos) {
            if (this.__tutLutDragging) {
                updateRatio(this, pos);
                return true;
            }
            return originalMouseMove?.apply(this, arguments);
        };

        nodeType.prototype.onMouseUp = function () {
            const wasDragging = this.__tutLutDragging;
            this.__tutLutDragging = false;
            if (wasDragging) return true;
            return originalMouseUp?.apply(this, arguments);
        };

        nodeType.prototype.onRemoved = function () {
            this.__tutLutFileInput?.remove();
            this.__tutLutFileInput = null;
            return originalRemoved?.apply(this, arguments);
        };
    },
});
