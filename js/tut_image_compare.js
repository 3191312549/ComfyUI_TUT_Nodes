import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_ID = "TUT_ImageCompare";
const MIN_WIDTH = 320;
const MIN_HEIGHT = 300;
const LABEL_HEIGHT = 22;

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

function loadPreview(node, data, name) {
    const url = previewUrl(data);
    if (!url) return { name, state: "missing", image: null };
    const entry = { name, state: "loading", image: new Image() };
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
    ctx.fillText("图像 A", x, top - 5);
    ctx.textAlign = "right";
    ctx.fillText("图像 B", x + width, top - 5);
    ctx.restore();
}

function drawCompare(ctx, node, width, y) {
    const height = Math.max(90, node.size[1] - y - 4);
    const a = node.__tutCompareA;
    const b = node.__tutCompareB;
    node.__tutCompareBounds = null;

    if (!a || !b) {
        drawMessage(ctx, width, y, height, "请执行工作流生成对比预览");
        return;
    }
    if (a.state === "error" || b.state === "error") {
        drawMessage(ctx, width, y, height, "预览图像加载失败");
        return;
    }
    if (a.state === "missing" || b.state === "missing") {
        drawMessage(ctx, width, y, height, a.state === "missing" ? "缺少图像 A" : "缺少图像 B");
        return;
    }
    if (a.state !== "ready" || b.state !== "ready") {
        drawMessage(ctx, width, y, height, "正在加载图像…");
        return;
    }

    const imageAreaHeight = Math.max(68, height - LABEL_HEIGHT);
    const imageAreaTop = y + LABEL_HEIGHT;
    const aspect = a.image.naturalWidth / a.image.naturalHeight;
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
    const split = x + drawWidth * (node.__tutCompareRatio ?? 0.5);
    node.__tutCompareBounds = [x, top, drawWidth, drawHeight];

    drawLabels(ctx, x, top, drawWidth);

    ctx.save();
    ctx.beginPath();
    ctx.rect(x, top, drawWidth, drawHeight);
    ctx.clip();
    ctx.drawImage(b.image, x, top, drawWidth, drawHeight);
    ctx.beginPath();
    ctx.rect(x, top, Math.max(0, split - x), drawHeight);
    ctx.clip();
    ctx.drawImage(a.image, x, top, drawWidth, drawHeight);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(0, 0, 0, 0.65)";
    ctx.shadowBlur = 2;
    ctx.beginPath();
    ctx.moveTo(split, top);
    ctx.lineTo(split, top + drawHeight);
    ctx.stroke();
    ctx.restore();
}

function updateRatio(node, pos) {
    const bounds = node.__tutCompareBounds;
    if (!bounds || !pos) return false;
    const [x, y, width, height] = bounds;
    if (pos[0] < x || pos[0] > x + width || pos[1] < y || pos[1] > y + height) return false;
    node.__tutCompareRatio = Math.max(0, Math.min(1, (pos[0] - x) / width));
    node.setDirtyCanvas?.(true, false);
    return true;
}

app.registerExtension({
    name: "TUT_Nodes.ImageCompare",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalExecuted = nodeType.prototype.onExecuted;
        const originalMouseMove = nodeType.prototype.onMouseMove;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.__tutCompareRatio = 0.5;
            this.addCustomWidget({
                name: "tut_image_compare",
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
            this.__tutCompareRatio = 0.5;
            this.__tutCompareA = loadPreview(this, output?.a_images?.[0], "A");
            this.__tutCompareB = loadPreview(this, output?.b_images?.[0], "B");
            this.setDirtyCanvas?.(true, true);
            return result;
        };

        nodeType.prototype.onMouseMove = function (_event, pos) {
            const result = originalMouseMove?.apply(this, arguments);
            updateRatio(this, pos);
            return result;
        };
    },
});
