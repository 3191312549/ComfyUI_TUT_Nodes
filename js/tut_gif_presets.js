import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const GIF_NODE_ID = "TUT_SaveAnimatedGIF";

function gifPreviewUrl(data) {
    if (!data?.filename) return null;
    const query = new URLSearchParams({
        filename: data.filename,
        type: data.type ?? "output",
        subfolder: data.subfolder ?? "",
        rand: String(Math.random()),
    });
    // Do not append ComfyUI's preview-format parameter: it can flatten an
    // animated GIF into a static preview frame.
    return api.apiURL(`/view?${query.toString()}`);
}

function installGifPreview(node) {
    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "100%",
        height: "100%",
        minHeight: "120px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        boxSizing: "border-box",
        borderRadius: "8px",
        background: "rgba(0, 0, 0, 0.28)",
        color: "rgba(220, 220, 220, 0.72)",
        font: "13px sans-serif",
    });

    const message = document.createElement("span");
    message.textContent = "执行后显示 GIF 动画预览";

    const image = document.createElement("img");
    image.alt = "GIF 动画预览";
    image.draggable = false;
    Object.assign(image.style, {
        width: "100%",
        height: "100%",
        display: "none",
        objectFit: "contain",
    });
    image.onload = () => {
        message.style.display = "none";
        image.style.display = "block";
        node.setDirtyCanvas?.(true, true);
    };
    image.onerror = () => {
        image.style.display = "none";
        message.style.display = "inline";
        message.textContent = "GIF 动画预览加载失败";
        node.setDirtyCanvas?.(true, true);
    };

    container.append(message, image);
    const widget = node.addDOMWidget("tut_gif_preview", "gif_preview", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 120,
        getMaxHeight: () => Math.max(120, node.size?.[1] ?? 680),
    });
    widget.serialize = false;
    node.__tutGifPreviewImage = image;
    node.__tutGifPreviewMessage = message;
    node.hideOutputImages = true;
    node.setSize?.([
        Math.max(node.size?.[0] ?? 0, 340),
        Math.max(node.size?.[1] ?? 0, 680),
    ]);
}

function updateGifPreview(node, output) {
    const image = node.__tutGifPreviewImage;
    const message = node.__tutGifPreviewMessage;
    if (!image || !message) return;

    const preview = output?.gif_preview?.[0] ?? output?.images?.[0];
    const url = gifPreviewUrl(preview);
    if (!url) {
        image.style.display = "none";
        message.style.display = "inline";
        message.textContent = "没有收到 GIF 预览文件";
        return;
    }

    message.style.display = "inline";
    message.textContent = "正在加载 GIF 动画…";
    image.style.display = "none";
    image.src = url;
}

export const GIF_COMPRESSION_PRESETS = {
    "高画质": {
        resize_scale: 0.85,
        max_colors: 256,
        frame_step: 1,
        dither: true,
        optimize: true,
    },
    "均衡": {
        resize_scale: 0.75,
        max_colors: 128,
        frame_step: 1,
        dither: false,
        optimize: true,
    },
    "小体积": {
        resize_scale: 0.5,
        max_colors: 64,
        frame_step: 2,
        dither: false,
        optimize: true,
    },
};

export function applyGifCompressionPreset(node, presetName) {
    const values = GIF_COMPRESSION_PRESETS[presetName];
    if (!values) return false;

    app.graph?.beforeChange?.();
    try {
        for (const [name, value] of Object.entries(values)) {
            const widget = node.widgets?.find((candidate) => candidate.name === name);
            if (widget) {
                const previous = widget.value;
                const nextValue = name === "max_colors" ? `${value} 色` : value;
                widget.value = nextValue;
                widget.callback?.(nextValue, app.canvas, node);
                node.onWidgetChanged?.(name, nextValue, previous, widget);
            }
        }
    } finally {
        app.graph?.afterChange?.();
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
}

function installPresetCallback(node) {
    const presetWidget = node.widgets?.find(
        (widget) => widget.name === "compression_preset",
    );
    if (!presetWidget || presetWidget.__tutPresetCallback) return;

    const originalCallback = presetWidget.callback;
    presetWidget.callback = function (value) {
        const result = originalCallback?.apply(this, arguments);
        applyGifCompressionPreset(node, value);
        return result;
    };
    presetWidget.__tutPresetCallback = true;
}

function setComboValue(node, widget, value) {
    if (widget.value === value) return;
    const previous = widget.value;
    app.graph?.beforeChange?.();
    try {
        widget.value = value;
        widget.callback?.(value, app.canvas, node);
        node.onWidgetChanged?.(widget.name, value, previous, widget);
        node.setDirtyCanvas?.(true, true);
    } finally {
        app.graph?.afterChange?.();
    }
}

function openNativeComboPicker(node, widget, event) {
    const configuredValues = widget.options?.values;
    const values = typeof configuredValues === "function"
        ? configuredValues(widget, node)
        : configuredValues;
    const choices = Array.isArray(values) ? values : Object.keys(values || {});
    if (!choices.length) return;

    document.querySelector(".tut-gif-native-picker")?.remove();
    const select = document.createElement("select");
    select.className = "tut-gif-native-picker";
    Object.assign(select.style, {
        position: "fixed",
        left: `${Math.max(8, event?.clientX ?? 8)}px`,
        top: `${Math.max(8, event?.clientY ?? 8)}px`,
        width: "180px",
        height: "28px",
        zIndex: "100000",
        border: "1px solid #777",
        borderRadius: "5px",
        background: "#242424",
        color: "#eee",
        font: "13px sans-serif",
    });
    for (const choice of choices) {
        const option = document.createElement("option");
        option.value = String(choice);
        option.textContent = widget.options?.getOptionLabel?.(String(choice)) ?? String(choice);
        select.append(option);
    }
    select.value = String(widget.value ?? choices[0]);

    let committed = false;
    const cleanup = () => select.remove();
    const commit = () => {
        committed = true;
        setComboValue(node, widget, select.value);
        cleanup();
    };
    select.addEventListener("change", commit, { once: true });
    select.addEventListener("blur", () => {
        setTimeout(() => {
            if (!committed) cleanup();
        }, 0);
    }, { once: true });
    document.body.append(select);
    select.focus({ preventScroll: true });
    try {
        select.showPicker?.();
    } catch {
        select.size = Math.min(choices.length, 10);
    }
}

function installReliableComboPicker(node, widgetName, force = false) {
    const widget = node.widgets?.find((candidate) => candidate.name === widgetName);
    if (!widget || (widget.__tutReliablePicker && !force)) return;
    widget.__tutOriginalOnClick ||= widget.onClick;
    const originalOnClick = widget.__tutOriginalOnClick;
    widget.onClick = function (context) {
        const relativeX = context?.e?.canvasX - node.pos[0];
        const width = this.width || node.size?.[0] || 0;
        if (relativeX < 40 || relativeX > width - 40) {
            return originalOnClick?.call(this, context);
        }
        openNativeComboPicker(node, this, context?.e);
    };
    widget.__tutReliablePicker = true;
}

app.registerExtension({
    name: "TUT_Nodes.GIFCompressionPresets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== GIF_NODE_ID) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            installPresetCallback(this);
            installGifPreview(this);
            return result;
        };

        nodeType.prototype.onExecuted = function (output) {
            const result = originalExecuted?.apply(this, arguments);
            updateGifPreview(this, output);
            return result;
        };
    },

    nodeCreated(node) {
        if (node.comfyClass !== GIF_NODE_ID) return;
        requestAnimationFrame(() => {
            installReliableComboPicker(node, "compression_preset", true);
            installReliableComboPicker(node, "max_colors", true);
        });
    },
});
