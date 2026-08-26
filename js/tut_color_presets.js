import { app } from "/scripts/app.js";

const COLOR_PRESETS = {
    TUT_FilmTone: {
        "中性电影": { toe: 0.16, shoulder: 0.2, density: 1, saturation_compression: 0.18, temperature: 0, highlight_tint: 0 },
        "暖调印片": { toe: 0.2, shoulder: 0.28, density: 1.04, saturation_compression: 0.22, temperature: 0.28, highlight_tint: 0.3 },
        "冷调惊悚": { toe: 0.28, shoulder: 0.18, density: 0.94, saturation_compression: 0.42, temperature: -0.35, highlight_tint: -0.3 },
        "低饱和剧情": { toe: 0.2, shoulder: 0.3, density: 0.96, saturation_compression: 0.7, temperature: -0.05, highlight_tint: 0.08 },
        "高反差银幕": { toe: 0.42, shoulder: 0.12, density: 1.12, saturation_compression: 0.28, temperature: 0.05, highlight_tint: 0.1 },
    },
    TUT_Halation: {
        "细腻胶片": { highlight_threshold: 0.78, softness: 0.16, radius: 5, spread: 0.4, red_orange_ratio: 0.5, strength: 0.35 },
        "经典35mm": { highlight_threshold: 0.7, softness: 0.2, radius: 9, spread: 0.65, red_orange_ratio: 0.58, strength: 0.58 },
        "强烈红晕": { highlight_threshold: 0.6, softness: 0.22, radius: 16, spread: 1.1, red_orange_ratio: 0.25, strength: 0.85 },
        "柔和暖晕": { highlight_threshold: 0.68, softness: 0.3, radius: 13, spread: 0.7, red_orange_ratio: 0.82, strength: 0.48 },
    },
    TUT_LensDiffusion: {
        "柔光镜": { mode: "柔光镜", radius: 4, highlight_threshold: 0.62, contrast_softening: 0.35, strength: 0.45 },
        "黑柔": { mode: "黑柔", radius: 6, highlight_threshold: 0.58, contrast_softening: 0.5, strength: 0.55 },
        "薄雾": { mode: "薄雾", radius: 12, highlight_threshold: 0.5, contrast_softening: 0.65, strength: 0.5 },
        "梦幻扩散": { mode: "梦幻扩散", radius: 10, highlight_threshold: 0.52, contrast_softening: 0.8, strength: 0.68 },
    },
    TUT_ColorCompressor: {
        "青橙聚合": { target_color: "#278E91", hue_range: 105, saturation_limit: 0.78, preserve_luminance: true, protect_skin: true, strength: 0.62 },
        "暖棕电影": { target_color: "#A4633A", hue_range: 125, saturation_limit: 0.58, preserve_luminance: true, protect_skin: false, strength: 0.58 },
        "冷蓝夜景": { target_color: "#315A8C", hue_range: 120, saturation_limit: 0.7, preserve_luminance: true, protect_skin: true, strength: 0.72 },
        "单色海报": { target_color: "#C7374A", hue_range: 180, saturation_limit: 0.9, preserve_luminance: true, protect_skin: false, strength: 0.9 },
        "柔和肤色": { target_color: "#C88162", hue_range: 65, saturation_limit: 0.48, preserve_luminance: true, protect_skin: false, strength: 0.42 },
    },
};

function applyPreset(node, presetName) {
    const values = COLOR_PRESETS[node.comfyClass]?.[presetName];
    if (!values) return false;
    for (const [name, value] of Object.entries(values)) {
        const widget = node.widgets?.find((candidate) => candidate.name === name);
        if (widget) widget.value = value;
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
}

app.registerExtension({
    name: "TUT_Nodes.ColorPresets",
    nodeCreated(node) {
        if (!COLOR_PRESETS[node.comfyClass]) return;
        const presetWidget = node.widgets?.find((widget) => widget.name === "preset");
        if (!presetWidget) return;
        const originalCallback = presetWidget.callback;
        presetWidget.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            applyPreset(node, value);
            return result;
        };
    },
});
