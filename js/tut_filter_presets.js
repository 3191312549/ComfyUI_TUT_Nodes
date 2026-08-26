import { app } from "/scripts/app.js";

const FILTER_PRESETS = {
    TUT_RetroPrintFilter: {
        "报纸": { color_mode: "单色", palette_text: "#202020,#E8E2D2", dot_size: 4, screen_angle: 45, registration_shift: 0.5, ink_bleed: 1, paper_grain: 0.35 },
        "双色海报": { color_mode: "双色", palette_text: "#19181C,#E2314D,#F4D64C", dot_size: 8, screen_angle: 15, registration_shift: 1.5, ink_bleed: 1.5, paper_grain: 0.2 },
        "Risograph": { color_mode: "三色", palette_text: "#19181C,#E2314D,#1D91C0,#F4D64C", dot_size: 6, screen_angle: 15, registration_shift: 2.5, ink_bleed: 1, paper_grain: 0.3 },
        "CMYK旧印刷": { color_mode: "CMYK", palette_text: "#111111,#00A8C8,#D62470,#F2D338", dot_size: 5, screen_angle: 15, registration_shift: 2, ink_bleed: 1, paper_grain: 0.25 },
    },
    TUT_ComicFilter: {
        "日漫黑白": { color_levels: 2, line_width: 2, line_strength: 2, line_threshold: 0.15, shadow_threshold: 0.55, shadow_halftone: true, dot_size: 6 },
        "彩色漫画": { color_levels: 6, line_width: 2, line_strength: 1.5, line_threshold: 0.18, shadow_threshold: 0.4, shadow_halftone: true, dot_size: 6 },
        "美漫": { color_levels: 5, line_width: 3, line_strength: 2, line_threshold: 0.14, shadow_threshold: 0.45, shadow_halftone: false, dot_size: 6 },
        "波普": { color_levels: 4, line_width: 2, line_strength: 1.6, line_threshold: 0.2, shadow_threshold: 0.5, shadow_halftone: true, dot_size: 10 },
    },
    TUT_PixelArtFilter: {
        "Game Boy": { pixel_size: 6, max_colors: 4, palette_mode: "Game Boy", dither: "Bayer 4x4", outline_strength: 0.4, outline_threshold: 0.3 },
        "NES风格": { pixel_size: 8, max_colors: 5, palette_mode: "NES风格", dither: "Bayer 2x2", outline_strength: 0.35, outline_threshold: 0.3 },
        "16-bit": { pixel_size: 4, max_colors: 16, palette_mode: "16-bit", dither: "无", outline_strength: 0.2, outline_threshold: 0.35 },
        "街机": { pixel_size: 6, max_colors: 8, palette_mode: "街机", dither: "Bayer 2x2", outline_strength: 0.35, outline_threshold: 0.28 },
    },
    TUT_GlassRefractionFilter: {
        "波纹玻璃": { mode: "波纹玻璃", amount: 12, scale: 32, angle: 0, blur: 0, chromatic_aberration: 1, roughness: 0.5 },
        "条纹玻璃": { mode: "条纹玻璃", amount: 18, scale: 24, angle: 0, blur: 0.8, chromatic_aberration: 1.5, roughness: 0.3 },
        "磨砂玻璃": { mode: "磨砂玻璃", amount: 10, scale: 48, angle: 0, blur: 1.5, chromatic_aberration: 0.5, roughness: 0.8 },
        "液态玻璃": { mode: "液态玻璃", amount: 16, scale: 42, angle: 20, blur: 0.4, chromatic_aberration: 2, roughness: 0.65 },
        "水滴透镜": { mode: "水滴透镜", amount: 28, scale: 64, angle: 0, blur: 0, chromatic_aberration: 2.5, roughness: 0.2 },
    },
    TUT_GlitchArtFilter: {
        "RGB故障": { mode: "RGB故障", rgb_shift: 10, block_count: 0, block_height: 8, scanline_strength: 0, sort_threshold: 0.5, noise_strength: 0 },
        "VHS": { mode: "VHS", rgb_shift: 5, block_count: 4, block_height: 6, scanline_strength: 0.45, sort_threshold: 0.5, noise_strength: 0.12 },
        "数据损坏": { mode: "数据损坏", rgb_shift: 8, block_count: 10, block_height: 16, scanline_strength: 0.15, sort_threshold: 0.5, noise_strength: 0.08 },
        "像素排序": { mode: "像素排序", rgb_shift: 0, block_count: 0, block_height: 8, scanline_strength: 0, sort_threshold: 0.5, noise_strength: 0 },
    },
};

function applyPreset(node, presetName) {
    const values = FILTER_PRESETS[node.comfyClass]?.[presetName];
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
    name: "TUT_Nodes.FilterPresets",
    nodeCreated(node) {
        if (!FILTER_PRESETS[node.comfyClass]) return;
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
