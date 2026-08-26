import { app } from "/scripts/app.js";

const EDGE_DEFAULTS = {
    shape_mode: "关闭", shape_amount: 0, shape_strength: 1,
    transition_mode: "关闭", transition_strength: 1,
    material_mode: "关闭", material_strength: 1,
    depth_mode: "关闭", depth_strength: 0.55,
    edge_width: 12, edge_color: "white", edge_color_hex: "#FFFFFF",
    detail_scale: 16, irregularity: 0.65,
    background_wrap: 0, background_blur: 0,
    depth_offset_x: 12, depth_offset_y: 12, shadow_blur: 14,
};

const PRESETS = {
    "自然悬浮": {
        shape_mode: "圆角", shape_amount: 32, shape_strength: 1,
        transition_mode: "羽化", transition_strength: 1,
        material_mode: "关闭", material_strength: 1,
        depth_mode: "柔投影", depth_strength: 0.55, edge_width: 12,
        background_wrap: 0.15, background_blur: 0,
        depth_offset_x: 12, depth_offset_y: 12, shadow_blur: 14,
    },
    "柔边照片": {
        shape_mode: "关闭", transition_mode: "羽化", transition_strength: 1,
        material_mode: "关闭", depth_mode: "关闭", edge_width: 28,
        background_wrap: 0.3, background_blur: 0,
    },
    "海报卡片": {
        shape_mode: "切角", shape_amount: 20, shape_strength: 1,
        transition_mode: "关闭", material_mode: "关闭",
        depth_mode: "斜面浮雕", depth_strength: 0.65, edge_width: 8,
        background_wrap: 0.08, background_blur: 0,
    },
    "白边贴纸": {
        shape_mode: "圆角", shape_amount: 18, transition_mode: "关闭",
        material_mode: "白边贴纸", material_strength: 1,
        depth_mode: "柔投影", depth_strength: 0.7, edge_width: 14,
        edge_color: "white", depth_offset_x: 8, depth_offset_y: 10, shadow_blur: 8,
        background_wrap: 0,
    },
    "撕纸拼贴": {
        shape_mode: "撕裂", shape_amount: 14, shape_strength: 1,
        transition_mode: "噪声溶解", transition_strength: 0.3,
        material_mode: "纸张纤维", material_strength: 1,
        depth_mode: "柔投影", depth_strength: 0.7, edge_width: 12,
        edge_color: "custom", edge_color_hex: "#F2E9D5", detail_scale: 10,
        irregularity: 0.8, depth_offset_x: 6, depth_offset_y: 8, shadow_blur: 6,
    },
    "烧焦照片": {
        shape_mode: "撕裂", shape_amount: 16, shape_strength: 1,
        transition_mode: "噪声溶解", transition_strength: 0.45,
        material_mode: "烧焦", material_strength: 1,
        depth_mode: "柔投影", depth_strength: 0.65, edge_width: 16,
        detail_scale: 12, irregularity: 0.9, background_wrap: 0.12,
        depth_offset_x: 8, depth_offset_y: 10, shadow_blur: 10,
    },
    "玻璃面板": {
        shape_mode: "圆角", shape_amount: 42, transition_mode: "羽化",
        transition_strength: 0.35, material_mode: "玻璃切边", material_strength: 0.85,
        depth_mode: "斜面浮雕", depth_strength: 0.5, edge_width: 14,
        edge_color: "cyan", background_wrap: 0.5, background_blur: 3,
    },
    "霓虹窗口": {
        shape_mode: "圆角", shape_amount: 28, transition_mode: "关闭",
        material_mode: "玻璃切边", material_strength: 0.35,
        depth_mode: "霓虹边光", depth_strength: 0.9, edge_width: 12,
        edge_color: "cyan", background_wrap: 0.25, shadow_blur: 18,
    },
    "墨水渗透": {
        shape_mode: "波浪", shape_amount: 8, transition_mode: "墨水扩散",
        transition_strength: 1, material_mode: "关闭", depth_mode: "关闭",
        edge_width: 26, detail_scale: 24, irregularity: 0.85,
        background_wrap: 0.45,
    },
    "像素崩解": {
        shape_mode: "关闭", transition_mode: "像素崩解", transition_strength: 1,
        material_mode: "关闭", depth_mode: "关闭", edge_width: 30,
        detail_scale: 8, irregularity: 0.9, background_wrap: 0.15,
    },
    "噪声消散": {
        shape_mode: "撕裂", shape_amount: 8, transition_mode: "噪声溶解",
        transition_strength: 1, material_mode: "关闭", depth_mode: "关闭",
        edge_width: 32, detail_scale: 14, irregularity: 1,
        background_wrap: 0.3,
    },
    "厚卡片": {
        shape_mode: "切角", shape_amount: 18, transition_mode: "关闭",
        material_mode: "白边贴纸", material_strength: 0.7,
        depth_mode: "伪厚度", depth_strength: 0.9, edge_width: 16,
        edge_color: "custom", edge_color_hex: "#303040",
        depth_offset_x: 16, depth_offset_y: 18, shadow_blur: 4,
        background_wrap: 0.08,
    },
};

function applyPreset(node, presetName) {
    const presetValues = PRESETS[presetName];
    if (!presetValues) return false;
    const values = { ...EDGE_DEFAULTS, ...presetValues };
    for (const [name, value] of Object.entries(values)) {
        const widget = node.widgets?.find((candidate) => candidate.name === name);
        if (widget) widget.value = value;
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
    return true;
}

app.registerExtension({
    name: "TUT_Nodes.CompositePresets",
    nodeCreated(node) {
        if (node.comfyClass !== "TUT_SoftLayerComposite") return;
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
