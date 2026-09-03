import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const PANEL_LAYOUTS = {
    "整页单格": [[0, 0, 1, 1]],
    "左右双格": [[0, 0, .5, 1], [.5, 0, 1, 1]],
    "上下双格": [[0, 0, 1, .5], [0, .5, 1, 1]],
    "上大下二": [[0, 0, 1, .62], [0, .62, .5, 1], [.5, .62, 1, 1]],
    "左大右二": [[0, 0, .62, 1], [.62, 0, 1, .5], [.62, .5, 1, 1]],
    "四宫格": [[0, 0, .5, .5], [.5, 0, 1, .5], [0, .5, .5, 1], [.5, .5, 1, 1]],
    "主格加三小格": [[0, 0, .64, 1], [.64, 0, 1, 1 / 3], [.64, 1 / 3, 1, 2 / 3], [.64, 2 / 3, 1, 1]],
    "五格错落": [[0, 0, .5, .54], [.5, 0, 1, .54], [0, .54, 1 / 3, 1], [1 / 3, .54, 2 / 3, 1], [2 / 3, .54, 1, 1]],
    "六宫格": [[0, 0, 1 / 3, .5], [1 / 3, 0, 2 / 3, .5], [2 / 3, 0, 1, .5], [0, .5, 1 / 3, 1], [1 / 3, .5, 2 / 3, 1], [2 / 3, .5, 1, 1]],
};
const AUTO_BY_COUNT = ["整页单格", "整页单格", "左右双格", "上大下二", "四宫格", "五格错落", "六宫格"];
const BUBBLE_SHAPES = ["椭圆对白框", "圆角矩形", "云朵思考框", "爆炸喊话框", "爆炸对话框", "闪光对话框", "方形旁白框", "无边框文字"];
const BUBBLE_SPIKE_COUNT_DEFAULT = 16;
const BUBBLE_SPIKE_DEPTH_DEFAULT = .22;
const BUBBLE_CLOUD_LOBES_DEFAULT = 10;
const BUBBLE_CLOUD_DEPTH_DEFAULT = .14;
const COMIC_FONT_LOADS = new Map();
let comicFontCatalogPromise = null;

function loadComicFontCatalog() {
    if (!comicFontCatalogPromise) comicFontCatalogPromise = fetch(api.apiURL("/tut_nodes/fonts/catalog"))
        .then(async (response) => {
            const payload = await response.json();
            if (!response.ok || !Array.isArray(payload?.fonts)) throw new Error(payload?.error || "字体目录加载失败");
            return payload.fonts.filter((entry) => entry && typeof entry.token === "string" && typeof entry.display_name === "string");
        });
    return comicFontCatalogPromise;
}

function ensureComicFontLoaded(entry) {
    if (!entry?.token || !entry?.preview_family) return Promise.reject(new Error("字体预览信息无效"));
    const existing = COMIC_FONT_LOADS.get(entry.token);
    if (existing) return existing.promise;
    const record = { status: "loading", family: entry.preview_family, error: "", promise: null };
    record.promise = Promise.resolve().then(async () => {
        if (typeof FontFace !== "function" || !document.fonts?.add) throw new Error("当前浏览器不支持字体预览加载");
        const url = api.apiURL(`/tut_nodes/fonts/file?token=${encodeURIComponent(entry.token)}`);
        const face = new FontFace(entry.preview_family, `url(${JSON.stringify(url)})`);
        await face.load();
        document.fonts.add(face);
        record.status = "loaded";
        return entry.preview_family;
    }).catch((error) => {
        record.status = "failed"; record.error = error?.message || "字体预览加载失败"; throw error;
    });
    COMIC_FONT_LOADS.set(entry.token, record);
    return record.promise;
}

function loadedComicFontFamily(token) {
    const record = COMIC_FONT_LOADS.get(token);
    return record?.status === "loaded" ? record.family : null;
}

const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, Number(value) || 0));
const clone = (value) => JSON.parse(JSON.stringify(value));

function hideStore(store) {
    store.serialize = true;
    store.options ||= {};
    store.options.serialize = true;
    store.options.hidden = true;
    store.hidden = true;
    store.type = "hidden";
    store.computeSize = () => [0, -4];
    store.draw = () => {};
    store.serializeValue = () => store.value;
}

function commit(node, store, data, callback) {
    const graph = app.graph;
    graph?.beforeChange?.();
    try {
        store.value = JSON.stringify(data);
        callback?.call(store, store.value);
        node.setDirtyCanvas?.(true, true);
        graph?.setDirtyCanvas?.(true, true);
    } finally {
        graph?.afterChange?.();
    }
}

function defaultPanels() {
    return {
        version: 1,
        panels: Array.from({ length: 6 }, () => ({ focus_x: .5, focus_y: .5, zoom: 1, flip: false })),
        layout_overrides: {},
    };
}

function parsePanels(value) {
    try {
        const parsed = JSON.parse(value);
        if (parsed?.version !== 1 || !Array.isArray(parsed.panels)) return defaultPanels();
        const result = defaultPanels();
        parsed.panels.slice(0, 6).forEach((panel, index) => {
            result.panels[index] = {
                focus_x: clamp(panel?.focus_x), focus_y: clamp(panel?.focus_y),
                zoom: clamp(panel?.zoom ?? 1, .25, 4), flip: panel?.flip === true,
            };
        });
        if (parsed.layout_overrides && typeof parsed.layout_overrides === "object") {
            for (const [name, rectangles] of Object.entries(parsed.layout_overrides)) {
                const defaults = PANEL_LAYOUTS[name];
                if (!defaults || !Array.isArray(rectangles) || rectangles.length !== defaults.length) continue;
                const clean = rectangles.map((rect, index) => {
                    if (!Array.isArray(rect) || rect.length !== 4) return clone(defaults[index]);
                    const x0 = clamp(rect[0]), y0 = clamp(rect[1]), x1 = clamp(rect[2]), y1 = clamp(rect[3]);
                    return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : clone(defaults[index]);
                });
                result.layout_overrides[name] = clean;
            }
        }
        return result;
    } catch {
        return defaultPanels();
    }
}

function commitWidget(node, widget, value) {
    if (!widget || widget.value === value) return;
    const graph = app.graph;
    graph?.beforeChange?.();
    try {
        widget.value = value;
        widget.callback?.call(widget, value);
        node.setDirtyCanvas?.(true, true);
        graph?.setDirtyCanvas?.(true, true);
    } finally { graph?.afterChange?.(); }
}

function button(label) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    return element;
}

function range(min, max, step) {
    const element = document.createElement("input");
    element.type = "range"; element.min = min; element.max = max; element.step = step;
    return element;
}

function installPanelEditor(node) {
    if (node.comfyClass !== "TUT_ComicPanelCanvas" || node.__tutComicPanelEditor) return;
    const store = node.widgets?.find((widget) => widget.name === "panel_data");
    const layoutWidget = node.widgets?.find((widget) => widget.name === "layout");
    const widthWidget = node.widgets?.find((widget) => widget.name === "canvas_width");
    const heightWidget = node.widgets?.find((widget) => widget.name === "canvas_height");
    if (!store || !layoutWidget || !widthWidget || !heightWidget || typeof node.addDOMWidget !== "function") return;
    hideStore(store);
    const previousStoreCallback = store.callback;
    let data = parsePanels(store.value);
    let selected = 0, editMode = "frame", dragState = null;

    const root = document.createElement("div");
    root.style.cssText = "display:flex;flex-direction:column;gap:7px;width:100%;padding:5px;box-sizing:border-box;color:#ddd;font:12px sans-serif;overflow:hidden";
    const toolbar = document.createElement("div");
    toolbar.style.cssText = "display:flex;gap:5px;align-items:center;flex-wrap:wrap";
    const preset = document.createElement("select");
    for (const value of ["1024 × 1024", "1024 × 1344", "1024 × 1536", "1344 × 1024", "1536 × 1024", "1536 × 1536", "自定义"]) preset.add(new Option(value, value));
    const width = document.createElement("input"), height = document.createElement("input");
    for (const input of [width, height]) {
        input.type = "number"; input.min = "64"; input.max = "8192"; input.step = "8";
        input.style.cssText = "width:72px;background:#20242a;color:#eee;border:1px solid #555;border-radius:4px;padding:3px";
    }
    preset.style.cssText = "background:#20242a;color:#eee;border:1px solid #555;border-radius:4px;padding:3px";
    const frameMode = button("调整画框"), cameraMode = button("调整镜头"), resetLayout = button("恢复模板");
    toolbar.append(document.createTextNode("画布"), preset, document.createTextNode("宽"), width, document.createTextNode("高"), height, frameMode, cameraMode, resetLayout);
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "display:block;max-width:100%;height:auto;max-height:500px;align-self:center;background:#eee;border:1px solid #555;border-radius:5px;touch-action:none";
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:7px;flex-wrap:wrap";
    const zoom = range("0.25", "4", "0.05");
    zoom.style.flex = "1";
    const zoomText = document.createElement("span");
    const flip = button("水平翻转");
    const reset = button("重置当前格");
    row.append(document.createTextNode("镜头缩放"), zoom, zoomText, flip, reset);
    const hint = document.createElement("div");
    hint.textContent = "调整画框：拖动画格移动，拖动四角改变尺寸；调整镜头：拖动黄色焦点。画框使用相对坐标，改画布尺寸后仍保持布局。";
    hint.style.color = "#aaa";
    root.append(toolbar, canvas, row, hint);
    const context = canvas.getContext("2d");

    function currentLayout() {
        const value = layoutWidget.value;
        return value === "自动匹配数量" ? "六宫格" : (PANEL_LAYOUTS[value] ? value : "六宫格");
    }
    function rectangles() { return data.layout_overrides[currentLayout()] || PANEL_LAYOUTS[currentLayout()]; }
    function editableRectangles() {
        const name = currentLayout();
        if (!data.layout_overrides[name]) data.layout_overrides[name] = clone(PANEL_LAYOUTS[name]);
        return data.layout_overrides[name];
    }
    function save() { commit(node, store, data, previousStoreCallback); }
    function canvasSize() {
        return {
            width: Math.max(64, Math.min(8192, Math.round(Number(widthWidget.value) || 1024))),
            height: Math.max(64, Math.min(8192, Math.round(Number(heightWidget.value) || 1536))),
        };
    }
    function updateCanvasSize() {
        const size = canvasSize(), scale = Math.min(720 / size.width, 500 / size.height);
        canvas.width = Math.max(80, Math.round(size.width * scale));
        canvas.height = Math.max(80, Math.round(size.height * scale));
        width.value = String(size.width); height.value = String(size.height);
        const label = `${size.width} × ${size.height}`;
        preset.value = Array.from(preset.options).some((option) => option.value === label) ? label : "自定义";
    }
    function updateModeButtons() {
        frameMode.style.background = editMode === "frame" ? "#167d86" : "";
        cameraMode.style.background = editMode === "camera" ? "#167d86" : "";
    }
    function render() {
        updateCanvasSize(); updateModeButtons();
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "#f6f3ed"; context.fillRect(0, 0, canvas.width, canvas.height);
        selected = Math.max(0, Math.min(selected, rectangles().length - 1));
        rectangles().forEach((rect, index) => {
            const [x0, y0, x1, y1] = rect;
            const x = x0 * canvas.width + 3, y = y0 * canvas.height + 3;
            const w = Math.max(2, (x1 - x0) * canvas.width - 6), h = Math.max(2, (y1 - y0) * canvas.height - 6);
            context.fillStyle = index === selected ? "#315d78" : "#293039";
            context.fillRect(x, y, w, h);
            context.strokeStyle = index === selected ? "#62d7ff" : "#0d0f11";
            context.lineWidth = index === selected ? 5 : 3;
            context.strokeRect(x, y, w, h);
            const panel = data.panels[index];
            const fx = x + panel.focus_x * w, fy = y + panel.focus_y * h;
            context.strokeStyle = "#ffcf45"; context.lineWidth = 2;
            context.beginPath(); context.moveTo(fx - 10, fy); context.lineTo(fx + 10, fy); context.moveTo(fx, fy - 10); context.lineTo(fx, fy + 10); context.stroke();
            context.fillStyle = "#fff"; context.font = "bold 18px sans-serif"; context.fillText(`${index + 1}`, x + 10, y + 24);
            if (panel.flip) { context.fillStyle = "#ffcf45"; context.fillText("↔", x + w - 28, y + 24); }
            if (index === selected && editMode === "frame") {
                context.fillStyle = "#62d7ff";
                for (const [hx, hy] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) context.fillRect(hx * canvas.width - 6, hy * canvas.height - 6, 12, 12);
            }
        });
        zoom.value = data.panels[selected].zoom;
        zoomText.textContent = `${data.panels[selected].zoom.toFixed(2)}×`;
        flip.style.background = data.panels[selected].flip ? "#2f83a8" : "";
    }
    function locate(event) {
        const bounds = canvas.getBoundingClientRect();
        return { x: clamp((event.clientX - bounds.left) / bounds.width), y: clamp((event.clientY - bounds.top) / bounds.height) };
    }
    function panelAt(point) {
        const rects = rectangles();
        for (let index = rects.length - 1; index >= 0; index--) {
            const [x0, y0, x1, y1] = rects[index];
            if (point.x >= x0 && point.x <= x1 && point.y >= y0 && point.y <= y1) return index;
        }
        return -1;
    }
    function cornerAt(point, rect) {
        const corners = [["nw", rect[0], rect[1]], ["ne", rect[2], rect[1]], ["sw", rect[0], rect[3]], ["se", rect[2], rect[3]]];
        return corners.find(([, x, y]) => Math.hypot((point.x - x) * canvas.width, (point.y - y) * canvas.height) <= 16)?.[0] || null;
    }
    function updateFocus(point) {
        const rect = rectangles()[selected];
        data.panels[selected].focus_x = clamp((point.x - rect[0]) / (rect[2] - rect[0]));
        data.panels[selected].focus_y = clamp((point.y - rect[1]) / (rect[3] - rect[1]));
        render();
    }
    const down = (event) => {
        event.preventDefault();
        const point = locate(event), index = panelAt(point);
        if (index < 0) return;
        selected = index;
        if (editMode === "camera") {
            dragState = { kind: "camera" }; updateFocus(point);
        } else {
            const rect = rectangles()[selected];
            dragState = { kind: cornerAt(point, rect) || "move", start: point, rect: clone(rect) };
            editableRectangles();
        }
        canvas.setPointerCapture?.(event.pointerId); render();
    };
    const move = (event) => {
        if (!dragState) return;
        const point = locate(event);
        if (dragState.kind === "camera") { updateFocus(point); return; }
        const rects = editableRectangles(), source = dragState.rect, next = clone(source);
        const minW = Math.max(.03, 24 / canvas.width), minH = Math.max(.03, 24 / canvas.height);
        if (dragState.kind === "move") {
            const dx = Math.max(-source[0], Math.min(1 - source[2], point.x - dragState.start.x));
            const dy = Math.max(-source[1], Math.min(1 - source[3], point.y - dragState.start.y));
            next[0] += dx; next[2] += dx; next[1] += dy; next[3] += dy;
        } else {
            if (dragState.kind.includes("w")) next[0] = Math.min(source[2] - minW, point.x);
            if (dragState.kind.includes("e")) next[2] = Math.max(source[0] + minW, point.x);
            if (dragState.kind.includes("n")) next[1] = Math.min(source[3] - minH, point.y);
            if (dragState.kind.includes("s")) next[3] = Math.max(source[1] + minH, point.y);
            next[0] = clamp(next[0]); next[1] = clamp(next[1]); next[2] = clamp(next[2]); next[3] = clamp(next[3]);
        }
        rects[selected] = next; render();
    };
    const up = (event) => { if (dragState) save(); dragState = null; canvas.releasePointerCapture?.(event.pointerId); };
    canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
    zoom.addEventListener("input", () => { data.panels[selected].zoom = Number(zoom.value); save(); render(); });
    flip.addEventListener("click", () => { data.panels[selected].flip = !data.panels[selected].flip; save(); render(); });
    reset.addEventListener("click", () => {
        data.panels[selected] = { focus_x: .5, focus_y: .5, zoom: 1, flip: false };
        if (data.layout_overrides[currentLayout()]) data.layout_overrides[currentLayout()][selected] = clone(PANEL_LAYOUTS[currentLayout()][selected]);
        save(); render();
    });
    frameMode.addEventListener("click", () => { editMode = "frame"; render(); });
    cameraMode.addEventListener("click", () => { editMode = "camera"; render(); });
    resetLayout.addEventListener("click", () => { delete data.layout_overrides[currentLayout()]; save(); render(); });
    preset.addEventListener("change", () => {
        const match = preset.value.match(/^(\d+) × (\d+)$/);
        if (match) { commitWidget(node, widthWidget, Number(match[1])); commitWidget(node, heightWidget, Number(match[2])); render(); }
    });
    function commitSizeInputs() {
        commitWidget(node, widthWidget, Math.max(64, Math.min(8192, Math.round(Number(width.value) || 1024))));
        commitWidget(node, heightWidget, Math.max(64, Math.min(8192, Math.round(Number(height.value) || 1536))));
        render();
    }
    for (const input of [width, height]) {
        input.addEventListener("change", commitSizeInputs);
        input.addEventListener("keydown", (event) => { if (event.key === "Enter") { commitSizeInputs(); input.blur(); } });
        input.addEventListener("pointerdown", (event) => event.stopPropagation());
    }
    const previousLayoutCallback = layoutWidget.callback;
    layoutWidget.callback = function (...args) { const result = previousLayoutCallback?.apply(this, args); selected = 0; render(); return result; };
    const previousWidthCallback = widthWidget.callback, previousHeightCallback = heightWidget.callback;
    widthWidget.callback = function (...args) { const result = previousWidthCallback?.apply(this, args); render(); return result; };
    heightWidget.callback = function (...args) { const result = previousHeightCallback?.apply(this, args); render(); return result; };
    store.callback = function (...args) { const result = previousStoreCallback?.apply(this, args); data = parsePanels(store.value); render(); return result; };
    const widget = node.addDOMWidget("tut_comic_panel_editor", "TUT_COMIC_PANEL_EDITOR", root, { serialize: false, hideOnZoom: false });
    widget.computeSize = (nodeWidth) => [nodeWidth, Math.min(680, canvas.height + 150)];
    node.__tutComicPanelEditor = true;
    render();
    node.setSize?.([Math.max(node.size?.[0] || 360, 540), Math.max(node.size?.[1] || 300, 900)]);
}

function newBubble(fontName, index) {
    return {
        id: `bubble-${Date.now()}-${index}`, shape: "椭圆对白框", x: .5, y: .28, w: .34, h: .2,
        text: "输入台词", font_name: fontName, font_size: 36,
        text_direction: "horizontal",
        text_color: "#111111", fill_color: "#ffffff", border_color: "#111111", border_width: 4, opacity: 1,
        spike_count: BUBBLE_SPIKE_COUNT_DEFAULT, spike_depth: BUBBLE_SPIKE_DEPTH_DEFAULT,
        cloud_lobes: BUBBLE_CLOUD_LOBES_DEFAULT, cloud_depth: BUBBLE_CLOUD_DEPTH_DEFAULT,
    };
}

function parseBubbles(value) {
    try {
        const parsed = JSON.parse(value);
        if (parsed?.version !== 1 || !Array.isArray(parsed.bubbles)) return { version: 1, merge_overlaps: false, bubbles: [] };
        const bubbles = parsed.bubbles.slice(0, 32).map((item) => {
            if (!item || typeof item !== "object" || Array.isArray(item)) return item;
            const clean = { ...item };
            for (const axis of ["x", "y"]) delete clean[["tail", axis].join("_")];
            clean.spike_count = Math.round(clamp(clean.spike_count ?? BUBBLE_SPIKE_COUNT_DEFAULT, 6, 32));
            clean.spike_depth = clamp(clean.spike_depth ?? BUBBLE_SPIKE_DEPTH_DEFAULT, .05, .70);
            clean.cloud_lobes = Math.round(clamp(clean.cloud_lobes ?? BUBBLE_CLOUD_LOBES_DEFAULT, 6, 16));
            clean.cloud_depth = clamp(clean.cloud_depth ?? BUBBLE_CLOUD_DEPTH_DEFAULT, .05, .30);
            clean.text_direction = ["horizontal", "vertical_ltr", "vertical_rtl"].includes(clean.text_direction) ? clean.text_direction : "horizontal";
            return clean;
        });
        return { version: 1, merge_overlaps: parsed.merge_overlaps === true, bubbles };
    } catch { return { version: 1, merge_overlaps: false, bubbles: [] }; }
}

function bubblePreviewUrl(data) {
    if (!data?.filename) return null;
    const query = new URLSearchParams({ filename: data.filename, type: data.type ?? "temp", subfolder: data.subfolder ?? "" });
    const format = app.getPreviewFormatParam?.() ?? "";
    const random = app.getRandParam?.() ?? `&rand=${Math.random()}`;
    return api.apiURL(`/view?${query.toString()}${format}${random}`);
}

function ensureBubbleStyle() {
    if (document.getElementById("tut-comic-bubble-v2-style")) return;
    const style = document.createElement("style");
    style.id = "tut-comic-bubble-v2-style";
    style.textContent = `
      .tut-bubble-wrap{display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:0;box-sizing:border-box;color:#ddd;font:12px sans-serif;overflow:hidden}
      .tut-bubble-toolbar,.tut-bubble-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.tut-bubble-toolbar{justify-content:space-between;flex-wrap:nowrap}.tut-bubble-tools{display:flex;gap:5px;align-items:center;min-width:0}
      .tut-bubble-btn,.tut-bubble-select,.tut-bubble-input,.tut-bubble-textarea{background:#25292e;color:#eee;border:1px solid #555;border-radius:5px;padding:5px 7px;box-sizing:border-box}
      .tut-bubble-btn{cursor:pointer;white-space:nowrap}.tut-bubble-btn:hover{border-color:#aaa}.tut-bubble-btn.active{background:#167d86;border-color:#35d0c8;color:#fff}.tut-bubble-btn:disabled{opacity:.35;cursor:not-allowed}
      .tut-bubble-count{color:#aeb4bb;white-space:nowrap;font-variant-numeric:tabular-nums}.tut-bubble-workspace{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:8px;flex:1;min-height:0;overflow:hidden}
      .tut-bubble-monitor{background:#070707;border:1px solid #333;border-radius:8px;padding:10px;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;min-width:0}
      .tut-bubble-screen{position:relative;background:#202328;border:2px solid #555;overflow:hidden;line-height:0;max-width:100%;max-height:100%}.tut-bubble-canvas{display:block;width:100%;height:100%;touch-action:none;cursor:default}
      .tut-bubble-sidebar{background:#15171a;border:1px solid #34383d;border-radius:7px;display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}.tut-bubble-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:#30343a;border-bottom:1px solid #3a3e44}
      .tut-bubble-tab{border:0;border-radius:0;background:#202328;color:#aaa;padding:8px 3px;cursor:pointer;font-weight:700}.tut-bubble-tab.active{background:#167d86;color:#fff}.tut-bubble-sidebar-body{padding:9px;min-height:0;overflow:auto;flex:1}
      .tut-bubble-panel{display:none;flex-direction:column;gap:9px;min-width:0}.tut-bubble-panel.active{display:flex}.tut-bubble-panel h4{margin:0;color:#35d0c8;font-size:12px}.tut-bubble-field{display:flex;flex-direction:column;gap:3px;min-width:0;color:#aaa;font-size:10px}
      .tut-bubble-field>.tut-bubble-input,.tut-bubble-field>.tut-bubble-select,.tut-bubble-field>.tut-bubble-textarea{width:100%;min-width:0}.tut-bubble-textarea{resize:vertical;min-height:92px;line-height:1.45;font:12px sans-serif}
      .tut-bubble-dialogue{display:flex;flex-direction:column;gap:5px}.tut-bubble-writing{display:flex;gap:5px}.tut-bubble-writing .tut-bubble-btn{min-width:36px;padding:4px 9px;font-size:14px}.tut-bubble-writing .tut-bubble-help{display:flex;align-items:center;margin-left:3px}
      .tut-bubble-search{width:100%;min-width:0}.tut-bubble-search-status{min-height:14px;color:#92979e;font-size:10px;line-height:1.4}.tut-bubble-burst,.tut-bubble-cloud{display:none;flex-direction:column;gap:8px;padding-top:2px}.tut-bubble-burst.active,.tut-bubble-cloud.active{display:flex}.tut-bubble-burst h4,.tut-bubble-cloud h4{margin:0;color:#35d0c8;font-size:12px}
      .tut-bubble-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.tut-bubble-range{width:100%;min-width:0}.tut-bubble-color{width:100%;height:29px;padding:1px}.tut-bubble-index{display:block;font-weight:700;color:#fff;background:#167d86;border-radius:4px;padding:4px 7px}
      .tut-bubble-toggle{display:flex;align-items:center;gap:7px;color:#ddd;cursor:pointer}.tut-bubble-toggle input{accent-color:#35d0c8}.tut-bubble-toggle-copy{display:flex;flex-direction:column;gap:2px}.tut-bubble-toggle-copy small{color:#92979e;line-height:1.4}
      .tut-bubble-help{color:#92979e;font-size:10px;line-height:1.45}.tut-bubble-layer-list{display:flex;flex-direction:column;gap:5px}.tut-bubble-layer-row{display:grid;grid-template-columns:24px minmax(0,1fr) auto;align-items:center;gap:6px;background:#22262b;border:1px solid #444;border-radius:5px;padding:6px 7px;color:#ddd;cursor:pointer}
      .tut-bubble-layer-row.active{border-color:#35d0c8;background:#164f55}.tut-bubble-layer-rank{color:#8f979f;text-align:center;font-variant-numeric:tabular-nums}.tut-bubble-layer-row.active .tut-bubble-layer-rank{color:#fff}.tut-bubble-layer-actions{display:grid;grid-template-columns:1fr 1fr;gap:6px}.tut-bubble-layer-actions .tut-bubble-btn{width:100%}
      @media (max-width:900px){.tut-bubble-workspace{grid-template-columns:minmax(0,1fr) 260px}}
    `;
    document.head.appendChild(style);
}

function bubbleButton(label, title = "") {
    const element = button(label); element.className = "tut-bubble-btn"; element.title = title; return element;
}

function bubbleField(label, control) {
    const field = document.createElement("label"); field.className = "tut-bubble-field";
    const caption = document.createElement("span"); caption.textContent = label; field.append(caption, control); return field;
}

function drawBubblePath(context, bubble, x, y, w, h) {
    context.beginPath();
    if (bubble.shape === "椭圆对白框") context.ellipse(x, y, w / 2, h / 2, 0, 0, Math.PI * 2);
    else if (bubble.shape === "圆角矩形") context.roundRect(x - w / 2, y - h / 2, w, h, Math.max(4, Math.min(w, h) / 6));
    else if (bubble.shape === "闪光对话框") context.ellipse(x, y, w * .34, h * .34, 0, 0, Math.PI * 2);
    else if (bubble.shape === "爆炸喊话框") {
        const pointCount = Math.round(clamp(bubble.spike_count ?? BUBBLE_SPIKE_COUNT_DEFAULT, 6, 32)) * 2;
        const innerRadius = .5 * (1 - clamp(bubble.spike_depth ?? BUBBLE_SPIKE_DEPTH_DEFAULT, .05, .70));
        for (let point = 0; point < pointCount; point++) {
            const angle = -Math.PI / 2 + point * Math.PI * 2 / pointCount, radius = point % 2 ? innerRadius : .5;
            const px = x + Math.cos(angle) * w * radius, py = y + Math.sin(angle) * h * radius;
            point ? context.lineTo(px, py) : context.moveTo(px, py);
        }
        context.closePath();
    } else if (bubble.shape === "爆炸对话框") {
        context.moveTo(x - w * .48, y - h * .46);
        context.bezierCurveTo(x - w * .39, y - h * .39, x - w * .30, y - h * .34, x - w * .22, y - h * .36);
        context.lineTo(x - w * .17, y - h * .35); context.lineTo(x - w * .23, y - h * .40); context.lineTo(x - w * .13, y - h * .36);
        context.bezierCurveTo(x + w * .13, y - h * .35, x + w * .34, y - h * .42, x + w * .48, y - h * .50);
        context.bezierCurveTo(x + w * .45, y - h * .40, x + w * .42, y - h * .31, x + w * .43, y - h * .25);
        context.lineTo(x + w * .54, y - h * .32); context.lineTo(x + w * .45, y - h * .18);
        context.bezierCurveTo(x + w * .41, y - h * .02, x + w * .41, y + h * .13, x + w * .44, y + h * .22);
        context.lineTo(x + w * .50, y + h * .16); context.lineTo(x + w * .44, y + h * .31); context.lineTo(x + w * .48, y + h * .48);
        context.bezierCurveTo(x + w * .20, y + h * .37, x - w * .18, y + h * .35, x - w * .43, y + h * .49);
        context.lineTo(x - w * .35, y + h * .31); context.lineTo(x - w * .48, y + h * .36); context.lineTo(x - w * .41, y + h * .19);
        context.bezierCurveTo(x - w * .38, y + h * .02, x - w * .40, y - h * .28, x - w * .48, y - h * .46);
        context.closePath();
    } else if (bubble.shape === "云朵思考框") {
        const lobeCount = Math.round(clamp(bubble.cloud_lobes ?? BUBBLE_CLOUD_LOBES_DEFAULT, 6, 16));
        const cloudDepth = clamp(bubble.cloud_depth ?? BUBBLE_CLOUD_DEPTH_DEFAULT, .05, .30);
        const baseValleyRadius = .48 - .35 * cloudDepth, baseControlRadius = .54 + .15 * cloudDepth;
        const weights = Array.from({ length: lobeCount }, (_, index) => 1 + .20 * Math.sin(index * 2.17 + .3) + .10 * Math.sin(index * .91 + 1.4));
        const weightTotal = weights.reduce((sum, value) => sum + value, 0);
        const angleSteps = weights.map((weight) => Math.PI * 2 * weight / weightTotal);
        const startAngle = -Math.PI / 2 - angleSteps[0] / 2;
        let currentAngle = startAngle;
        const firstRadius = baseValleyRadius * (1 + .045 * Math.sin(.5));
        context.moveTo(x + Math.cos(startAngle) * w * firstRadius, y + Math.sin(startAngle) * h * firstRadius);
        for (let lobe = 0; lobe < lobeCount; lobe++) {
            const angleStep = angleSteps[lobe], start = currentAngle, middle = start + angleStep / 2, end = start + angleStep;
            const endRadius = baseValleyRadius * (1 + .045 * Math.sin((lobe + 1) * 1.73 + .5));
            const controlRadius = baseControlRadius * (1 + .12 * Math.sin(lobe * 2.39 + .8));
            context.quadraticCurveTo(
                x + Math.cos(middle) * w * controlRadius, y + Math.sin(middle) * h * controlRadius,
                x + Math.cos(end) * w * endRadius, y + Math.sin(end) * h * endRadius,
            );
            currentAngle = end;
        }
        context.closePath();
    } else context.rect(x - w / 2, y - h / 2, w, h);
}

function drawFlashRays(context, bubble, x, y, w, h) {
    const rayCount = 96, angleStep = Math.PI * 2 / rayCount;
    const halfAngle = angleStep * Math.min(.32, .16 + (bubble.border_width || 0) / 80);
    context.beginPath();
    for (let index = 0; index < rayCount; index++) {
        const angle = -Math.PI / 2 + index * angleStep;
        const outerRadius = .47 + .03 * (.5 + .5 * Math.sin(index * 2.07 + .4));
        context.moveTo(x + Math.cos(angle) * w * .31, y + Math.sin(angle) * h * .31);
        context.lineTo(x + Math.cos(angle - halfAngle) * w * .40, y + Math.sin(angle - halfAngle) * h * .40);
        context.lineTo(x + Math.cos(angle) * w * outerRadius, y + Math.sin(angle) * h * outerRadius);
        context.lineTo(x + Math.cos(angle + halfAngle) * w * .40, y + Math.sin(angle + halfAngle) * h * .40);
        context.closePath();
    }
    context.fill();
}

function drawFlashFill(context, color, x, y, w, h) {
    const solidColor = /^#[0-9a-f]{6}$/i.test(color) ? color : "#ffffff";
    context.save(); context.translate(x, y); context.scale(w, h);
    const gradient = context.createRadialGradient(0, 0, .34, 0, 0, .50);
    gradient.addColorStop(0, solidColor);
    gradient.addColorStop(.25, `${solidColor}d7`);
    gradient.addColorStop(.50, `${solidColor}80`);
    gradient.addColorStop(.75, `${solidColor}28`);
    gradient.addColorStop(1, `${solidColor}00`);
    context.fillStyle = gradient; context.beginPath(); context.ellipse(0, 0, .50, .50, 0, 0, Math.PI * 2); context.fill(); context.restore();
}

function drawBubbleText(context, bubble, x, y, w, h, previewScale = 1) {
    const sourceFontSize = Math.max(8, Math.min(160, Number(bubble.font_size) || 36));
    const minimumFontSize = Math.max(.5, 8 * previewScale);
    const shrinkStep = Math.max(.25, 2 * previewScale);
    let fontSize = Math.max(.5, sourceFontSize * previewScale);
    const previewFamily = loadedComicFontFamily(bubble.font_name);
    if (!previewFamily) return;
    const setPreviewFont = () => { context.font = `${fontSize}px "${previewFamily}"`; };
    context.fillStyle = bubble.text_color || "#111111"; setPreviewFont();
    context.textAlign = "center"; context.textBaseline = "middle";
    const specialPadding = ["爆炸喊话框", "爆炸对话框", "闪光对话框"].includes(bubble.shape);
    const padding = Math.max(3 * previewScale, Math.min(w, h) * (specialPadding ? .16 : .10));
    const availableWidth = Math.max(1, w - padding * 2), availableHeight = Math.max(1, h - padding * 2);
    const direction = bubble.text_direction || "horizontal";
    if (direction !== "horizontal") {
        const makeColumns = (capacity) => String(bubble.text || "").split("\n").flatMap((paragraph) => {
            const characters = Array.from(paragraph);
            if (!characters.length) return [[]];
            const columns = [];
            for (let offset = 0; offset < characters.length; offset += capacity) columns.push(characters.slice(offset, offset + capacity));
            return columns;
        });
        let lineStep, columnStep, columns;
        while (true) {
            lineStep = fontSize * 1.15; columnStep = fontSize * 1.15;
            columns = makeColumns(Math.max(1, Math.floor(availableHeight / lineStep)));
            if (columns.length * columnStep <= availableWidth || fontSize <= minimumFontSize) break;
            fontSize = Math.max(minimumFontSize, fontSize - shrinkStep); setPreviewFont();
        }
        columns = columns.slice(0, Math.max(1, Math.floor(availableWidth / columnStep)));
        const totalWidth = columns.length * columnStep;
        columns.forEach((characters, logicalIndex) => {
            const visualIndex = direction === "vertical_ltr" ? logicalIndex : columns.length - 1 - logicalIndex;
            const columnX = x - totalWidth / 2 + (visualIndex + .5) * columnStep;
            const startY = y - (characters.length - 1) * lineStep / 2;
            characters.forEach((character, row) => context.fillText(character, columnX, startY + row * lineStep));
        });
        return;
    }
    let lines, lineHeight, lineSpacing;
    while (true) {
        lines = []; lineHeight = fontSize * 1.15; lineSpacing = fontSize * .15;
        for (const paragraph of String(bubble.text || "").split("\n")) {
            let line = "";
            for (const character of paragraph) {
                const next = line + character;
                if (line && context.measureText(next).width > availableWidth) { lines.push(line); line = character; } else line = next;
            }
            lines.push(line);
        }
        const totalHeight = lines.length * lineHeight + Math.max(0, lines.length - 1) * lineSpacing;
        if (totalHeight <= availableHeight || fontSize <= minimumFontSize) break;
        fontSize = Math.max(minimumFontSize, fontSize - shrinkStep); setPreviewFont();
    }
    const lineAdvance = lineHeight + lineSpacing;
    const startY = y - (lines.length - 1) * lineAdvance / 2;
    lines.forEach((line, index) => context.fillText(line, x, startY + index * lineAdvance));
}

function installBubbleEditor(node) {
    if (node.comfyClass !== "TUT_ComicSpeechBubble" || node.__tutComicBubbleEditor) return;
    const store = node.widgets?.find((widget) => widget.name === "bubble_data");
    const fontWidget = node.widgets?.find((widget) => widget.name === "default_font");
    if (!store || !fontWidget || typeof node.addDOMWidget !== "function") return;
    ensureBubbleStyle();
    hideStore(store);
    hideStore(fontWidget);
    const previousStoreCallback = store.callback;
    let data = parseBubbles(store.value), selected = -1, dragState = null, sourcePreview = null, previewScale = 1;

    const root = document.createElement("div"); root.className = "tut-bubble-wrap";
    const toolbar = document.createElement("div"); toolbar.className = "tut-bubble-toolbar";
    const tools = document.createElement("div"); tools.className = "tut-bubble-tools";
    const add = bubbleButton("＋ 添加对话框"); const remove = bubbleButton("删除");
    tools.append(add, remove);
    const count = document.createElement("span"); count.className = "tut-bubble-count";
    toolbar.append(tools, count);

    const workspace = document.createElement("div"); workspace.className = "tut-bubble-workspace";
    const monitor = document.createElement("div"); monitor.className = "tut-bubble-monitor";
    const screen = document.createElement("div"); screen.className = "tut-bubble-screen";
    const canvas = document.createElement("canvas"); canvas.className = "tut-bubble-canvas"; canvas.width = 960; canvas.height = 640;
    screen.append(canvas); monitor.append(screen);

    const sidebar = document.createElement("aside"); sidebar.className = "tut-bubble-sidebar";
    const tabs = document.createElement("div"); tabs.className = "tut-bubble-tabs";
    const sidebarBody = document.createElement("div"); sidebarBody.className = "tut-bubble-sidebar-body";
    const panels = {}, tabButtons = {};
    for (const [key, label] of [["content", "内容"], ["style", "样式"], ["layers", "图层"]]) {
        const tab = button(label); tab.className = "tut-bubble-tab"; tab.dataset.panel = key; tabs.append(tab); tabButtons[key] = tab;
        const panel = document.createElement("div"); panel.className = "tut-bubble-panel"; sidebarBody.append(panel); panels[key] = panel;
    }
    sidebar.append(tabs, sidebarBody); workspace.append(monitor, sidebar); root.append(toolbar, workspace);

    const selectedLabel = document.createElement("span"); selectedLabel.className = "tut-bubble-index";
    const text = document.createElement("textarea"); text.rows = 5; text.placeholder = "选择对话框后输入台词"; text.className = "tut-bubble-textarea";
    const dialogue = document.createElement("div"); dialogue.className = "tut-bubble-dialogue";
    const writingModes = document.createElement("div"); writingModes.className = "tut-bubble-writing";
    const horizontalText = bubbleButton("横", "横排文字");
    const verticalLtr = bubbleButton("→", "竖排文字：列从左到右"); verticalLtr.setAttribute("aria-label", "竖排列从左到右");
    const verticalRtl = bubbleButton("←", "竖排文字：列从右到左"); verticalRtl.setAttribute("aria-label", "竖排列从右到左");
    const writingHelp = document.createElement("span"); writingHelp.className = "tut-bubble-help"; writingHelp.textContent = "竖排列方向";
    writingModes.append(horizontalText, verticalLtr, verticalRtl, writingHelp); dialogue.append(text, writingModes);
    const shape = document.createElement("select"); shape.className = "tut-bubble-select"; BUBBLE_SHAPES.forEach((value) => shape.add(new Option(value, value)));
    const allFonts = [...(fontWidget.options?.values || [fontWidget.value])];
    let fontEntries = new Map(), fontCatalogReady = false, fontCatalogError = "";
    const fontSearch = document.createElement("input"); fontSearch.type = "search"; fontSearch.className = "tut-bubble-input tut-bubble-search";
    fontSearch.placeholder = "搜索字体名称或路径"; fontSearch.setAttribute("aria-label", "搜索字体");
    const fontSearchStatus = document.createElement("div"); fontSearchStatus.className = "tut-bubble-search-status";
    const fontPreviewStatus = document.createElement("div"); fontPreviewStatus.className = "tut-bubble-search-status";
    const font = document.createElement("select"); font.className = "tut-bubble-select";
    const fontSize = range("8", "160", "1"); fontSize.className = "tut-bubble-range";
    const fontSizeValue = document.createElement("span"); fontSizeValue.className = "tut-bubble-count";
    panels.content.append(
        selectedLabel, bubbleField("台词", dialogue), bubbleField("对话框形状", shape),
        bubbleField("搜索字体", fontSearch), fontSearchStatus, bubbleField("字体", font), fontPreviewStatus,
        bubbleField("字号", Object.assign(document.createElement("div"), { className: "tut-bubble-row" })),
    );
    panels.content.lastElementChild.lastElementChild.append(fontSize, fontSizeValue);

    const burstControls = document.createElement("div"); burstControls.className = "tut-bubble-burst";
    const spikeCount = range("6", "32", "1"); spikeCount.className = "tut-bubble-range";
    const spikeDepth = range("5", "70", "1"); spikeDepth.className = "tut-bubble-range";
    const spikeCountValue = document.createElement("span"), spikeDepthValue = document.createElement("span");
    spikeCountValue.className = spikeDepthValue.className = "tut-bubble-count";
    const spikeCountRow = document.createElement("div"); spikeCountRow.className = "tut-bubble-row"; spikeCountRow.append(spikeCount, spikeCountValue);
    const spikeDepthRow = document.createElement("div"); spikeDepthRow.className = "tut-bubble-row"; spikeDepthRow.append(spikeDepth, spikeDepthValue);
    burstControls.append(
        Object.assign(document.createElement("h4"), { textContent: "爆炸框形状" }),
        bubbleField("尖角数量", spikeCountRow), bubbleField("尖角深度", spikeDepthRow),
    );
    panels.content.append(burstControls);

    const cloudControls = document.createElement("div"); cloudControls.className = "tut-bubble-cloud";
    const cloudLobes = range("6", "16", "1"); cloudLobes.className = "tut-bubble-range";
    const cloudDepth = range("5", "30", "1"); cloudDepth.className = "tut-bubble-range";
    const cloudLobesValue = document.createElement("span"), cloudDepthValue = document.createElement("span");
    cloudLobesValue.className = cloudDepthValue.className = "tut-bubble-count";
    const cloudLobesRow = document.createElement("div"); cloudLobesRow.className = "tut-bubble-row"; cloudLobesRow.append(cloudLobes, cloudLobesValue);
    const cloudDepthRow = document.createElement("div"); cloudDepthRow.className = "tut-bubble-row"; cloudDepthRow.append(cloudDepth, cloudDepthValue);
    cloudControls.append(
        Object.assign(document.createElement("h4"), { textContent: "云朵框形状" }),
        bubbleField("云瓣数量", cloudLobesRow), bubbleField("云瓣起伏", cloudDepthRow),
    );
    panels.content.append(cloudControls);

    const opacity = range("0", "1", ".01"); opacity.className = "tut-bubble-range";
    const borderWidth = range("0", "32", "1"); borderWidth.className = "tut-bubble-range";
    const opacityValue = document.createElement("span"), borderValue = document.createElement("span");
    opacityValue.className = borderValue.className = "tut-bubble-count";
    const textColor = document.createElement("input"), fillColor = document.createElement("input"), borderColor = document.createElement("input");
    [textColor, fillColor, borderColor].forEach((input) => { input.type = "color"; input.className = "tut-bubble-input tut-bubble-color"; });
    const opacityRow = document.createElement("div"); opacityRow.className = "tut-bubble-row"; opacityRow.append(opacity, opacityValue);
    const borderRow = document.createElement("div"); borderRow.className = "tut-bubble-row"; borderRow.append(borderWidth, borderValue);
    const colorFields = document.createElement("div"); colorFields.className = "tut-bubble-fields";
    colorFields.append(bubbleField("文字颜色", textColor), bubbleField("填充颜色", fillColor), bubbleField("边框颜色", borderColor));
    const mergeOverlaps = document.createElement("input"); mergeOverlaps.type = "checkbox";
    const mergeToggle = document.createElement("label"); mergeToggle.className = "tut-bubble-toggle";
    const mergeCopy = document.createElement("span"); mergeCopy.className = "tut-bubble-toggle-copy";
    mergeCopy.append(
        Object.assign(document.createElement("span"), { textContent: "合并重叠边框" }),
        Object.assign(document.createElement("small"), { textContent: "相交气泡合为一个轮廓，并使用最上层气泡的外观。" }),
    );
    mergeToggle.append(mergeOverlaps, mergeCopy);
    panels.style.append(
        Object.assign(document.createElement("h4"), { textContent: "外观设置" }),
        mergeToggle, bubbleField("整体透明度", opacityRow), bubbleField("边框宽度", borderRow), colorFields,
    );

    const layerHelp = document.createElement("div"); layerHelp.className = "tut-bubble-help"; layerHelp.textContent = "列表顶部显示最前方的对话框。点击条目可选中并调整遮挡顺序。";
    const layerList = document.createElement("div"); layerList.className = "tut-bubble-layer-list";
    const layerActions = document.createElement("div"); layerActions.className = "tut-bubble-layer-actions";
    const toTop = bubbleButton("置于顶层"), moveUp = bubbleButton("上移一层"), moveDown = bubbleButton("下移一层"), toBottom = bubbleButton("置于底层");
    layerActions.append(toTop, moveUp, moveDown, toBottom);
    panels.layers.append(Object.assign(document.createElement("h4"), { textContent: "对话框图层" }), layerHelp, layerList, layerActions);

    let activePanel = "content";
    function setActivePanel(key) {
        activePanel = key;
        for (const [name, panel] of Object.entries(panels)) panel.classList.toggle("active", name === key);
        for (const [name, tab] of Object.entries(tabButtons)) tab.classList.toggle("active", name === key);
    }
    for (const [key, tab] of Object.entries(tabButtons)) tab.addEventListener("click", () => setActivePanel(key));
    setActivePanel(activePanel);

    const context = canvas.getContext("2d");
    function save() { commit(node, store, data, previousStoreCallback); }
    function selectedBubble() { return data.bubbles[selected] || null; }
    let lastFontQuery = null, lastFontSelection = null;
    function filterFonts() {
        const query = fontSearch.value.trim().toLocaleLowerCase();
        const currentFont = selectedBubble()?.font_name || fontWidget.value;
        if (!fontCatalogReady && !fontCatalogError) {
            font.replaceChildren(Object.assign(new Option("正在读取字体信息…", ""), { disabled: true }));
            font.disabled = true; fontSearchStatus.textContent = "正在读取字体信息…"; return;
        }
        if (query === lastFontQuery && currentFont === lastFontSelection && font.options.length) {
            font.disabled = !selectedBubble() || (font.options.length === 1 && font.options[0].disabled);
            return;
        }
        lastFontQuery = query; lastFontSelection = currentFont;
        const matches = allFonts.filter((value) => {
            const entry = fontEntries.get(value);
            const searchable = entry ? `${entry.display_name} ${entry.search_text || ""}` : String(value);
            return !query || searchable.toLocaleLowerCase().includes(query);
        });
        font.replaceChildren();
        const currentIsMatch = matches.includes(currentFont);
        if (currentFont && !currentIsMatch) {
            const currentLabel = fontEntries.get(currentFont)?.display_name || currentFont;
            font.add(new Option(`当前：${currentLabel}`, currentFont));
        }
        if (!matches.length) {
            const option = new Option("未找到匹配字体", ""); option.disabled = true; font.add(option);
            font.value = currentFont;
            fontSearchStatus.textContent = "未找到匹配字体；当前字体未改变";
        } else {
            for (const value of matches) font.add(new Option(fontEntries.get(value)?.display_name || value, value));
            font.value = currentFont;
            fontSearchStatus.textContent = query
                ? `找到 ${matches.length} 个字体${currentIsMatch ? "" : "；当前字体未改变，请在下拉框中选择"}`
                : `共 ${matches.length} 个字体`;
        }
        font.disabled = !selectedBubble();
    }
    const watchedFonts = new Set();
    function watchFont(token) {
        const entry = fontEntries.get(token);
        if (!entry || watchedFonts.has(token)) return;
        watchedFonts.add(token); updateFontPreviewStatus();
        ensureComicFontLoaded(entry).then(() => {
            if (root.isConnected) { drawCanvas(); updateFontPreviewStatus(); }
        }).catch(() => {
            if (root.isConnected) updateFontPreviewStatus();
        });
    }
    function updateFontPreviewStatus() {
        if (fontCatalogError) { fontPreviewStatus.textContent = `字体目录加载失败：${fontCatalogError}`; return; }
        if (!fontCatalogReady) { fontPreviewStatus.textContent = "正在准备字体预览…"; return; }
        const token = selectedBubble()?.font_name;
        if (!token) { fontPreviewStatus.textContent = ""; return; }
        const record = COMIC_FONT_LOADS.get(token);
        if (!fontEntries.has(token)) fontPreviewStatus.textContent = "旧路径字体无法在浏览器预览，最终输出仍可正常尝试渲染。";
        else if (!record || record.status === "loading") fontPreviewStatus.textContent = "正在加载所选字体预览…";
        else if (record.status === "failed") fontPreviewStatus.textContent = `字体预览加载失败：${record.error}；最终输出不受影响。`;
        else fontPreviewStatus.textContent = "预览已使用所选字体";
    }
    function drawBubble(bubble, index, parts = { fill: true, stroke: true, text: true, selection: true }) {
        const style = parts.style || bubble;
        const x = bubble.x * canvas.width, y = bubble.y * canvas.height;
        const w = bubble.w * canvas.width, h = bubble.h * canvas.height;
        context.save(); context.globalAlpha = clamp(style.opacity);
        context.fillStyle = style.fill_color || "#fff"; context.strokeStyle = style.border_color || "#111";
        const scaledBorderWidth = (style.border_width || 0) * previewScale;
        context.lineWidth = Math.max(.5, parts.merged ? scaledBorderWidth * 2 : scaledBorderWidth || .5);
        drawBubblePath(context, bubble, x, y, w, h);
        if (bubble.shape === "闪光对话框") {
            if (parts.fill) drawFlashFill(context, style.fill_color || "#ffffff", x, y, w, h);
            if (parts.stroke && (style.border_width || 0) > 0) {
                context.fillStyle = style.border_color || "#111";
                drawFlashRays(context, style, x, y, w, h);
            }
        } else if (bubble.shape !== "无边框文字") {
            if (parts.fill) context.fill();
            if (parts.stroke && (style.border_width || 0) > 0) context.stroke();
        }
        if (parts.text) drawBubbleText(context, bubble, x, y, w, h, previewScale);
        context.restore();
        if (parts.selection && index === selected) {
            context.strokeStyle = "#35d0c8"; context.lineWidth = 3; context.strokeRect(x - w / 2 - 4, y - h / 2 - 4, w + 8, h + 8);
            context.fillStyle = "#35d0c8";
            for (const [hx, hy] of [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x - w / 2, y + h / 2], [x + w / 2, y + h / 2]]) context.fillRect(hx - 6, hy - 6, 12, 12);
        }
    }
    function renderLayerList() {
        layerList.replaceChildren();
        [...data.bubbles].map((bubble, index) => ({ bubble, index })).reverse().forEach(({ bubble, index }, rank) => {
            const row = document.createElement("div"); row.className = `tut-bubble-layer-row${index === selected ? " active" : ""}`;
            const rankLabel = document.createElement("span"); rankLabel.className = "tut-bubble-layer-rank"; rankLabel.textContent = String(rank + 1);
            const name = document.createElement("span"); name.textContent = String(bubble.text || bubble.shape || `对话框 ${index + 1}`).replace(/\s+/g, " ").slice(0, 18);
            const position = document.createElement("span"); position.className = "tut-bubble-help"; position.textContent = rank === 0 ? "最前" : rank === data.bubbles.length - 1 ? "最底" : "";
            row.append(rankLabel, name, position); row.addEventListener("click", () => { selected = index; render(); }); layerList.append(row);
        });
    }
    function resizeCanvasToPreview() {
        const naturalWidth = sourcePreview?.naturalWidth || 960, naturalHeight = sourcePreview?.naturalHeight || 640;
        const scale = Math.min(960 / naturalWidth, 680 / naturalHeight, 1);
        const nextWidth = Math.max(160, Math.round(naturalWidth * scale)), nextHeight = Math.max(120, Math.round(naturalHeight * scale));
        if (canvas.width !== nextWidth) canvas.width = nextWidth;
        if (canvas.height !== nextHeight) canvas.height = nextHeight;
        previewScale = sourcePreview?.naturalWidth ? Math.min(nextWidth / naturalWidth, nextHeight / naturalHeight) : 1;
        screen.style.aspectRatio = `${canvas.width} / ${canvas.height}`;
        screen.style.width = canvas.width >= canvas.height ? "100%" : "auto";
        screen.style.height = canvas.height > canvas.width ? "100%" : "auto";
    }
    function drawCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (sourcePreview?.complete && sourcePreview.naturalWidth) context.drawImage(sourcePreview, 0, 0, canvas.width, canvas.height);
        else { context.fillStyle = "#24282d"; context.fillRect(0, 0, canvas.width, canvas.height); }
        if (fontCatalogReady) for (const token of new Set(data.bubbles.map((bubble) => bubble.font_name))) watchFont(token);
        if (data.merge_overlaps) {
            const parents = data.bubbles.map((_, index) => index);
            const find = (index) => { while (parents[index] !== index) { parents[index] = parents[parents[index]]; index = parents[index]; } return index; };
            const join = (left, right) => { left = find(left); right = find(right); if (left !== right) parents[right] = left; };
            for (let left = 0; left < data.bubbles.length; left++) for (let right = left + 1; right < data.bubbles.length; right++) {
                const a = data.bubbles[left], b = data.bubbles[right];
                if (!["无边框文字", "闪光对话框"].includes(a.shape) && !["无边框文字", "闪光对话框"].includes(b.shape) && Math.abs(a.x - b.x) < (a.w + b.w) / 2 && Math.abs(a.y - b.y) < (a.h + b.h) / 2) join(left, right);
            }
            const groups = new Map();
            data.bubbles.forEach((_, index) => { const root = find(index); if (!groups.has(root)) groups.set(root, []); groups.get(root).push(index); });
            for (const indices of groups.values()) {
                const style = data.bubbles[Math.max(...indices)];
                if (indices.length === 1 && style.shape === "闪光对话框") {
                    indices.forEach((index) => drawBubble(data.bubbles[index], index, { fill: true, stroke: false, text: false, selection: false, style }));
                    indices.forEach((index) => drawBubble(data.bubbles[index], index, { fill: false, stroke: true, text: false, selection: false, merged: true, style }));
                } else {
                    indices.forEach((index) => drawBubble(data.bubbles[index], index, { fill: false, stroke: true, text: false, selection: false, merged: true, style }));
                    indices.forEach((index) => drawBubble(data.bubbles[index], index, { fill: true, stroke: false, text: false, selection: false, style }));
                }
            }
            data.bubbles.forEach((bubble, index) => drawBubble(bubble, index, { fill: false, stroke: false, text: true, selection: false }));
            data.bubbles.forEach((bubble, index) => drawBubble(bubble, index, { fill: false, stroke: false, text: false, selection: true }));
        } else data.bubbles.forEach((bubble, index) => drawBubble(bubble, index));
    }
    function render() {
        resizeCanvasToPreview();
        drawCanvas();
        const bubble = selectedBubble(), disabled = !bubble;
        mergeOverlaps.checked = data.merge_overlaps === true;
        [text, horizontalText, verticalLtr, verticalRtl, shape, fontSearch, fontSize, opacity, borderWidth, textColor, fillColor, borderColor, remove, toTop, moveUp, moveDown, toBottom, spikeCount, spikeDepth, cloudLobes, cloudDepth].forEach((control) => control.disabled = disabled);
        count.textContent = `${data.bubbles.length} / 32 个对话框`;
        selectedLabel.textContent = bubble ? `已选择：对话框 ${selected + 1}` : "未选择对话框";
        if (bubble) {
            text.value = bubble.text || ""; shape.value = bubble.shape;
            fontSize.value = bubble.font_size; opacity.value = bubble.opacity; borderWidth.value = bubble.border_width;
            textColor.value = bubble.text_color; fillColor.value = bubble.fill_color; borderColor.value = bubble.border_color;
            fontSizeValue.textContent = `${bubble.font_size}px`; opacityValue.textContent = `${Math.round(bubble.opacity * 100)}%`; borderValue.textContent = `${bubble.border_width}px`;
            spikeCount.value = bubble.spike_count ?? BUBBLE_SPIKE_COUNT_DEFAULT; spikeDepth.value = Math.round((bubble.spike_depth ?? BUBBLE_SPIKE_DEPTH_DEFAULT) * 100);
            spikeCountValue.textContent = `${spikeCount.value} 个`; spikeDepthValue.textContent = `${spikeDepth.value}%`;
            cloudLobes.value = bubble.cloud_lobes ?? BUBBLE_CLOUD_LOBES_DEFAULT; cloudDepth.value = Math.round((bubble.cloud_depth ?? BUBBLE_CLOUD_DEPTH_DEFAULT) * 100);
            cloudLobesValue.textContent = `${cloudLobes.value} 瓣`; cloudDepthValue.textContent = `${cloudDepth.value}%`;
        } else { text.value = ""; fontSizeValue.textContent = "—"; opacityValue.textContent = "—"; borderValue.textContent = "—"; spikeCountValue.textContent = "—"; spikeDepthValue.textContent = "—"; cloudLobesValue.textContent = "—"; cloudDepthValue.textContent = "—"; }
        const textDirection = bubble?.text_direction || "horizontal";
        horizontalText.classList.toggle("active", textDirection === "horizontal");
        verticalLtr.classList.toggle("active", textDirection === "vertical_ltr");
        verticalRtl.classList.toggle("active", textDirection === "vertical_rtl");
        burstControls.classList.toggle("active", bubble?.shape === "爆炸喊话框");
        cloudControls.classList.toggle("active", bubble?.shape === "云朵思考框");
        filterFonts();
        updateFontPreviewStatus();
        renderLayerList();
    }
    function update(field, value) { const bubble = selectedBubble(); if (!bubble) return; bubble[field] = value; save(); render(); }
    function point(event) { const bounds = canvas.getBoundingClientRect(); return { x: clamp((event.clientX - bounds.left) / bounds.width), y: clamp((event.clientY - bounds.top) / bounds.height) }; }
    function hit(pointValue) {
        for (let index = data.bubbles.length - 1; index >= 0; index--) { const bubble = data.bubbles[index]; if (Math.abs(pointValue.x - bubble.x) <= bubble.w / 2 && Math.abs(pointValue.y - bubble.y) <= bubble.h / 2) return index; }
        return -1;
    }
    const down = (event) => {
        event.preventDefault(); const p = point(event), current = selectedBubble();
        const corners = current ? [["nw", current.x - current.w / 2, current.y - current.h / 2], ["ne", current.x + current.w / 2, current.y - current.h / 2], ["sw", current.x - current.w / 2, current.y + current.h / 2], ["se", current.x + current.w / 2, current.y + current.h / 2]] : [];
        const corner = corners.find(([, x, y]) => Math.hypot((p.x - x) * canvas.width, (p.y - y) * canvas.height) <= 18)?.[0];
        if (corner) dragState = { kind: "resize", corner, start: p, original: clone(current) };
        else {
            selected = hit(p); const bubble = selectedBubble();
            dragState = bubble ? { kind: "move", offsetX: p.x - bubble.x, offsetY: p.y - bubble.y } : null;
        }
        canvas.setPointerCapture?.(event.pointerId); render();
    };
    const move = (event) => {
        const bubble = selectedBubble(); if (!bubble || !dragState) return; const p = point(event);
        if (dragState.kind === "resize") {
            const original = dragState.original, left = original.x - original.w / 2, right = original.x + original.w / 2, top = original.y - original.h / 2, bottom = original.y + original.h / 2;
            const nextLeft = dragState.corner.includes("w") ? Math.min(p.x, right - .03) : left;
            const nextRight = dragState.corner.includes("e") ? Math.max(p.x, left + .03) : right;
            const nextTop = dragState.corner.includes("n") ? Math.min(p.y, bottom - .03) : top;
            const nextBottom = dragState.corner.includes("s") ? Math.max(p.y, top + .03) : bottom;
            const boundedLeft = clamp(nextLeft), boundedRight = clamp(nextRight), boundedTop = clamp(nextTop), boundedBottom = clamp(nextBottom);
            bubble.w = Math.max(.03, boundedRight - boundedLeft); bubble.h = Math.max(.03, boundedBottom - boundedTop);
            bubble.x = (boundedLeft + boundedRight) / 2; bubble.y = (boundedTop + boundedBottom) / 2;
        } else {
            bubble.x = clamp(p.x - dragState.offsetX, bubble.w / 2, 1 - bubble.w / 2);
            bubble.y = clamp(p.y - dragState.offsetY, bubble.h / 2, 1 - bubble.h / 2);
        }
        drawCanvas();
    };
    const up = (event) => { if (dragState) save(); dragState = null; canvas.releasePointerCapture?.(event.pointerId); };
    canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move); canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
    add.addEventListener("click", () => {
        if (data.bubbles.length >= 32) return;
        const inheritedFont = selectedBubble()?.font_name || fontWidget.value;
        data.bubbles.push(newBubble(inheritedFont, data.bubbles.length)); selected = data.bubbles.length - 1; save(); render();
    });
    remove.addEventListener("click", () => { if (selected < 0) return; data.bubbles.splice(selected, 1); selected = Math.min(selected, data.bubbles.length - 1); save(); render(); });
    function moveLayer(target) {
        if (selected < 0 || target < 0 || target >= data.bubbles.length || target === selected) return;
        const [bubble] = data.bubbles.splice(selected, 1); data.bubbles.splice(target, 0, bubble); selected = target; save(); render();
    }
    toTop.addEventListener("click", () => moveLayer(data.bubbles.length - 1)); moveUp.addEventListener("click", () => moveLayer(selected + 1));
    moveDown.addEventListener("click", () => moveLayer(selected - 1)); toBottom.addEventListener("click", () => moveLayer(0));
    text.addEventListener("input", () => update("text", text.value)); shape.addEventListener("change", () => update("shape", shape.value)); font.addEventListener("change", () => update("font_name", font.value));
    horizontalText.addEventListener("click", () => update("text_direction", "horizontal"));
    verticalLtr.addEventListener("click", () => update("text_direction", "vertical_ltr"));
    verticalRtl.addEventListener("click", () => update("text_direction", "vertical_rtl"));
    fontSearch.addEventListener("input", filterFonts);
    mergeOverlaps.addEventListener("change", () => { data.merge_overlaps = mergeOverlaps.checked; save(); render(); });
    fontSize.addEventListener("input", () => update("font_size", Number(fontSize.value))); opacity.addEventListener("input", () => update("opacity", Number(opacity.value))); borderWidth.addEventListener("input", () => update("border_width", Number(borderWidth.value)));
    spikeCount.addEventListener("input", () => update("spike_count", Number(spikeCount.value))); spikeDepth.addEventListener("input", () => update("spike_depth", Number(spikeDepth.value) / 100));
    cloudLobes.addEventListener("input", () => update("cloud_lobes", Number(cloudLobes.value))); cloudDepth.addEventListener("input", () => update("cloud_depth", Number(cloudDepth.value) / 100));
    textColor.addEventListener("input", () => update("text_color", textColor.value)); fillColor.addEventListener("input", () => update("fill_color", fillColor.value)); borderColor.addEventListener("input", () => update("border_color", borderColor.value));
    store.callback = function (...args) { const result = previousStoreCallback?.apply(this, args); data = parseBubbles(store.value); selected = Math.min(selected, data.bubbles.length - 1); render(); return result; };
    node.__tutBubbleSetPreviews = (items) => {
        const url = bubblePreviewUrl(items?.[0]);
        if (!url) { sourcePreview = null; render(); return; }
        const image = new Image(); image.onload = () => { sourcePreview = image; render(); }; image.onerror = () => { sourcePreview = null; render(); }; image.src = url;
    };
    loadComicFontCatalog().then((entries) => {
        fontEntries = new Map(entries.map((entry) => [entry.token, entry]));
        fontCatalogReady = true; lastFontQuery = null; lastFontSelection = null; render();
    }).catch((error) => {
        fontCatalogError = error?.message || "未知错误"; lastFontQuery = null; lastFontSelection = null; render();
    });
    const widget = node.addDOMWidget("tut_comic_bubble_editor", "TUT_COMIC_BUBBLE_EDITOR", root, { serialize: false, hideOnZoom: false });
    widget.computeSize = () => [Math.max(1280, Number(node.size?.[0]) || 0), 820];
    node.__tutComicBubbleEditor = true;
    const syncDomWidth = (size = node.size) => {
      const logicalWidth = Math.max(1280, Number(size?.[0]) || 0);
      root.style.width = `${Math.max(320, logicalWidth - 20)}px`;
      widget.computeSize = () => [logicalWidth, 820];
    };
    let enforcingMinimumSize = false;
    const priorResize = node.onResize;
    node.onResize = function (size) {
      if (!enforcingMinimumSize && (Number(size?.[0]) < 1280 || Number(size?.[1]) < 960)) {
        enforcingMinimumSize = true;
        this.setSize?.([Math.max(1280, Number(size?.[0]) || 0), Math.max(960, Number(size?.[1]) || 0)]);
        enforcingMinimumSize = false;
        size = this.size;
      }
      priorResize?.call(this, size); syncDomWidth(size); requestAnimationFrame(drawCanvas);
    };
    const ensureDefaultSize = () => node.setSize?.([Math.max(node.size?.[0] || 0, 1280), Math.max(node.size?.[1] || 0, 960)]);
    syncDomWidth();
    ensureDefaultSize(); render();
    requestAnimationFrame(() => { ensureDefaultSize(); render(); setTimeout(() => { ensureDefaultSize(); render(); }, 100); });
}

app.registerExtension({
    name: "TUT_Nodes.ComicEditors",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TUT_ComicSpeechBubble" || nodeType.prototype.__tutBubblePreviewHook) return;
        const previousExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = previousExecuted?.apply(this, arguments);
            this.__tutBubbleSetPreviews?.(output?.input_previews || []);
            return result;
        };
        nodeType.prototype.__tutBubblePreviewHook = true;
    },
    nodeCreated(node) { installBubbleEditor(node); },
});
