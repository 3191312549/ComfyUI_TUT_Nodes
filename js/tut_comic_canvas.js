import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const AUTO_LAYOUT = "自动匹配数量";
const CUSTOM_LAYOUT = "自由画框";
const MAX_PANELS = 6;
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
    [CUSTOM_LAYOUT]: [[.08, .08, .92, .92]],
};
const AUTO_PREVIEW_LAYOUT = "六宫格";
const SIZE_PRESETS = [
    ["1024 × 1024（1:1）", 1024, 1024], ["1536 × 1536（1:1）", 1536, 1536],
    ["1024 × 1344（16:21）", 1024, 1344], ["1024 × 1536（2:3）", 1024, 1536],
    ["1152 × 1536（3:4）", 1152, 1536], ["1536 × 2048（3:4）", 1536, 2048],
    ["864 × 1536（9:16）", 864, 1536], ["1536 × 864（16:9）", 1536, 864],
    ["1344 × 1024（21:16）", 1344, 1024], ["1536 × 1024（3:2）", 1536, 1024],
    ["1536 × 1152（4:3）", 1536, 1152], ["2048 × 1536（4:3）", 2048, 1536],
];
const WIDGET_NAMES = [
    "layout", "canvas_width", "canvas_height", "page_margin", "gutter", "border_width",
    "border_color", "background_color", "fit_mode", "empty_fill", "panel_data",
];

const clone = (value) => JSON.parse(JSON.stringify(value));
const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, Number(value) || 0));

function previewUrl(data) {
    if (!data?.filename) return null;
    const query = new URLSearchParams({ filename: data.filename, type: data.type ?? "temp", subfolder: data.subfolder ?? "" });
    const format = app.getPreviewFormatParam?.() ?? "";
    const random = app.getRandParam?.() ?? `&rand=${Math.random()}`;
    return api.apiURL(`/view?${query.toString()}${format}${random}`);
}

function ensureStyle() {
    if (document.getElementById("tut-comic-canvas-v2-style")) return;
    const style = document.createElement("style");
    style.id = "tut-comic-canvas-v2-style";
    style.textContent = `
      .tut-comic-wrap{display:flex;flex-direction:column;gap:8px;height:100%;min-height:0;color:#ddd;font:12px sans-serif;overflow:hidden}
      .tut-comic-toolbar,.tut-comic-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
      .tut-comic-toolbar{justify-content:space-between;flex-wrap:nowrap}.tut-comic-tools{display:flex;gap:5px;align-items:center;flex-wrap:nowrap;min-width:0}.tut-comic-options{display:flex;gap:8px;align-items:center;flex-wrap:nowrap;white-space:nowrap}
      .tut-comic-btn,.tut-comic-select,.tut-comic-input{background:#25292e;color:#eee;border:1px solid #555;border-radius:5px;padding:4px 7px;box-sizing:border-box}
      .tut-comic-btn{cursor:pointer;white-space:nowrap}.tut-comic-btn:hover{border-color:#aaa}.tut-comic-btn.active{background:#167d86;border-color:#35d0c8;color:#fff}.tut-comic-btn:disabled{opacity:.35;cursor:not-allowed}
      .tut-comic-workspace{display:grid;grid-template-columns:minmax(0,1fr) 288px;gap:8px;flex:1;min-height:0;overflow:hidden}.tut-comic-monitor{background:#070707;border:1px solid #333;border-radius:8px;padding:10px;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden;min-width:0}
      .tut-comic-screen{position:relative;background:#f7f4ec;border:2px solid #555;border-radius:0;overflow:hidden;line-height:0;box-shadow:none}
      .tut-comic-bg,.tut-comic-overlay{position:absolute;inset:0;display:block}.tut-comic-overlay{touch-action:none;cursor:crosshair}
      .tut-comic-sidebar{background:#15171a;border:1px solid #34383d;border-radius:7px;display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}.tut-comic-sidebar-tabs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:#30343a;border-bottom:1px solid #3a3e44}.tut-comic-tab{border:0;border-radius:0;background:#202328;color:#aaa;padding:8px 3px;cursor:pointer;font-weight:700}.tut-comic-tab.active{background:#167d86;color:#fff}.tut-comic-sidebar-body{padding:9px;min-height:0;overflow:auto;flex:1}
      .tut-comic-panel{min-width:0;display:none;flex-direction:column;gap:9px}.tut-comic-panel.active{display:flex}
      .tut-comic-panel h4{margin:0;color:#35d0c8;font-size:12px}.tut-comic-label{color:#aaa;font-size:10px}
      .tut-comic-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.tut-comic-fields.three{grid-template-columns:repeat(3,minmax(0,1fr))}
      .tut-comic-field{display:flex;flex-direction:column;gap:3px;min-width:0;color:#aaa;font-size:10px}.tut-comic-field>.tut-comic-input,.tut-comic-field>.tut-comic-select{width:100%;min-width:0}
      .tut-comic-num{width:100%;text-align:right}.tut-comic-color{width:100%;height:27px;padding:1px}.tut-comic-wide{width:100%;min-width:0}
      .tut-comic-camera-row{display:grid;grid-template-columns:auto minmax(70px,1fr) auto;gap:6px;align-items:center}.tut-comic-actions{display:flex;gap:6px;align-items:center;flex-wrap:nowrap}
      .tut-comic-edge-row{display:grid;grid-template-columns:auto repeat(4,minmax(42px,1fr));gap:5px;align-items:center}.tut-comic-edge-toggle{display:flex;gap:3px;align-items:center;justify-content:center;white-space:nowrap;background:#25292e;border:1px solid #555;border-radius:5px;padding:4px 5px;box-sizing:border-box}.tut-comic-edge-toggle:has(input:checked){background:#167d86;border-color:#35d0c8;color:#fff}.tut-comic-edge-toggle input{margin:0}
      .tut-comic-range{width:100%;min-width:0}.tut-comic-index{display:block;font-weight:700;color:#fff;background:#167d86;border-radius:4px;padding:4px 7px}
      .tut-comic-layer-help{color:#92979e;font-size:10px;line-height:1.45}.tut-comic-layer-list{display:flex;flex-direction:column;gap:5px}.tut-comic-layer-row{display:grid;grid-template-columns:22px minmax(0,1fr) auto;align-items:center;gap:6px;background:#22262b;border:1px solid #444;border-radius:5px;padding:6px 7px;color:#ddd;cursor:pointer}.tut-comic-layer-row.active{border-color:#35d0c8;background:#164f55}.tut-comic-layer-rank{color:#8f979f;text-align:center;font-variant-numeric:tabular-nums}.tut-comic-layer-row.active .tut-comic-layer-rank{color:#fff}.tut-comic-layer-actions{display:grid;grid-template-columns:1fr 1fr;gap:6px}.tut-comic-layer-actions .tut-comic-btn{width:100%}
      @media (max-width:900px){.tut-comic-workspace{grid-template-columns:minmax(0,1fr) 250px}}
    `;
    document.head.appendChild(style);
}

function makeButton(label, title = "") {
    const element = document.createElement("button");
    element.type = "button"; element.className = "tut-comic-btn";
    element.textContent = label; element.title = title;
    return element;
}

function makeSelect(values) {
    const element = document.createElement("select");
    element.className = "tut-comic-select";
    for (const value of values) element.add(new Option(value, value));
    return element;
}

function makeNumber(min, max, step = 1) {
    const element = document.createElement("input");
    element.type = "number"; element.min = String(min); element.max = String(max); element.step = String(step);
    element.className = "tut-comic-input tut-comic-num";
    return element;
}

function makeField(label, control) {
    const field = document.createElement("label"); field.className = "tut-comic-field";
    const caption = document.createElement("span"); caption.textContent = label;
    field.append(caption, control); return field;
}

function makeFields(fields, columns = 2) {
    const group = document.createElement("div"); group.className = `tut-comic-fields${columns === 3 ? " three" : ""}`;
    group.append(...fields); return group;
}

function parseData(value) {
    const fallback = {
        version: 1,
        panels: Array.from({ length: MAX_PANELS }, () => ({ focus_x: .5, focus_y: .5, zoom: 1, flip: false, overflow_top: false, overflow_bottom: false, overflow_left: false, overflow_right: false })),
        layout_overrides: {},
        layer_orders: {},
    };
    try {
        const parsed = JSON.parse(value || "{}");
        if (parsed?.version !== 1 || !Array.isArray(parsed.panels)) return fallback;
        parsed.panels.slice(0, MAX_PANELS).forEach((item, index) => {
            fallback.panels[index] = {
                focus_x: clamp(item?.focus_x), focus_y: clamp(item?.focus_y),
                zoom: clamp(item?.zoom ?? 1, .25, 4), flip: item?.flip === true,
                overflow_top: item?.overflow_top === true, overflow_bottom: item?.overflow_bottom === true,
                overflow_left: item?.overflow_left === true, overflow_right: item?.overflow_right === true,
            };
        });
        if (parsed.layout_overrides && typeof parsed.layout_overrides === "object") {
            for (const [name, rectangles] of Object.entries(parsed.layout_overrides)) {
                if (!PANEL_LAYOUTS[name] || !Array.isArray(rectangles)) continue;
                const expected = name === CUSTOM_LAYOUT ? [1, MAX_PANELS] : [PANEL_LAYOUTS[name].length, PANEL_LAYOUTS[name].length];
                if (rectangles.length < expected[0] || rectangles.length > expected[1]) continue;
                const clean = rectangles.map((rect) => {
                    if (!Array.isArray(rect) || rect.length !== 4) return null;
                    const out = rect.map((number) => clamp(number));
                    return out[2] > out[0] && out[3] > out[1] ? out : null;
                });
                if (clean.every(Boolean)) fallback.layout_overrides[name] = clean;
            }
        }
        if (parsed.layer_orders && typeof parsed.layer_orders === "object") {
            for (const [name, order] of Object.entries(parsed.layer_orders)) {
                if (!PANEL_LAYOUTS[name] || !Array.isArray(order)) continue;
                const count = (fallback.layout_overrides[name] || PANEL_LAYOUTS[name]).length;
                if (order.length === count && order.every(Number.isInteger) && [...order].sort((a, b) => a - b).every((value, index) => value === index)) fallback.layer_orders[name] = [...order];
            }
        }
    } catch (_) {}
    return fallback;
}

function hideWidget(widget) {
    if (!widget) return;
    widget.serialize = true;
    widget.options = widget.options || {};
    widget.options.serialize = true;
    widget.options.hidden = true;
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
}

function installComicCanvas(node) {
    if (node.comfyClass !== "TUT_ComicPanelCanvas" || node.__tutComicCanvasV2) return;
    const widgets = Object.fromEntries((node.widgets || []).map((widget) => [widget.name, widget]));
    if (WIDGET_NAMES.some((name) => !widgets[name]) || typeof node.addDOMWidget !== "function") return;
    ensureStyle();
    WIDGET_NAMES.forEach((name) => hideWidget(widgets[name]));
    const priorStoreCallback = widgets.panel_data.callback;
    let data = parseData(widgets.panel_data.value);
    let selected = 0, mode = "frame", drawMode = false, previewImages = [];
    let moving = null, resizing = null, drawing = null, snapGuides = [];
    let snapEnabled = node.properties?.tutComicCanvasState?.snap_enabled !== false;
    let activeSidebar = node.properties?.tutComicCanvasState?.sidebar_tab || "camera";

    const root = document.createElement("div"); root.className = "tut-comic-wrap";
    const toolbar = document.createElement("div"); toolbar.className = "tut-comic-toolbar";
    const tools = document.createElement("div"); tools.className = "tut-comic-tools";
    const layout = makeSelect([AUTO_LAYOUT, ...Object.keys(PANEL_LAYOUTS)]);
    const frameBtn = makeButton("▭ 调整画框"), cameraBtn = makeButton("＋ 调整镜头");
    const drawBtn = makeButton("✎ 绘制画框", "自由画框模式下拖动空白处创建画框");
    const deleteBtn = makeButton("删除画框"), resetBtn = makeButton("恢复当前模板");
    const snapToggle = document.createElement("label"); snapToggle.className = "tut-comic-row";
    const snapCheck = document.createElement("input"); snapCheck.type = "checkbox"; snapCheck.checked = snapEnabled;
    snapToggle.append(snapCheck, document.createTextNode("吸附对齐"));
    const gridToggle = document.createElement("label"); gridToggle.className = "tut-comic-row";
    const gridCheck = document.createElement("input"); gridCheck.type = "checkbox"; gridCheck.checked = true;
    gridToggle.append(gridCheck, document.createTextNode("参考网格"));
    const toolbarOptions = document.createElement("div"); toolbarOptions.className = "tut-comic-options"; toolbarOptions.append(snapToggle, gridToggle);
    tools.append(layout, frameBtn, cameraBtn, drawBtn, deleteBtn, resetBtn);
    toolbar.append(tools, toolbarOptions); root.append(toolbar);

    const monitor = document.createElement("div"); monitor.className = "tut-comic-monitor";
    const screen = document.createElement("div"); screen.className = "tut-comic-screen";
    const bg = document.createElement("canvas"); bg.className = "tut-comic-bg";
    const overlay = document.createElement("canvas"); overlay.className = "tut-comic-overlay";
    screen.append(bg, overlay); monitor.append(screen);

    const workspace = document.createElement("div"); workspace.className = "tut-comic-workspace";
    const sidebar = document.createElement("aside"); sidebar.className = "tut-comic-sidebar";
    const sidebarTabs = document.createElement("div"); sidebarTabs.className = "tut-comic-sidebar-tabs";
    const panels = document.createElement("div"); panels.className = "tut-comic-sidebar-body";
    const sizePanel = document.createElement("div"); sizePanel.className = "tut-comic-panel";
    sizePanel.innerHTML = "<h4>画布与间距</h4>";
    const sizePreset = makeSelect([...SIZE_PRESETS.map((item) => item[0]), "自定义"]);
    const width = makeNumber(64, 8192, 8), height = makeNumber(64, 8192, 8);
    const margin = makeNumber(0, 2048), gutter = makeNumber(0, 1024);
    sizePanel.append(
        makeField("尺寸预设", Object.assign(sizePreset, { className: "tut-comic-select tut-comic-wide" })),
        makeFields([makeField("宽", width), makeField("高", height)]),
        makeFields([makeField("强制页边距", margin), makeField("格间距", gutter)]),
    );

    const settingsPanel = document.createElement("div"); settingsPanel.className = "tut-comic-panel";
    settingsPanel.innerHTML = "<h4>画框与镜头</h4>";
    const selectedLabel = document.createElement("span"); selectedLabel.className = "tut-comic-index";
    const zoom = document.createElement("input"); zoom.type = "range"; zoom.min = ".25"; zoom.max = "4"; zoom.step = ".05"; zoom.className = "tut-comic-range";
    const zoomValue = document.createElement("span"), flip = makeButton("水平翻转"), resetCamera = makeButton("重置镜头");
    const cameraRow = document.createElement("div"); cameraRow.className = "tut-comic-camera-row";
    cameraRow.append(document.createTextNode("缩放"), zoom, zoomValue);
    const cameraActions = document.createElement("div"); cameraActions.className = "tut-comic-actions"; cameraActions.append(flip, resetCamera);
    const edgeRow = document.createElement("div"); edgeRow.className = "tut-comic-edge-row";
    edgeRow.append(document.createTextNode("开放边缘"));
    const edgeChecks = {};
    for (const [key, label, ariaLabel] of [["overflow_top", "上", "上边缘开放"], ["overflow_bottom", "下", "下边缘开放"], ["overflow_left", "左", "左边缘开放"], ["overflow_right", "右", "右边缘开放"]]) {
        const toggle = document.createElement("label"); toggle.className = "tut-comic-edge-toggle";
        const check = document.createElement("input"); check.type = "checkbox"; check.setAttribute("aria-label", ariaLabel);
        toggle.append(check, document.createTextNode(label)); edgeChecks[key] = check; edgeRow.append(toggle);
    }
    settingsPanel.append(selectedLabel, cameraRow, cameraActions, edgeRow);

    const stylePanel = document.createElement("div"); stylePanel.className = "tut-comic-panel";
    stylePanel.innerHTML = "<h4>页面样式</h4>";
    const borderWidth = makeNumber(0, 128), borderColor = document.createElement("input"), backgroundColor = document.createElement("input");
    borderColor.type = backgroundColor.type = "color"; borderColor.className = backgroundColor.className = "tut-comic-input tut-comic-color";
    const fitMode = makeSelect(["裁切填充", "完整显示"]), emptyFill = makeSelect(["留空", "循环填充", "复制最后一张"]);
    stylePanel.append(
        makeFields([makeField("边框", borderWidth), makeField("边框色", borderColor), makeField("背景色", backgroundColor)], 3),
        makeField("图片适配", fitMode),
        makeField("空格填充", emptyFill),
    );
    const layerPanel = document.createElement("div"); layerPanel.className = "tut-comic-panel";
    layerPanel.innerHTML = '<h4>镜头图层</h4><div class="tut-comic-layer-help">列表顶部显示最前方画格。开放边缘的图片会按此顺序覆盖其他画格和画框线。</div>';
    const layerList = document.createElement("div"); layerList.className = "tut-comic-layer-list";
    const layerActions = document.createElement("div"); layerActions.className = "tut-comic-layer-actions";
    const layerTop = makeButton("置于顶层"), layerUp = makeButton("上移一层"), layerDown = makeButton("下移一层"), layerBottom = makeButton("置于底层"), layerReset = makeButton("恢复默认顺序");
    layerActions.append(layerTop, layerUp, layerDown, layerBottom); layerPanel.append(layerList, layerActions, layerReset);
    const sidebarPanels = { canvas: sizePanel, camera: settingsPanel, style: stylePanel, layers: layerPanel };
    const sidebarButtons = {};
    for (const [key, label] of [["canvas", "画布"], ["camera", "画框"], ["style", "样式"], ["layers", "图层"]]) {
        const button = makeButton(label); button.className = "tut-comic-tab"; button.dataset.panel = key; sidebarButtons[key] = button; sidebarTabs.append(button);
    }
    panels.append(sizePanel, settingsPanel, stylePanel, layerPanel); sidebar.append(sidebarTabs, panels); workspace.append(monitor, sidebar); root.append(workspace);

    function showSidebar(key, persist = true) {
        activeSidebar = sidebarPanels[key] ? key : "camera";
        for (const [name, panel] of Object.entries(sidebarPanels)) panel.classList.toggle("active", name === activeSidebar);
        for (const [name, button] of Object.entries(sidebarButtons)) button.classList.toggle("active", name === activeSidebar);
        if (persist) { node.properties = node.properties || {}; node.properties.tutComicCanvasState = { ...(node.properties.tutComicCanvasState || {}), sidebar_tab: activeSidebar }; markDirty(); }
    }
    showSidebar(activeSidebar, false);

    const bgCtx = bg.getContext("2d"), ctx = overlay.getContext("2d");

    function layoutName() {
        if (layout.value !== AUTO_LAYOUT) return layout.value;
        const count = Math.max(1, Math.min(MAX_PANELS, previewImages.length || MAX_PANELS));
        return { 1: "整页单格", 2: "左右双格", 3: "上大下二", 4: "四宫格", 5: "五格错落", 6: AUTO_PREVIEW_LAYOUT }[count];
    }
    function canvasSize() {
        return [Math.max(64, Number(width.value) || 1024), Math.max(64, Number(height.value) || 1536)];
    }
    function boundsFor(cw, ch, marginPixels) {
        const mx = Math.min(.49, Math.max(0, Number(marginPixels) || 0) / Math.max(1, cw));
        const my = Math.min(.49, Math.max(0, Number(marginPixels) || 0) / Math.max(1, ch));
        return [mx, my, 1 - mx, 1 - my];
    }
    function pageBounds(marginPixels = margin.value) {
        const [cw, ch] = canvasSize(); return boundsFor(cw, ch, marginPixels);
    }
    function pointInPage(pointValue) {
        const [x0, y0, x1, y1] = pageBounds();
        return { x: clamp(pointValue.x, x0, x1), y: clamp(pointValue.y, y0, y1) };
    }
    function remapOverrides(oldBounds, nextBounds) {
        const [ox0, oy0, ox1, oy1] = oldBounds, [nx0, ny0, nx1, ny1] = nextBounds;
        const mapX = (value) => clamp(nx0 + ((value - ox0) / Math.max(.001, ox1 - ox0)) * (nx1 - nx0), nx0, nx1);
        const mapY = (value) => clamp(ny0 + ((value - oy0) / Math.max(.001, oy1 - oy0)) * (ny1 - ny0), ny0, ny1);
        for (const rectangles of Object.values(data.layout_overrides)) {
            rectangles.forEach((rect, index) => {
                const next = [mapX(rect[0]), mapY(rect[1]), mapX(rect[2]), mapY(rect[3])];
                if (next[2] - next[0] >= .005 && next[3] - next[1] >= .005) rectangles[index] = next;
            });
        }
    }
    function materializedTemplate(name = layoutName()) {
        if (name === CUSTOM_LAYOUT) return clone(PANEL_LAYOUTS[CUSTOM_LAYOUT]);
        const [cw, ch] = canvasSize(), m = Math.max(0, Number(margin.value) || 0), gap = Math.max(0, Number(gutter.value) || 0);
        const contentW = Math.max(1, cw - 2 * m), contentH = Math.max(1, ch - 2 * m);
        return PANEL_LAYOUTS[name].map(([x0, y0, x1, y1]) => [
            clamp((m + x0 * contentW + (x0 > 0 ? gap / 2 : 0)) / cw),
            clamp((m + y0 * contentH + (y0 > 0 ? gap / 2 : 0)) / ch),
            clamp((m + x1 * contentW - (x1 < 1 ? gap / 2 : 0)) / cw),
            clamp((m + y1 * contentH - (y1 < 1 ? gap / 2 : 0)) / ch),
        ]);
    }
    function rectangles() { return data.layout_overrides[layoutName()] || materializedTemplate(); }
    function editableRectangles() {
        const name = layoutName();
        if (!data.layout_overrides[name]) data.layout_overrides[name] = materializedTemplate(name);
        return data.layout_overrides[name];
    }
    function layerOrder(name = layoutName()) {
        const count = (data.layout_overrides[name] || materializedTemplate(name)).length;
        const current = data.layer_orders[name];
        if (Array.isArray(current) && current.length === count && [...current].sort((a, b) => a - b).every((value, index) => value === index)) return current;
        delete data.layer_orders[name];
        return Array.from({ length: count }, (_, index) => index);
    }
    function moveSelectedLayer(destination) {
        const name = layoutName(), order = [...layerOrder(name)], position = order.indexOf(selected);
        if (position < 0) return;
        let nextPosition = position;
        if (destination === "top") nextPosition = order.length - 1;
        else if (destination === "up") nextPosition = Math.min(order.length - 1, position + 1);
        else if (destination === "down") nextPosition = Math.max(0, position - 1);
        else if (destination === "bottom") nextPosition = 0;
        if (nextPosition === position) return;
        order.splice(position, 1); order.splice(nextPosition, 0, selected); data.layer_orders[name] = order; writeData();
    }
    function alignmentTargets(index) {
        const [minX, minY, maxX, maxY] = pageBounds(), spanX = maxX - minX, spanY = maxY - minY;
        const xs = [minX, minX + spanX / 3, minX + spanX / 2, minX + spanX * 2 / 3, maxX];
        const ys = [minY, minY + spanY / 3, minY + spanY / 2, minY + spanY * 2 / 3, maxY];
        rectangles().forEach((rect, otherIndex) => {
            if (otherIndex === index) return;
            xs.push(rect[0], (rect[0] + rect[2]) / 2, rect[2]);
            ys.push(rect[1], (rect[1] + rect[3]) / 2, rect[3]);
        });
        return { xs: [...new Set(xs)], ys: [...new Set(ys)] };
    }
    function nearestSnap(anchors, targets, threshold) {
        let best = null;
        for (const anchor of anchors) for (const target of targets) {
            const delta = target - anchor, distance = Math.abs(delta);
            if (distance <= threshold && (!best || distance < best.distance)) best = { delta, target, distance };
        }
        return best;
    }
    function snapMove(source, dx, dy, index) {
        snapGuides = [];
        if (!snapEnabled) return { dx, dy };
        const bounds = overlay.getBoundingClientRect(), targets = alignmentTargets(index);
        const xSnap = nearestSnap([source[0] + dx, (source[0] + source[2]) / 2 + dx, source[2] + dx], targets.xs, 10 / Math.max(1, bounds.width));
        const ySnap = nearestSnap([source[1] + dy, (source[1] + source[3]) / 2 + dy, source[3] + dy], targets.ys, 10 / Math.max(1, bounds.height));
        if (xSnap) { dx += xSnap.delta; snapGuides.push({ axis: "x", value: xSnap.target }); }
        if (ySnap) { dy += ySnap.delta; snapGuides.push({ axis: "y", value: ySnap.target }); }
        return { dx, dy };
    }
    function snapCoordinate(value, axis, index) {
        if (!snapEnabled) return value;
        const bounds = overlay.getBoundingClientRect(), targets = alignmentTargets(index)[axis === "x" ? "xs" : "ys"];
        const snapped = nearestSnap([value], targets, 10 / Math.max(1, axis === "x" ? bounds.width : bounds.height));
        if (!snapped) return value;
        snapGuides.push({ axis, value: snapped.target }); return snapped.target;
    }
    function markDirty() { node.setDirtyCanvas?.(true, true); app.graph?.setDirtyCanvas?.(true, true); }
    function setWidget(name, value, notify = true) {
        const widget = widgets[name]; if (!widget || widget.value === value) return;
        widget.value = value;
        if (notify) widget.callback?.call(widget, value);
        markDirty();
    }
    function writeData() {
        widgets.panel_data.value = JSON.stringify(data);
        priorStoreCallback?.call(widgets.panel_data, widgets.panel_data.value);
        node.properties = node.properties || {};
        node.properties.tutComicCanvasState = { version: 2, panel_data: widgets.panel_data.value, snap_enabled: snapEnabled, sidebar_tab: activeSidebar, saved_at: Date.now() };
        markDirty(); draw();
    }
    function resizeCanvas() {
        const [cw, ch] = canvasSize(), backingScale = Math.min(1600 / cw, 1600 / ch, 1);
        const pw = Math.max(64, Math.round(cw * backingScale)), ph = Math.max(64, Math.round(ch * backingScale));
        if (bg.width !== pw || bg.height !== ph) { bg.width = overlay.width = pw; bg.height = overlay.height = ph; }
        const maxW = Math.max(100, monitor.clientWidth - 24), maxH = Math.max(100, monitor.clientHeight - 24);
        const fit = Math.min(maxW / pw, maxH / ph, 1);
        const displayW = Math.max(60, Math.round(pw * fit)), displayH = Math.max(60, Math.round(ph * fit));
        screen.style.width = `${displayW}px`; screen.style.height = `${displayH}px`;
        bg.style.width = overlay.style.width = `${displayW}px`; bg.style.height = overlay.style.height = `${displayH}px`;
    }
    function drawGrid() {
        if (!gridCheck.checked) return;
        bgCtx.save(); bgCtx.strokeStyle = "rgba(20,25,30,.18)"; bgCtx.lineWidth = 1;
        for (const fraction of [1 / 3, 2 / 3]) {
            bgCtx.beginPath(); bgCtx.moveTo(bg.width * fraction, 0); bgCtx.lineTo(bg.width * fraction, bg.height); bgCtx.stroke();
            bgCtx.beginPath(); bgCtx.moveTo(0, bg.height * fraction); bgCtx.lineTo(bg.width, bg.height * fraction); bgCtx.stroke();
        }
        bgCtx.restore();
    }
    function previewFor(index) {
        if (index < previewImages.length) return previewImages[index];
        if (!previewImages.length || emptyFill.value === "留空") return null;
        return emptyFill.value === "循环填充" ? previewImages[index % previewImages.length] : previewImages[previewImages.length - 1];
    }
    function drawPreview(image, x, y, w, h, panel) {
        if (!image?.complete || !image.naturalWidth) return false;
        const contain = fitMode.value === "完整显示";
        const scale = (contain ? Math.min(w / image.naturalWidth, h / image.naturalHeight) : Math.max(w / image.naturalWidth, h / image.naturalHeight)) * panel.zoom;
        const drawW = Math.max(1, image.naturalWidth * scale), drawH = Math.max(1, image.naturalHeight * scale);
        const drawX = x + (w - drawW) * panel.focus_x, drawY = y + (h - drawH) * panel.focus_y;
        const clipX = panel.overflow_left ? 0 : x, clipY = panel.overflow_top ? 0 : y;
        const clipRight = panel.overflow_right ? overlay.width : x + w, clipBottom = panel.overflow_bottom ? overlay.height : y + h;
        ctx.save(); ctx.beginPath(); ctx.rect(clipX, clipY, Math.max(0, clipRight - clipX), Math.max(0, clipBottom - clipY)); ctx.clip();
        if (panel.flip) { ctx.translate(x * 2 + w, 0); ctx.scale(-1, 1); ctx.drawImage(image, drawX, drawY, drawW, drawH); }
        else ctx.drawImage(image, drawX, drawY, drawW, drawH);
        ctx.restore(); return true;
    }
    function setPreviews(items) {
        previewImages = (items || []).slice(0, MAX_PANELS).map((item) => {
            const image = new Image(), url = previewUrl(item);
            image.onload = image.onerror = () => draw();
            if (url) image.src = url;
            return image;
        });
        selected = 0; draw();
    }
    node.__tutComicSetPreviews = setPreviews;
    function renderLayerList(rects) {
        const order = [...layerOrder()].reverse();
        layerList.replaceChildren();
        order.forEach((panelIndex, rank) => {
            const row = document.createElement("div"); row.className = `tut-comic-layer-row${panelIndex === selected ? " active" : ""}`; row.dataset.panelIndex = String(panelIndex);
            const rankLabel = document.createElement("span"); rankLabel.className = "tut-comic-layer-rank"; rankLabel.textContent = String(rank + 1);
            const name = document.createElement("span"); name.textContent = `画格 ${panelIndex + 1}`;
            const position = document.createElement("span"); position.className = "tut-comic-label"; position.textContent = rank === 0 ? "最前" : rank === order.length - 1 ? "最底" : "";
            row.append(rankLabel, name, position); row.addEventListener("click", () => { selected = panelIndex; draw(); }); layerList.append(row);
        });
        const position = layerOrder().indexOf(selected), last = rects.length - 1;
        layerTop.disabled = layerUp.disabled = position >= last;
        layerBottom.disabled = layerDown.disabled = position <= 0;
    }
    function draw() {
        resizeCanvas();
        bgCtx.clearRect(0, 0, bg.width, bg.height); bgCtx.fillStyle = backgroundColor.value || "#ffffff"; bgCtx.fillRect(0, 0, bg.width, bg.height); drawGrid();
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        const rects = rectangles(); selected = Math.max(0, Math.min(selected, rects.length - 1));
        const displayWidth = Math.max(1, overlay.getBoundingClientRect().width);
        const uiScale = overlay.width / displayWidth;
        const [previewCanvasWidth] = canvasSize();
        const frameStroke = Math.max(0, Number(borderWidth.value) || 0) * overlay.width / previewCanvasWidth;
        const geometry = rects.map((rect) => {
            const [x0, y0, x1, y1] = rect, x = x0 * overlay.width, y = y0 * overlay.height, w = (x1 - x0) * overlay.width, h = (y1 - y0) * overlay.height;
            const stroke = Math.min(frameStroke, Math.max(0, Math.min(w, h) / 2 - 1));
            return { x, y, w, h, stroke, innerX: x + stroke, innerY: y + stroke, innerW: Math.max(1, w - stroke * 2), innerH: Math.max(1, h - stroke * 2) };
        });
        geometry.forEach(({ x, y, w, h, stroke }) => {
            if (stroke > 0) { ctx.fillStyle = borderColor.value || "#111111"; ctx.fillRect(x, y, w, h); }
        });
        for (const index of layerOrder()) {
            const { innerX, innerY, innerW, innerH } = geometry[index];
            const panel = data.panels[index], hasPreview = drawPreview(previewFor(index), innerX, innerY, innerW, innerH, panel);
            ctx.save(); ctx.fillStyle = hasPreview ? (index === selected ? "rgba(26,118,150,.10)" : "rgba(0,0,0,.04)") : (index === selected ? "rgba(26,118,150,.52)" : "rgba(33,39,47,.82)"); ctx.fillRect(innerX, innerY, innerW, innerH);
            ctx.restore();
        }
        geometry.forEach(({ x, y, w, h }, index) => {
            if (index === selected) { ctx.strokeStyle = "#35d0c8"; ctx.lineWidth = 2 * uiScale; ctx.strokeRect(x, y, w, h); }
            const badgeX = x + 4 * uiScale, badgeY = y + 4 * uiScale, badgeW = 24 * uiScale, badgeH = 18 * uiScale;
            ctx.fillStyle = index === selected ? "#00aeb8" : "#137b80"; ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
            ctx.strokeStyle = "rgba(0,0,0,.8)"; ctx.lineWidth = 2 * uiScale; ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);
            ctx.fillStyle = "#fff"; ctx.font = `bold ${12 * uiScale}px sans-serif`; ctx.textBaseline = "middle"; ctx.textAlign = "center"; ctx.fillText(String(index + 1), badgeX + badgeW / 2, badgeY + badgeH / 2);
            if (index === selected && mode === "frame") {
                ctx.fillStyle = "#fff"; const handle = 5 * uiScale;
                for (const [hx, hy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) { ctx.fillRect(hx - handle, hy - handle, handle * 2, handle * 2); ctx.strokeStyle = "#167d86"; ctx.lineWidth = 3 * uiScale; ctx.strokeRect(hx - handle, hy - handle, handle * 2, handle * 2); }
            }
        });
        if (snapGuides.length) {
            ctx.save(); ctx.strokeStyle = "#ff4fc8"; ctx.lineWidth = 2 * uiScale; ctx.setLineDash([6 * uiScale, 5 * uiScale]);
            for (const guide of snapGuides) {
                ctx.beginPath();
                if (guide.axis === "x") { const gx = guide.value * overlay.width; ctx.moveTo(gx, 0); ctx.lineTo(gx, overlay.height); }
                else { const gy = guide.value * overlay.height; ctx.moveTo(0, gy); ctx.lineTo(overlay.width, gy); }
                ctx.stroke();
            }
            ctx.restore();
        }
        if (drawing) {
            const x = Math.min(drawing.x0, drawing.x1) * overlay.width, y = Math.min(drawing.y0, drawing.y1) * overlay.height;
            const w = Math.abs(drawing.x1 - drawing.x0) * overlay.width, h = Math.abs(drawing.y1 - drawing.y0) * overlay.height;
            ctx.fillStyle = "rgba(53,208,200,.18)"; ctx.fillRect(x, y, w, h); ctx.strokeStyle = "#35d0c8"; ctx.lineWidth = 3; ctx.strokeRect(x, y, w, h);
        }
        const rect = rects[selected], panel = data.panels[selected];
        selectedLabel.textContent = `画格 ${selected + 1} / ${rects.length}`;
        zoom.value = String(panel.zoom); zoomValue.textContent = `${panel.zoom.toFixed(2)}×`; flip.classList.toggle("active", panel.flip);
        for (const [key, check] of Object.entries(edgeChecks)) check.checked = panel[key] === true;
        frameBtn.classList.toggle("active", mode === "frame" && !drawMode); cameraBtn.classList.toggle("active", mode === "camera"); drawBtn.classList.toggle("active", drawMode);
        overlay.style.cursor = drawMode ? "crosshair" : mode === "camera" ? (moving?.camera ? "grabbing" : "grab") : "move";
        const free = layout.value === CUSTOM_LAYOUT; drawBtn.disabled = !free || rects.length >= MAX_PANELS; deleteBtn.disabled = !free || rects.length <= 1;
        renderLayerList(rects);
    }
    function point(event) {
        const bounds = overlay.getBoundingClientRect();
        return { x: clamp((event.clientX - bounds.left) / Math.max(1, bounds.width)), y: clamp((event.clientY - bounds.top) / Math.max(1, bounds.height)) };
    }
    function hitTest(p) {
        const rects = rectangles(), bx = 16 / Math.max(1, overlay.getBoundingClientRect().width), by = 16 / Math.max(1, overlay.getBoundingClientRect().height);
        for (const index of [...layerOrder()].reverse()) {
            const [x0, y0, x1, y1] = rects[index];
            for (const [handle, hx, hy] of [["nw", x0, y0], ["ne", x1, y0], ["sw", x0, y1], ["se", x1, y1]]) {
                if (Math.abs(p.x - hx) <= bx && Math.abs(p.y - hy) <= by) return { index, handle };
            }
            if (p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1) return { index, handle: "move" };
        }
        return null;
    }
    overlay.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return; event.preventDefault(); event.stopPropagation(); overlay.setPointerCapture?.(event.pointerId);
        const p = point(event); snapGuides = [];
        if (drawMode && layout.value === CUSTOM_LAYOUT) { const bounded = pointInPage(p); drawing = { x0: bounded.x, y0: bounded.y, x1: bounded.x, y1: bounded.y }; draw(); return; }
        const hit = hitTest(p); if (!hit) return; selected = hit.index;
        if (mode === "camera") {
            const panel = data.panels[selected];
            moving = { camera: true, start: p, focus_x: panel.focus_x, focus_y: panel.focus_y };
        } else if (hit.handle === "move") moving = { start: p, rect: clone(rectangles()[selected]) };
        else resizing = { handle: hit.handle, rect: clone(rectangles()[selected]) };
        editableRectangles(); draw();
    });
    overlay.addEventListener("pointermove", (event) => {
        const p = point(event);
        if (drawing) { snapGuides = []; const bounded = pointInPage(p); drawing.x1 = bounded.x; drawing.y1 = bounded.y; draw(); return; }
        if (moving?.camera) {
            snapGuides = [];
            const rect = rectangles()[selected], panel = data.panels[selected];
            panel.focus_x = clamp(moving.focus_x - (p.x - moving.start.x) / Math.max(.001, rect[2] - rect[0]));
            panel.focus_y = clamp(moving.focus_y - (p.y - moving.start.y) / Math.max(.001, rect[3] - rect[1]));
            draw(); return;
        }
        if (moving) {
            const source = moving.rect, [minX, minY, maxX, maxY] = pageBounds();
            let dx = Math.max(minX - source[0], Math.min(maxX - source[2], p.x - moving.start.x));
            let dy = Math.max(minY - source[1], Math.min(maxY - source[3], p.y - moving.start.y));
            ({ dx, dy } = snapMove(source, dx, dy, selected));
            dx = Math.max(minX - source[0], Math.min(maxX - source[2], dx));
            dy = Math.max(minY - source[1], Math.min(maxY - source[3], dy));
            editableRectangles()[selected] = [source[0] + dx, source[1] + dy, source[2] + dx, source[3] + dy]; draw(); return;
        }
        if (resizing) {
            const next = clone(resizing.rect), minW = .025, minH = .025, [minX, minY, maxX, maxY] = pageBounds();
            snapGuides = [];
            if (resizing.handle.includes("w")) next[0] = Math.max(minX, Math.min(next[2] - minW, snapCoordinate(p.x, "x", selected)));
            if (resizing.handle.includes("e")) next[2] = Math.min(maxX, Math.max(next[0] + minW, snapCoordinate(p.x, "x", selected)));
            if (resizing.handle.includes("n")) next[1] = Math.max(minY, Math.min(next[3] - minH, snapCoordinate(p.y, "y", selected)));
            if (resizing.handle.includes("s")) next[3] = Math.min(maxY, Math.max(next[1] + minH, snapCoordinate(p.y, "y", selected)));
            editableRectangles()[selected] = next.map((value) => clamp(value)); draw();
        }
    });
    const finishPointer = (event) => {
        if (drawing) {
            const rect = [Math.min(drawing.x0, drawing.x1), Math.min(drawing.y0, drawing.y1), Math.max(drawing.x0, drawing.x1), Math.max(drawing.y0, drawing.y1)];
            if (rect[2] - rect[0] >= .025 && rect[3] - rect[1] >= .025 && rectangles().length < MAX_PANELS) {
                const name = layoutName(), order = [...layerOrder(name)], editable = editableRectangles(); editable.push(rect); selected = editable.length - 1; order.push(selected); data.layer_orders[name] = order;
            }
            drawing = null; drawMode = false;
        }
        if (moving || resizing) { moving = null; resizing = null; }
        snapGuides = [];
        writeData(); overlay.releasePointerCapture?.(event.pointerId);
    };
    overlay.addEventListener("pointerup", finishPointer); overlay.addEventListener("pointercancel", finishPointer);
    overlay.addEventListener("wheel", (event) => {
        if (mode !== "camera") return;
        const hit = hitTest(point(event)); if (!hit) return;
        event.preventDefault(); event.stopPropagation(); selected = hit.index;
        const panel = data.panels[selected];
        const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1);
        const next = clamp(panel.zoom * Math.exp(-delta * .0015), .25, 4);
        panel.zoom = Math.round(next / .05) * .05;
        writeData();
    }, { passive: false });

    function commitCanvasSize() {
        const w = Math.max(64, Math.min(8192, Math.round(Number(width.value) || 1024))), h = Math.max(64, Math.min(8192, Math.round(Number(height.value) || 1536)));
        width.value = String(w); height.value = String(h);
        const preset = SIZE_PRESETS.find((item) => item[1] === w && item[2] === h);
        sizePreset.value = preset ? preset[0] : "自定义";
        setWidget("canvas_width", w); setWidget("canvas_height", h); draw();
    }
    for (const [key, button] of Object.entries(sidebarButtons)) button.addEventListener("click", () => showSidebar(key));
    layerTop.addEventListener("click", () => moveSelectedLayer("top")); layerUp.addEventListener("click", () => moveSelectedLayer("up"));
    layerDown.addEventListener("click", () => moveSelectedLayer("down")); layerBottom.addEventListener("click", () => moveSelectedLayer("bottom"));
    layerReset.addEventListener("click", () => { delete data.layer_orders[layoutName()]; writeData(); });
    layout.addEventListener("change", () => { setWidget("layout", layout.value); selected = 0; drawMode = false; if (layout.value === CUSTOM_LAYOUT) editableRectangles(); writeData(); });
    frameBtn.addEventListener("click", () => { mode = "frame"; drawMode = false; snapGuides = []; draw(); });
    cameraBtn.addEventListener("click", () => { mode = "camera"; drawMode = false; snapGuides = []; draw(); });
    drawBtn.addEventListener("click", () => { if (!drawBtn.disabled) { mode = "frame"; drawMode = !drawMode; draw(); } });
    deleteBtn.addEventListener("click", () => { if (!deleteBtn.disabled) { const name = layoutName(), removed = selected, order = layerOrder(name).filter((index) => index !== removed).map((index) => index > removed ? index - 1 : index); editableRectangles().splice(removed, 1); data.layer_orders[name] = order; selected = Math.max(0, removed - 1); writeData(); } });
    resetBtn.addEventListener("click", () => { if (layoutName() === CUSTOM_LAYOUT) { data.layout_overrides[CUSTOM_LAYOUT] = clone(PANEL_LAYOUTS[CUSTOM_LAYOUT]); delete data.layer_orders[CUSTOM_LAYOUT]; } else delete data.layout_overrides[layoutName()]; selected = 0; writeData(); });
    snapCheck.addEventListener("change", () => { snapEnabled = snapCheck.checked; snapGuides = []; writeData(); });
    gridCheck.addEventListener("change", draw);
    sizePreset.addEventListener("change", () => { const found = SIZE_PRESETS.find((item) => item[0] === sizePreset.value); if (found) { width.value = String(found[1]); height.value = String(found[2]); commitCanvasSize(); } });
    for (const input of [width, height]) { input.addEventListener("change", commitCanvasSize); input.addEventListener("keydown", (event) => { if (event.key === "Enter") { commitCanvasSize(); input.blur(); } }); }
    margin.addEventListener("change", () => {
        const oldMargin = Math.max(0, Number(widgets.page_margin.value) || 0);
        const nextMargin = Math.max(0, Math.round(Number(margin.value) || 0));
        margin.value = String(nextMargin);
        if (nextMargin !== oldMargin) remapOverrides(pageBounds(oldMargin), pageBounds(nextMargin));
        setWidget("page_margin", nextMargin); writeData();
    });
    gutter.addEventListener("change", () => { setWidget("gutter", Math.max(0, Math.round(Number(gutter.value) || 0))); draw(); });
    borderWidth.addEventListener("change", () => { setWidget("border_width", Math.max(0, Math.round(Number(borderWidth.value) || 0))); draw(); });
    borderColor.addEventListener("input", () => { setWidget("border_color", borderColor.value); draw(); }); backgroundColor.addEventListener("input", () => { setWidget("background_color", backgroundColor.value); draw(); });
    fitMode.addEventListener("change", () => { setWidget("fit_mode", fitMode.value); draw(); }); emptyFill.addEventListener("change", () => { setWidget("empty_fill", emptyFill.value); draw(); });
    zoom.addEventListener("input", () => { data.panels[selected].zoom = Number(zoom.value); writeData(); });
    flip.addEventListener("click", () => { data.panels[selected].flip = !data.panels[selected].flip; writeData(); });
    resetCamera.addEventListener("click", () => { Object.assign(data.panels[selected], { focus_x: .5, focus_y: .5, zoom: 1, flip: false }); writeData(); });
    for (const [key, check] of Object.entries(edgeChecks)) check.addEventListener("change", () => { data.panels[selected][key] = check.checked; writeData(); });

    layout.value = widgets.layout.value || AUTO_LAYOUT; width.value = String(widgets.canvas_width.value); height.value = String(widgets.canvas_height.value);
    margin.value = String(widgets.page_margin.value); gutter.value = String(widgets.gutter.value); borderWidth.value = String(widgets.border_width.value);
    borderColor.value = widgets.border_color.value || "#111111"; backgroundColor.value = widgets.background_color.value || "#ffffff";
    fitMode.value = widgets.fit_mode.value; emptyFill.value = widgets.empty_fill.value; commitCanvasSize();
    widgets.panel_data.callback = function (...args) { const result = priorStoreCallback?.apply(this, args); data = parseData(widgets.panel_data.value); selected = Math.min(selected, rectangles().length - 1); draw(); return result; };
    const domWidget = node.addDOMWidget("tut_comic_canvas_v2", "TUT_COMIC_CANVAS_V2", root, { serialize: false, hideOnZoom: false });
    domWidget.computeSize = (nodeWidth) => [nodeWidth, 900];
    node.__tutComicCanvasV2 = true;
    const ensureDefaultSize = () => node.setSize?.([Math.max(node.size?.[0] || 0, 1100), Math.max(node.size?.[1] || 0, 1040)]);
    ensureDefaultSize();
    requestAnimationFrame(() => { ensureDefaultSize(); draw(); setTimeout(() => { ensureDefaultSize(); draw(); }, 100); setTimeout(() => { ensureDefaultSize(); draw(); }, 350); });
}

app.registerExtension({
    name: "TUT_Nodes.ComicCanvasV2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TUT_ComicPanelCanvas" || nodeType.prototype.__tutComicPreviewHook) return;
        const previousExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = previousExecuted?.apply(this, arguments);
            this.__tutComicSetPreviews?.(output?.input_previews || []);
            return result;
        };
        nodeType.prototype.__tutComicPreviewHook = true;
    },
    nodeCreated(node) { installComicCanvas(node); },
});
