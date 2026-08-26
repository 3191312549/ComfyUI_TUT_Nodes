import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// Original inputs remain serialized stores. Visible controls use LiteGraph's
// native widgets so the layout matches the Node 1.0 renderer.
const NODE_ID = "TUT_LoadExcel";
const STATE_KEY = "tut_excel_state";
const FORMULA_MODES = ["缓存值", "公式文本"];
const RANGE_MODES = ["自动有效区", "自定义"];

function hideStore(widget) {
    widget.type = "converted-widget";
    widget.hidden = true;
    widget.options ||= {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.serializeValue = () => widget.value;
}

function displayOnly(widget) {
    widget.serialize = false;
    widget.options ||= {};
    widget.options.serialize = false;
    widget.serializeValue = () => undefined;
    return widget;
}

function setStore(node, widget, value) {
    if (widget.value === value) return;
    const previous = widget.value;
    app.graph?.beforeChange?.();
    try {
        widget.value = value;
        node.properties ||= {};
        node.properties[STATE_KEY] ||= {};
        node.properties[STATE_KEY][widget.name] = value;
        widget.callback?.(value, app.canvas, node);
        node.onWidgetChanged?.(widget.name, value, previous, widget);
        node.setDirtyCanvas?.(true, true);
    } finally {
        app.graph?.afterChange?.();
    }
}

function fileName(path) {
    return String(path || "").replaceAll("\\", "/").split("/").at(-1) || "未选择 Excel";
}

function parseRange(value) {
    const match = String(value || "").replaceAll("$", "")
        .match(/^([A-Za-z]+[1-9][0-9]*)(?::([A-Za-z]+[1-9][0-9]*))?$/);
    if (!match) return { start: "A1", end: "" };
    return { start: match[1].toUpperCase(), end: (match[2] || "").toUpperCase() };
}

async function jsonResponse(response) {
    let result = {};
    try {
        result = await response.json();
    } catch {
        // The HTTP status below is clearer when a response is not JSON.
    }
    if (!response.ok) throw new Error(result?.error || `${response.status} ${response.statusText}`);
    return result;
}

function inspectWorkbook(path) {
    return api.fetchApi("/tut_nodes/excel/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ excel_path: path }),
    }).then(jsonResponse);
}

function listWorkbooks() {
    return api.fetchApi("/tut_nodes/excel/files").then(jsonResponse)
        .then((result) => Array.isArray(result?.files) ? result.files : []);
}

function installExcelControls(node) {
    if (node.comfyClass !== NODE_ID || node.__tutExcelControls) return;
    if (typeof node.addWidget !== "function") return;

    const pathStore = node.widgets?.find((w) => w.name === "excel_path");
    const sheetStore = node.widgets?.find((w) => w.name === "sheet_name");
    const rangeStore = node.widgets?.find((w) => w.name === "cell_range");
    const formulaStore = node.widgets?.find((w) => w.name === "formula_mode");
    if (!pathStore || !sheetStore || !rangeStore || !formulaStore) return;
    const stores = [pathStore, sheetStore, rangeStore, formulaStore];
    stores.forEach(hideStore);

    const fileValues = [""];
    const fileLabels = new Map([["", "未选择 Excel"]]);
    const sheetValues = [""];
    const sheetLabels = new Map([["", "未选择工作表"]]);
    let sheets = [];
    let generation = 0;

    const fileWidget = displayOnly(node.addWidget("combo", "Excel", pathStore.value || "",
        (value) => selectFile(value), {
            values: fileValues,
            getOptionLabel: (value) => fileLabels.get(value) || fileName(value),
        }));
    const chooseWidget = displayOnly(node.addWidget("button", "选择本机 Excel", null,
        () => pickLocalExcel(), { serialize: false }));
    const sheetWidget = displayOnly(node.addWidget("combo", "工作表", sheetStore.value || "",
        (value) => selectSheet(value), {
            values: sheetValues,
            getOptionLabel: (value) => sheetLabels.get(value) || value || "未选择工作表",
        }));
    const refreshWidget = displayOnly(node.addWidget("button", "刷新工作表", null,
        () => refreshWorkbook(), { serialize: false }));
    const rangeModeWidget = displayOnly(node.addWidget("combo", "读取范围",
        rangeStore.value ? "自定义" : "自动有效区", (value) => changeRangeMode(value),
        { values: RANGE_MODES }));
    const parsed = parseRange(rangeStore.value);
    const startWidget = displayOnly(node.addWidget("text", "起始单元格", parsed.start,
        () => saveCustomRange(), { serialize: false }));
    const endWidget = displayOnly(node.addWidget("text", "结束单元格", parsed.end,
        () => saveCustomRange(), { serialize: false }));
    const formulaWidget = displayOnly(node.addWidget("combo", "公式模式",
        FORMULA_MODES.includes(formulaStore.value) ? formulaStore.value : "缓存值",
        (value) => setStore(node, formulaStore, value), { values: FORMULA_MODES }));
    const statusWidget = displayOnly(node.addWidget("text", "状态", "请选择 Excel 文件",
        null, { serialize: false }));
    statusWidget.disabled = true;

    function setStatus(text, isError = false) {
        statusWidget.value = String(text || "");
        statusWidget.options.color = isError ? "#ff8a80" : undefined;
        node.setDirtyCanvas?.(true, true);
    }

    function currentSheet() {
        return sheets.find((sheet) => sheet.name === sheetWidget.value);
    }

    function updateRangeUi() {
        const custom = rangeModeWidget.value === "自定义";
        startWidget.disabled = !custom;
        endWidget.disabled = !custom;
        if (!custom) {
            setStore(node, rangeStore, "");
            const sheet = currentSheet();
            setStatus(sheet ? `自动读取 ${sheet.name} 的有效区域：${sheet.range}` : "自动读取工作表有效区域");
        }
        node.setDirtyCanvas?.(true, true);
    }

    function saveCustomRange() {
        if (rangeModeWidget.value !== "自定义") return;
        const start = String(startWidget.value || "").trim().toUpperCase();
        const end = String(endWidget.value || "").trim().toUpperCase();
        startWidget.value = start;
        endWidget.value = end;
        const value = end ? `${start}:${end}` : start;
        setStore(node, rangeStore, value);
        setStatus(value ? `自定义范围：${value}` : "自定义范围：未填写");
    }

    function addSavedFile() {
        const current = String(pathStore.value || "");
        if (current && !fileValues.includes(current)) fileValues.push(current);
        if (current) fileLabels.set(current, fileName(current));
        fileWidget.value = current;
    }

    async function populateFiles() {
        let files = [];
        try {
            files = await listWorkbooks();
        } catch {
            // A saved absolute path remains usable if the listing route fails.
        }
        fileValues.splice(0, fileValues.length, "");
        fileLabels.clear();
        fileLabels.set("", "未选择 Excel");
        for (const file of files) {
            if (!file?.token) continue;
            fileValues.push(file.token);
            fileLabels.set(file.token, file.name || fileName(file.token));
        }
        addSavedFile();
        node.setDirtyCanvas?.(true, true);
    }

    function populateSheets(result) {
        sheets = Array.isArray(result?.sheets) ? result.sheets : [];
        if (!sheets.length) throw new Error("文件中没有可读取的工作表");
        sheetValues.splice(0, sheetValues.length);
        sheetLabels.clear();
        for (const sheet of sheets) {
            sheetValues.push(sheet.name);
            sheetLabels.set(sheet.name, `${sheet.name}  (${sheet.range})`);
        }
        const saved = String(sheetStore.value || "");
        const selected = sheetValues.includes(saved) ? saved : sheetValues[0];
        sheetWidget.value = selected;
        if (saved !== selected) setStore(node, sheetStore, selected);
        updateRangeUi();
    }

    async function refreshWorkbook() {
        const path = String(pathStore.value || "");
        if (!path) return setStatus("请先选择 Excel 文件", true);
        const currentGeneration = ++generation;
        refreshWidget.disabled = true;
        setStatus("正在读取工作表…");
        try {
            const result = await inspectWorkbook(path);
            if (currentGeneration === generation) populateSheets(result);
        } catch (error) {
            if (currentGeneration === generation) {
                setStatus(`读取失败：${error instanceof Error ? error.message : String(error)}`, true);
            }
        } finally {
            if (currentGeneration === generation) refreshWidget.disabled = false;
        }
    }

    async function pickLocalExcel() {
        chooseWidget.disabled = true;
        setStatus("正在打开本机文件选择器…");
        try {
            const result = await api.fetchApi("/tut_nodes/excel/pick-local", { method: "POST" })
                .then(jsonResponse);
            if (result?.cancelled) return updateRangeUi();
            if (!result?.path) throw new Error("文件选择器没有返回有效路径");
            setStore(node, pathStore, result.path);
            resetSelection();
            await populateFiles();
            await refreshWorkbook();
        } catch (error) {
            setStatus(`选择失败：${error instanceof Error ? error.message : String(error)}`, true);
        } finally {
            chooseWidget.disabled = false;
        }
    }

    function resetSelection() {
        setStore(node, sheetStore, "");
        setStore(node, rangeStore, "");
        rangeModeWidget.value = "自动有效区";
        startWidget.value = "A1";
        endWidget.value = "";
    }

    async function selectFile(value) {
        setStore(node, pathStore, String(value || ""));
        resetSelection();
        if (value) return refreshWorkbook();
        sheets = [];
        sheetValues.splice(0, sheetValues.length, "");
        sheetLabels.clear();
        sheetLabels.set("", "未选择工作表");
        sheetWidget.value = "";
        setStatus("请选择 Excel 文件");
    }

    function selectSheet(value) {
        setStore(node, sheetStore, String(value || ""));
        updateRangeUi();
    }

    function changeRangeMode(value) {
        if (value === "自定义") {
            const range = parseRange(currentSheet()?.range || "A1");
            if (!startWidget.value) startWidget.value = range.start;
            if (!endWidget.value) endWidget.value = range.end || range.start;
            saveCustomRange();
        }
        updateRangeUi();
    }

    function restore() {
        const saved = node.properties?.[STATE_KEY];
        if (saved && typeof saved === "object") {
            for (const store of stores) {
                if (typeof saved[store.name] === "string") store.value = saved[store.name];
            }
        } else {
            node.properties ||= {};
            node.properties[STATE_KEY] = Object.fromEntries(stores.map((w) => [w.name, w.value]));
        }
        addSavedFile();
        sheetWidget.value = String(sheetStore.value || "");
        const range = parseRange(rangeStore.value);
        startWidget.value = range.start;
        endWidget.value = range.end;
        rangeModeWidget.value = rangeStore.value ? "自定义" : "自动有效区";
        formulaWidget.value = FORMULA_MODES.includes(formulaStore.value) ? formulaStore.value : "缓存值";
        updateRangeUi();
    }

    const previousConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = previousConfigure?.apply(this, args);
        queueMicrotask(() => {
            restore();
            populateFiles().then(() => pathStore.value && refreshWorkbook());
        });
        return result;
    };

    node.__tutExcelControls = true;
    const size = node.computeSize?.() || node.size || [340, 0];
    node.setSize?.([Math.max(size[0] || 0, 340), size[1] || node.size?.[1] || 0]);
    restore();
    populateFiles().then(() => pathStore.value && refreshWorkbook());
}

app.registerExtension({ name: "TUT_Nodes.ExcelControls", nodeCreated: installExcelControls });
