import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "TUT_AnimaArtistPromptMixer";
const MIN_WEIGHT = 0.1;
const MAX_WEIGHT = 3.0;
const WEIGHT_STEP = 0.1;
const COMPACT_EDITOR_HEIGHT = 124;
const COMPACT_NODE_HEIGHT = 220;
const NODE_CHROME_HEIGHT = COMPACT_NODE_HEIGHT - COMPACT_EDITOR_HEIGHT;
const EDITOR_FIXED_HEIGHT = COMPACT_EDITOR_HEIGHT - 32;
const AUTO_MAX_CHIPS_HEIGHT = 190;
const STATE_KEY = "tut_anima_artist_data";

function addStyles() {
    if (document.getElementById("tut-anima-artist-styles")) return;
    const style = document.createElement("style");
    style.id = "tut-anima-artist-styles";
    style.textContent = `
      .tut-anima-artists{box-sizing:border-box;display:flex;flex-direction:column;width:100%;height:100%;min-height:0;padding:8px;color:#ddd;font:12px sans-serif;user-select:none}
      .tut-anima-search-wrap{position:relative;flex:none}.tut-anima-search{box-sizing:border-box;width:100%;padding:7px 9px;border:1px solid #555;border-radius:7px;background:#17191c;color:#eee;outline:none}
      .tut-anima-search:focus{border-color:#7f9cff}.tut-anima-results{position:absolute;z-index:20;left:0;right:0;top:100%;max-height:220px;overflow:auto;margin-top:3px;border:1px solid #4b4f57;border-radius:7px;background:#1d2025;box-shadow:0 8px 22px #0009}
      .tut-anima-result{display:flex;justify-content:space-between;gap:10px;width:100%;padding:7px 9px;border:0;border-bottom:1px solid #30343a;background:transparent;color:#eee;text-align:left;cursor:pointer}
      .tut-anima-result:hover,.tut-anima-result:focus{background:#303746}.tut-anima-count{color:#9da7b8;white-space:nowrap}
      .tut-anima-status{flex:none;min-height:16px;padding:4px 2px;color:#9299a5}.tut-anima-status.error{color:#ff9292}
      .tut-anima-chips{display:flex;flex:1 1 auto;flex-wrap:wrap;align-content:flex-start;gap:7px;min-height:32px;overflow:auto;padding:4px 1px;touch-action:none}
      .tut-anima-chip{display:inline-flex;align-items:center;gap:6px;max-width:100%;padding:5px 7px 5px 9px;border:1px solid #65739a;border-radius:999px;background:#293044;color:#f2f4ff;cursor:ew-resize;touch-action:none}
      .tut-anima-chip.dragging{border-color:#9bb2ff;background:#354267}.tut-anima-chip-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tut-anima-edit{width:130px;min-width:70px;padding:1px 4px;border:1px solid #9bb2ff;border-radius:4px;background:#17191c;color:#fff;outline:none}.tut-anima-weight{min-width:25px;color:#b9c8ff;font-variant-numeric:tabular-nums}
      .tut-anima-remove{display:grid;place-items:center;width:17px;height:17px;padding:0;border:0;border-radius:50%;background:#ffffff18;color:#ddd;cursor:pointer}.tut-anima-remove:hover{background:#ff6c6c;color:#fff}
      .tut-anima-empty{padding:7px 2px;color:#777}.tut-anima-hint{flex:none;padding-top:5px;color:#777}
    `;
    document.head.append(style);
}

function hideStore(widget) {
    widget.serialize = true;
    widget.options ||= {};
    widget.options.serialize = true;
    widget.options.hidden = true;
    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.serializeValue = () => widget.value;
}

function cleanName(value) {
    let name = String(value || "").trim().replace(/^@/, "").trim();
    return name.replace(/\s+/g, " ");
}

function keyFor(name) {
    return cleanName(name).toLocaleLowerCase();
}

function clampWeight(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 1;
    return Math.round(Math.min(MAX_WEIGHT, Math.max(MIN_WEIGHT, number)) / WEIGHT_STEP) * WEIGHT_STEP;
}

function parseArtistToken(value) {
    const text = String(value || "").trim();
    const weighted = text.match(/^\(\s*(@.+):([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)$/);
    const parsedWeight = Number(weighted?.[2]);
    if (weighted && Number.isFinite(parsedWeight) && parsedWeight >= MIN_WEIGHT && parsedWeight <= MAX_WEIGHT) {
        return { name: cleanName(weighted[1]), weight: clampWeight(parsedWeight) };
    }
    if (/^\(\s*@.+\)$/.test(text)) return { name: cleanName(text.slice(1, -1)), weight: 1 };
    return { name: cleanName(text), weight: 1 };
}

function defaultData() {
    return { version: 1, artists: [] };
}

function isSerializedArtistData(value) {
    if (typeof value !== "string") return false;
    try {
        const parsed = JSON.parse(value);
        return parsed?.version === 1 && Array.isArray(parsed.artists);
    } catch {
        return false;
    }
}

function parseData(value) {
    try {
        const parsed = JSON.parse(value);
        if (parsed?.version !== 1 || !Array.isArray(parsed.artists)) return defaultData();
        const deduped = new Map();
        for (const item of parsed.artists) {
            const name = cleanName(item?.name);
            if (!name) continue;
            deduped.set(keyFor(name), { name, weight: clampWeight(item?.weight) });
        }
        return { version: 1, artists: [...deduped.values()] };
    } catch {
        return defaultData();
    }
}

function installEditor(node, store) {
    if (node.__tutAnimaArtistEditor || typeof node.addDOMWidget !== "function") return;
    addStyles();
    hideStore(store);
    let data = parseData(store.value);
    let requestNumber = 0;
    let debounceTimer = 0;
    let activeController = null;
    let finishActiveDrag = null;
    let resizeFrame = 0;
    let composing = false;

    const root = document.createElement("div");
    root.className = "tut-anima-artists";
    const searchWrap = document.createElement("div");
    searchWrap.className = "tut-anima-search-wrap";
    const search = document.createElement("input");
    search.type = "search";
    search.className = "tut-anima-search";
    search.placeholder = "搜索 Anima 画师，回车可添加自定义标签";
    search.autocomplete = "off";
    const results = document.createElement("div");
    results.className = "tut-anima-results";
    results.hidden = true;
    const status = document.createElement("div");
    status.className = "tut-anima-status";
    const chips = document.createElement("div");
    chips.className = "tut-anima-chips";
    const hint = document.createElement("div");
    hint.className = "tut-anima-hint";
    hint.textContent = "左右拖动调整权重（0.1–3.0），双击编辑标签";
    searchWrap.append(search, results);
    root.append(searchWrap, status, chips, hint);

    const commit = () => {
        const previous = store.value;
        store.value = JSON.stringify(data);
        node.properties ||= {};
        node.properties[STATE_KEY] = store.value;
        store.callback?.(store.value, node, app);
        node.onWidgetChanged?.(store.name, store.value, previous, store);
        node.graph?.setDirtyCanvas?.(true, true);
    };

    const addArtist = (artist) => {
        const name = cleanName(artist?.name);
        if (!name) return;
        const existing = data.artists.find((item) => keyFor(item.name) === keyFor(name));
        if (existing) {
            existing.name = name;
            existing.weight = clampWeight(artist?.weight ?? existing.weight);
        } else {
            data.artists.push({ name, weight: clampWeight(artist?.weight) });
        }
        commit();
        drawChips();
    };

    const addTypedArtists = () => {
        const values = search.value.split(/[,，、\n\r]+/).map((part) => part.trim()).filter(Boolean);
        values.map(parseArtistToken).filter((item) => item.name).forEach(addArtist);
        search.value = "";
        results.hidden = true;
        status.textContent = "";
    };

    const scheduleEditorResize = () => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(() => {
            const contentHeight = Math.max(32, Math.ceil(chips.scrollHeight));
            const requiredNodeHeight = NODE_CHROME_HEIGHT + EDITOR_FIXED_HEIGHT + Math.min(AUTO_MAX_CHIPS_HEIGHT, contentHeight);
            const width = Math.max(node.size?.[0] || 0, 420);
            if ((node.size?.[1] || 0) < requiredNodeHeight) node.setSize?.([width, requiredNodeHeight]);
            node.graph?.setDirtyCanvas?.(true, true);
        });
    };

    const beginArtistEdit = (index, chip, nameLabel) => {
        const artist = data.artists[index];
        if (!artist || chip.querySelector(".tut-anima-edit")) return;
        finishActiveDrag?.();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "tut-anima-edit";
        input.value = artist.name;
        input.setAttribute("aria-label", "编辑画师标签");
        chip.replaceChild(input, nameLabel);
        let finished = false;
        const cancel = () => {
            if (finished) return;
            finished = true;
            drawChips();
        };
        const save = () => {
            if (finished) return;
            const nextName = cleanName(input.value);
            if (!nextName || nextName === artist.name) {
                cancel();
                return;
            }
            finished = true;
            node.graph?.beforeChange?.();
            artist.name = nextName;
            const duplicateIndex = data.artists.findIndex((item, itemIndex) => itemIndex !== index && keyFor(item.name) === keyFor(nextName));
            if (duplicateIndex >= 0) data.artists.splice(duplicateIndex, 1);
            commit();
            drawChips();
            node.graph?.afterChange?.();
        };
        input.addEventListener("pointerdown", (event) => event.stopPropagation());
        input.addEventListener("dblclick", (event) => event.stopPropagation());
        input.addEventListener("keydown", (event) => {
            event.stopPropagation();
            if (event.isComposing || event.keyCode === 229) return;
            if (event.key === "Enter") {
                event.preventDefault();
                save();
            } else if (event.key === "Escape") {
                event.preventDefault();
                cancel();
            }
        });
        input.addEventListener("blur", save, { once: true });
        input.focus();
        input.select();
        scheduleEditorResize();
    };

    const drawChips = () => {
        chips.replaceChildren();
        if (!data.artists.length) {
            const empty = document.createElement("div");
            empty.className = "tut-anima-empty";
            empty.textContent = "尚未添加画师";
            chips.append(empty);
            scheduleEditorResize();
            return;
        }
        data.artists.forEach((artist, index) => {
            const chip = document.createElement("div");
            chip.className = "tut-anima-chip";
            chip.title = "左右拖动调整权重";
            const name = document.createElement("span");
            name.className = "tut-anima-chip-name";
            name.textContent = `@${artist.name}`;
            const weight = document.createElement("span");
            weight.className = "tut-anima-weight";
            weight.textContent = Number(artist.weight).toFixed(1);
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "tut-anima-remove";
            remove.textContent = "×";
            remove.title = "删除画师";
            remove.addEventListener("pointerdown", (event) => event.stopPropagation());
            remove.addEventListener("click", (event) => {
                event.stopPropagation();
                node.graph?.beforeChange?.();
                data.artists.splice(index, 1);
                commit();
                drawChips();
                node.graph?.afterChange?.();
            });

            chip.addEventListener("pointerdown", (event) => {
                if (event.button !== 0 || event.target === remove) return;
                const startX = event.clientX;
                const startWeight = Number(artist.weight);
                finishActiveDrag?.();
                chip.setPointerCapture?.(event.pointerId);
                let dragging = false;
                const move = (moveEvent) => {
                    const distance = moveEvent.clientX - startX;
                    if (!dragging && Math.abs(distance) < 5) return;
                    if (!dragging) {
                        dragging = true;
                        chip.classList.add("dragging");
                        node.graph?.beforeChange?.();
                    }
                    moveEvent.preventDefault();
                    const next = clampWeight(startWeight + distance / 80);
                    if (next === artist.weight) return;
                    artist.weight = next;
                    weight.textContent = next.toFixed(1);
                    commit();
                };
                const finish = (finishEvent) => {
                    chip.classList.remove("dragging");
                    if (finishEvent?.pointerId !== undefined) chip.releasePointerCapture?.(finishEvent.pointerId);
                    chip.removeEventListener("pointermove", move);
                    chip.removeEventListener("pointerup", finish);
                    chip.removeEventListener("pointercancel", finish);
                    finishActiveDrag = null;
                    if (dragging) node.graph?.afterChange?.();
                };
                finishActiveDrag = finish;
                chip.addEventListener("pointermove", move);
                chip.addEventListener("pointerup", finish);
                chip.addEventListener("pointercancel", finish);
            });
            chip.addEventListener("dblclick", (event) => {
                if (event.target === remove) return;
                event.preventDefault();
                event.stopPropagation();
                beginArtistEdit(index, chip, name);
            });
            chip.append(name, weight, remove);
            chips.append(chip);
        });
        scheduleEditorResize();
    };

    const drawResults = (artists) => {
        results.replaceChildren();
        if (!artists.length) {
            results.hidden = true;
            status.className = "tut-anima-status";
            status.textContent = search.value.trim() ? "没有数据库匹配；按回车添加自定义画师" : "";
            return;
        }
        for (const artist of artists) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "tut-anima-result";
            const tag = document.createElement("span");
            tag.textContent = artist.tag;
            const count = document.createElement("span");
            count.className = "tut-anima-count";
            count.textContent = `使用数 ${Number(artist.usage_count || 0).toLocaleString()}`;
            button.append(tag, count);
            button.addEventListener("mousedown", (event) => event.preventDefault());
            button.addEventListener("click", () => {
                addArtist({ name: artist.tag, weight: 1 });
                search.value = "";
                results.hidden = true;
                status.textContent = "";
            });
            results.append(button);
        }
        results.hidden = false;
        status.className = "tut-anima-status";
        status.textContent = `${artists.length} 个匹配结果`;
    };

    const runSearch = async (expectedQuery = search.value.trim()) => {
        const current = requestNumber;
        const controller = new AbortController();
        activeController = controller;
        try {
            const params = new URLSearchParams({ q: expectedQuery, limit: "20" });
            const response = await fetch(api.apiURL(`/tut_nodes/anima/artists/search?${params}`), { signal: controller.signal });
            const payload = await response.json();
            if (current !== requestNumber || search.value.trim() !== expectedQuery) return;
            if (!response.ok) throw new Error(payload?.error || "画师数据库搜索失败");
            drawResults(Array.isArray(payload?.artists) ? payload.artists : []);
        } catch (error) {
            if (error?.name === "AbortError" || current !== requestNumber) return;
            results.hidden = true;
            status.className = "tut-anima-status error";
            status.textContent = `${error?.message || "画师数据库不可用"}；仍可按回车添加自定义画师`;
        }
    };

    const scheduleSearch = () => {
        if (composing) return;
        requestNumber += 1;
        clearTimeout(debounceTimer);
        activeController?.abort();
        const expectedQuery = search.value.trim();
        debounceTimer = setTimeout(() => runSearch(expectedQuery), 180);
    };
    search.addEventListener("input", scheduleSearch);
    search.addEventListener("compositionstart", () => {
        composing = true;
        requestNumber += 1;
        clearTimeout(debounceTimer);
        activeController?.abort();
    });
    search.addEventListener("compositionend", () => {
        composing = false;
        scheduleSearch();
    });
    search.addEventListener("focus", scheduleSearch);
    search.addEventListener("keydown", (event) => {
        if (composing || event.isComposing || event.keyCode === 229) return;
        if (event.key === "Enter") {
            event.preventDefault();
            addTypedArtists();
        } else if (event.key === "Escape") {
            results.hidden = true;
        }
    });
    search.addEventListener("blur", () => setTimeout(() => { results.hidden = true; }, 120));

    node.addDOMWidget("tut_anima_artist_editor", "TUT_ANIMA_ARTIST_EDITOR", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => COMPACT_EDITOR_HEIGHT,
        afterResize: () => node.graph?.setDirtyCanvas?.(true, true),
    });
    const priorSerialize = node.onSerialize;
    node.onSerialize = function (info) {
        const result = priorSerialize?.apply(this, arguments);
        const serialized = JSON.stringify(data);
        store.value = serialized;
        info.properties ||= {};
        info.properties[STATE_KEY] = serialized;
        info.widgets_values_named ||= {};
        info.widgets_values_named.artist_data = serialized;
        return result;
    };
    node.setSize?.([Math.max(node.size?.[0] || 0, 420), Math.max(node.size?.[1] || 0, COMPACT_NODE_HEIGHT)]);
    drawChips();

    node.__tutAnimaArtistEditor = {
        reload(info) {
            const candidates = [
                info?.properties?.[STATE_KEY],
                info?.widgets_values_named?.artist_data,
                node.properties?.[STATE_KEY],
                store.value,
            ];
            data = parseData(candidates.find(isSerializedArtistData));
            store.value = JSON.stringify(data);
            node.properties ||= {};
            node.properties[STATE_KEY] = store.value;
            drawChips();
        },
        destroy() {
            clearTimeout(debounceTimer);
            cancelAnimationFrame(resizeFrame);
            activeController?.abort();
            finishActiveDrag?.();
        },
    };
}

app.registerExtension({
    name: "TUT_Nodes.AnimaArtistPromptMixer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        const originalConfigure = nodeType.prototype.onConfigure;
        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const store = this.widgets?.find((widget) => widget.name === "artist_data");
            if (store) installEditor(this, store);
            return result;
        };
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            const info = arguments[0];
            queueMicrotask(() => this.__tutAnimaArtistEditor?.reload(info));
            return result;
        };
        nodeType.prototype.onRemoved = function () {
            this.__tutAnimaArtistEditor?.destroy();
            return originalRemoved?.apply(this, arguments);
        };
    },
});
