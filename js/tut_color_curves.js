import { app } from "/scripts/app.js";

const CHANNELS = ["RGB", "R", "G", "B"];
const CHANNEL_COLORS = { RGB: "#e6e6e6", R: "#ff6868", G: "#67dc83", B: "#6e9cff" };
const identity = () => [{ x: 0, y: 0 }, { x: 1, y: 1 }];
const defaultCurves = () => ({ version: 1, channels: Object.fromEntries(CHANNELS.map((key) => [key, identity()])) });

function parseOr(value, fallback) {
    try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === "object" ? parsed : fallback();
    } catch (_) {
        return fallback();
    }
}

function hideSerializedWidget(widget) {
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.serializeValue = () => widget.value;
}

function commit(node, widget, payload) {
    widget.value = JSON.stringify(payload);
    widget.callback?.(widget.value, node, app);
    node.graph?.setDirtyCanvas?.(true, true);
}

function panel() {
    const element = document.createElement("div");
    Object.assign(element.style, {
        boxSizing: "border-box", width: "100%", padding: "8px", borderRadius: "6px",
        background: "#202124", color: "#ddd", font: "12px sans-serif", userSelect: "none",
    });
    return element;
}

function button(label, action) {
    const item = document.createElement("button");
    item.textContent = label;
    Object.assign(item.style, { margin: "2px", padding: "3px 8px", cursor: "pointer" });
    item.addEventListener("click", action);
    return item;
}

function monotoneSlopes(points) {
    const widths = [];
    const secants = [];
    for (let index = 0; index < points.length - 1; index++) {
        const width = points[index + 1].x - points[index].x;
        widths.push(width);
        secants.push((points[index + 1].y - points[index].y) / width);
    }

    const slopes = new Array(points.length);
    slopes[0] = secants[0];
    slopes[slopes.length - 1] = secants[secants.length - 1];
    for (let index = 1; index < points.length - 1; index++) {
        const left = secants[index - 1];
        const right = secants[index];
        if (left === 0 || right === 0 || left * right < 0) {
            slopes[index] = 0;
        } else {
            const w1 = 2 * widths[index] + widths[index - 1];
            const w2 = widths[index] + 2 * widths[index - 1];
            slopes[index] = (w1 + w2) / (w1 / left + w2 / right);
        }
    }

    secants.forEach((secant, index) => {
        if (secant === 0) {
            slopes[index] = 0;
            slopes[index + 1] = 0;
            return;
        }
        const a = slopes[index] / secant;
        const b = slopes[index + 1] / secant;
        const magnitude = a * a + b * b;
        if (magnitude > 9) {
            const scale = 3 / Math.sqrt(magnitude);
            slopes[index] = scale * a * secant;
            slopes[index + 1] = scale * b * secant;
        }
    });
    return slopes;
}

function sampleMonotoneCurve(points, slopes, x) {
    if (points.length === 2) {
        return points[0].y + (points[1].y - points[0].y) * x;
    }
    let interval = points.length - 2;
    for (let index = 0; index < points.length - 1; index++) {
        if (x <= points[index + 1].x) {
            interval = index;
            break;
        }
    }
    const left = points[interval];
    const right = points[interval + 1];
    const width = right.x - left.x;
    const t = (x - left.x) / width;
    const t2 = t * t;
    const t3 = t2 * t;
    const value = (2 * t3 - 3 * t2 + 1) * left.y
        + (t3 - 2 * t2 + t) * width * slopes[interval]
        + (-2 * t3 + 3 * t2) * right.y
        + (t3 - t2) * width * slopes[interval + 1];
    return Math.min(1, Math.max(0, value));
}

function installCurveEditor(node, widget) {
    hideSerializedWidget(widget);
    const payload = parseOr(widget.value, defaultCurves);
    payload.version = 1;
    payload.channels ||= {};
    for (const channel of CHANNELS) {
        if (!Array.isArray(payload.channels[channel]) || payload.channels[channel].length < 2) {
            payload.channels[channel] = identity();
        }
    }

    let active = "RGB";
    let selected = -1;
    let dragging = false;
    const root = panel();
    const toolbar = document.createElement("div");
    const canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 190;
    Object.assign(canvas.style, { width: "100%", height: "190px", background: "#111", cursor: "crosshair" });

    const sortedPoints = () => payload.channels[active].sort((a, b) => a.x - b.x);
    const draw = () => {
        const ctx = canvas.getContext("2d");
        const width = canvas.width, height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        for (let index = 1; index < 4; index++) {
            const x = width * index / 4, y = height * index / 4;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
        }
        const points = sortedPoints();
        const slopes = monotoneSlopes(points);
        ctx.strokeStyle = CHANNEL_COLORS[active];
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        for (let sample = 0; sample <= width; sample++) {
            const x = sample / width;
            const y = sampleMonotoneCurve(points, slopes, x);
            sample ? ctx.lineTo(sample, (1 - y) * height) : ctx.moveTo(0, (1 - y) * height);
        }
        ctx.stroke();

        ctx.save();
        ctx.strokeStyle = "rgba(220, 220, 220, 0.72)";
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(0, height);
        ctx.lineTo(width, 0);
        ctx.stroke();
        ctx.restore();

        points.forEach((point, index) => {
            ctx.fillStyle = index === selected ? "#ffd54f" : CHANNEL_COLORS[active];
            ctx.beginPath(); ctx.arc(point.x * width, (1 - point.y) * height, 5, 0, Math.PI * 2); ctx.fill();
        });
    };
    const locate = (event) => {
        const bounds = canvas.getBoundingClientRect();
        return {
            x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
            y: Math.min(1, Math.max(0, 1 - (event.clientY - bounds.top) / bounds.height)),
        };
    };
    const nearest = (position) => {
        let best = -1, distance = 0.04;
        sortedPoints().forEach((point, index) => {
            const value = Math.hypot(point.x - position.x, point.y - position.y);
            if (value < distance) { best = index; distance = value; }
        });
        return best;
    };
    CHANNELS.forEach((channel) => toolbar.append(button(channel, () => {
        active = channel; selected = -1; draw();
    })));
    toolbar.append(button("重置当前", () => {
        payload.channels[active] = identity(); selected = -1; commit(node, widget, payload); draw();
    }));

    canvas.addEventListener("pointerdown", (event) => {
        const position = locate(event);
        selected = nearest(position);
        if (selected < 0 && sortedPoints().length < 16) {
            payload.channels[active].push(position);
            selected = sortedPoints().findIndex((point) => point === position);
        }
        dragging = selected >= 0;
        canvas.setPointerCapture?.(event.pointerId);
        draw();
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!dragging || selected < 0) return;
        const position = locate(event);
        const points = sortedPoints();
        const point = points[selected];
        point.y = position.y;
        if (selected !== 0 && selected !== points.length - 1) {
            point.x = Math.min(points[selected + 1].x - 0.001, Math.max(points[selected - 1].x + 0.001, position.x));
        }
        draw();
    });
    const finish = (event) => {
        if (!dragging) return;
        dragging = false;
        canvas.releasePointerCapture?.(event.pointerId);
        commit(node, widget, payload);
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
    canvas.addEventListener("dblclick", (event) => {
        const index = nearest(locate(event));
        const points = sortedPoints();
        if (index > 0 && index < points.length - 1) {
            points.splice(index, 1); selected = -1; commit(node, widget, payload); draw();
        }
    });
    root.append(toolbar, canvas);
    node.addDOMWidget("curve_editor", "tut_curve_editor", root, { serialize: false, hideOnZoom: true });
    node.setSize?.([Math.max(node.size?.[0] || 320, 340), Math.max(node.size?.[1] || 0, 360)]);
    commit(node, widget, payload);
    draw();
}

app.registerExtension({
    name: "TUT_Nodes.ColorCurves",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TUT_ColorCurves") return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = previous?.apply(this, arguments);
            const widget = this.widgets?.find((item) => item.name === "curve_data");
            if (widget) installCurveEditor(this, widget);
            return result;
        };
    },
});
