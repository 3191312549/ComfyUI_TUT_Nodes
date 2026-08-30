import { app } from "/scripts/app.js";

const NODE_ID = "TUT_ImageToBatch";
const MAX_INPUTS = 10;
const IMAGE_NAME = /^image_(\d+)$/;

function labelInputs(node) {
    node.inputs?.forEach((input) => {
        const match = IMAGE_NAME.exec(input.name || "");
        if (match) input.label = `图像 ${match[1]}`;
    });
}

function installDynamicInputs(node) {
    if (node.comfyClass !== NODE_ID || node.__tutImageToBatchInputs) return;
    let scheduled = false;
    let normalizing = false;

    function normalizeInputs() {
        if (normalizing) return;
        normalizing = true;
        try {
            if (!Array.isArray(node.inputs)) node.inputs = [];
            const imageInputs = node.inputs.filter((input) => IMAGE_NAME.test(input.name || ""));
            const emptyIndexes = imageInputs
                .filter((input) => input.link == null)
                .map((input) => node.inputs.indexOf(input))
                .filter((index) => index >= 0)
                .sort((first, second) => second - first);
            for (const index of emptyIndexes) node.removeInput(index);

            const connected = node.inputs.filter(
                (input) => IMAGE_NAME.test(input.name || "") && input.link != null,
            ).slice(0, MAX_INPUTS);
            connected.forEach((input, index) => {
                input.name = `image_${index + 1}`;
                input.label = `图像 ${index + 1}`;
                input.type = "IMAGE";
                input.shape = 7;
            });
            if (connected.length < MAX_INPUTS) {
                const index = connected.length + 1;
                const input = node.addInput(`image_${index}`, "IMAGE", { shape: 7 });
                if (input) input.label = `图像 ${index}`;
            }
            labelInputs(node);
            node.setDirtyCanvas?.(true, true);
            app.graph?.setDirtyCanvas?.(true, true);
        } finally {
            normalizing = false;
        }
    }

    function scheduleNormalize() {
        if (scheduled) return;
        scheduled = true;
        setTimeout(() => {
            scheduled = false;
            normalizeInputs();
        }, 20);
    }

    const previousConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) {
        const result = previousConnectionsChange?.apply(this, args);
        scheduleNormalize();
        return result;
    };
    const previousConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = previousConfigure?.apply(this, args);
        scheduleNormalize();
        return result;
    };
    node.__tutImageToBatchInputs = true;
    scheduleNormalize();
}

app.registerExtension({
    name: "TUT_Nodes.ImageToBatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) return;
        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previousCreated?.apply(this, args);
            installDynamicInputs(this);
            return result;
        };
    },
    nodeCreated(node) {
        installDynamicInputs(node);
    },
});
