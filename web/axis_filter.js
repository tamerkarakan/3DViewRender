async function getComfyApp() {
    const comfyApi = globalThis.comfyAPI;
    const app = comfyApi?.app?.app ?? comfyApi?.app;
    if (app?.registerExtension) {
        return app;
    }

    const legacy = await import("../../scripts/app.js");
    return legacy.app;
}

const NODE_ID = "T3DViewRenderSixSides";
const AXES = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"];
const OPPOSITE = {
    "+Z": "-Z",
    "-Z": "+Z",
    "+Y": "-Y",
    "-Y": "+Y",
    "+X": "-X",
    "-X": "+X",
};
const DEFAULT_FRONT = {
    "+Z": "-Y",
    "-Z": "+Y",
    "+Y": "+Z",
    "-Y": "+Z",
    "+X": "-Z",
    "-X": "+Z",
};

function canonicalAxis(value, fallback) {
    const text = String(value ?? fallback).trim().toUpperCase();
    return AXES.includes(text) ? text : fallback;
}

function validFrontAxes(upAxis) {
    const up = canonicalAxis(upAxis, "+Z");
    return AXES.filter((axis) => axis !== up && axis !== OPPOSITE[up]);
}

function setComboValues(widget, values) {
    widget.options = widget.options ?? {};
    widget.options.values = [...values];
}

function markDirty(node) {
    node.graph?.setDirtyCanvas?.(true, true);
}

function bindAxisWidgets(node) {
    const upWidget = node.widgets?.find((widget) => widget.name === "up_axis");
    const frontWidget = node.widgets?.find((widget) => widget.name === "front_axis");
    if (!upWidget || !frontWidget || frontWidget.__threeDViewRenderAxisBound) {
        return;
    }

    frontWidget.__threeDViewRenderAxisBound = true;

    const refreshFrontAxis = () => {
        const up = canonicalAxis(upWidget.value, "+Z");
        const valid = validFrontAxes(up);
        const preferred = valid.includes(DEFAULT_FRONT[up]) ? DEFAULT_FRONT[up] : valid[0];

        setComboValues(frontWidget, valid);
        if (!valid.includes(frontWidget.value)) {
            frontWidget.value = preferred;
            frontWidget.callback?.(frontWidget.value);
        }
        markDirty(node);
    };

    const originalUpCallback = upWidget.callback;
    upWidget.callback = function (value, ...rest) {
        const result = originalUpCallback?.apply(this, [value, ...rest]);
        refreshFrontAxis();
        return result;
    };

    const originalConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = originalConfigure?.apply(this, args);
        queueMicrotask(refreshFrontAxis);
        return result;
    };

    refreshFrontAxis();
}

function isTargetNode(node) {
    return (
        node.comfyClass === NODE_ID ||
        node.type === NODE_ID ||
        node.comfyClass === "3D View Render: Six Sides" ||
        node.type === "3D View Render: Six Sides"
    );
}

const app = await getComfyApp();

app.registerExtension({
    name: "3dviewrender.axis_filter",
    nodeCreated(node) {
        if (!isTargetNode(node)) {
            return;
        }
        queueMicrotask(() => bindAxisWidgets(node));
    },
});
