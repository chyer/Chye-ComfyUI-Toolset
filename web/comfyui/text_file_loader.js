import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { RgthreeBetterButtonWidget } from "./utils_widgets.js";
import { rgthree } from "./rgthree.js";

class CYHTextFileLoaderNode extends LGraphNode {
    constructor(title = "CYH Text File Loader") {
        super(title);
        this.comfyClass = "CYHTextFileLoader";
        this.serialize_widgets = true;
    }

    onNodeCreated() {
        // Add a button widget for loading files
        this.addWidget("button", "📁 Browse Files", "", () => {
            this.handleFileBrowse();
        }, { serialize: false });

        // Add a reload button
        this.addWidget("button", "🔄 Reload File", "", () => {
            this.handleReload();
        }, { serialize: false });
    }

    async handleFileBrowse() {
        try {
            // Use ComfyUI's file browser API
            const filePath = await app.openFileBrowser("text");
            if (filePath) {
                // Update the file path widget
                const filePathWidget = this.widgets.find(w => w.name === "file_path");
                if (filePathWidget) {
                    filePathWidget.value = filePath;
                    this.onWidgetChanged?.(filePathWidget);
                }
                
                // Trigger file loading
                await this.handleReload();
            }
        } catch (error) {
            console.error("Error browsing files:", error);
        }
    }

    async handleReload() {
        try {
            // Get the current file path
            const filePathWidget = this.widgets.find(w => w.name === "file_path");
            if (!filePathWidget || !filePathWidget.value) {
                console.warn("No file path specified");
                return;
            }

            // Call the backend API to load the file
            const response = await api.fetchApi("/chye/text_file_loader/load", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    file_path: filePathWidget.value
                })
            });

            if (response.ok) {
                const result = await response.json();
                console.log("File loaded successfully:", result);
                
                if (result.success) {
                    // Update the text content widget if it exists
                    const textWidget = this.widgets.find(w => w.name === "editable_text");
                    if (textWidget && result.content) {
                        textWidget.value = result.content;
                        this.onWidgetChanged?.(textWidget);
                    }
                    
                    // Trigger node execution
                    this.triggerExecution();
                } else {
                    console.error("Failed to load file:", result.error);
                    // Show error in the text widget if possible
                    const textWidget = this.widgets.find(w => w.name === "editable_text");
                    if (textWidget) {
                        textWidget.value = `Error: ${result.error}`;
                        this.onWidgetChanged?.(textWidget);
                    }
                }
            } else {
                const errorText = await response.text();
                console.error("Failed to load file:", errorText);
                
                // Show error in the text widget if possible
                const textWidget = this.widgets.find(w => w.name === "editable_text");
                if (textWidget) {
                    textWidget.value = `HTTP Error: ${response.status} - ${errorText}`;
                    this.onWidgetChanged?.(textWidget);
                }
            }
        } catch (error) {
            console.error("Error reloading file:", error);
        }
    }

    triggerExecution() {
        // Force the node to be marked as changed to trigger execution
        this.setDirtyCanvas(true, true);
        
        // If the node has outputs, trigger connected nodes
        if (this.outputs && this.outputs.length > 0) {
            this.triggerSlot(0);
        }
    }

    onWidgetChanged(widget) {
        // Handle widget changes
        if (widget.name === "file_path") {
            // File path changed, we might want to auto-reload
            this.handleReload();
        }
    }
}

// Register the extension
app.registerExtension({
    name: "Chye.TextFileLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CYHTextFileLoader") {
            // Replace the default implementation with our custom one
            LiteGraph.registerNodeType("CYHTextFileLoader", CYHTextFileLoaderNode);
        }
    },
});

// Also register for any existing nodes
app.graph._nodes.forEach(node => {
    if (node.type === "CYHTextFileLoader" && !(node instanceof CYHTextFileLoaderNode)) {
        const newNode = new CYHTextFileLoaderNode(node.title);
        newNode.configure(node.serialize());
        app.graph.replace(node, newNode);
    }
});