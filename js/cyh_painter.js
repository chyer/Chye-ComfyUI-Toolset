// CYH Interactive Painter Node - Frontend JavaScript
class CYHInteractivePainterNode {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.currentBrushSize = 20;
        this.currentBrushColor = "#FF0000";
        this.node = null;
    }

    async setup() {
        // Create canvas container
        const container = document.createElement('div');
        container.style.cssText = 'margin: 10px 0; border: 1px solid #ccc; padding: 10px;';
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.width = 512;
        this.canvas.height = 512;
        this.canvas.style.cssText = 'border: 1px solid #666; cursor: crosshair; background: white;';
        container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Create controls
        const controls = document.createElement('div');
        controls.style.cssText = 'margin: 10px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap;';
        
        // Brush size control
        const sizeContainer = document.createElement('div');
        sizeContainer.style.cssText = 'display: flex; align-items: center; gap: 5px;';
        
        const sizeLabel = document.createElement('label');
        sizeLabel.textContent = 'Brush Size:';
        sizeLabel.style.marginRight = '5px';
        sizeContainer.appendChild(sizeLabel);
        
        const sizeInput = document.createElement('input');
        sizeInput.type = 'range';
        sizeInput.min = '1';
        sizeInput.max = '100';
        sizeInput.value = '20';
        sizeInput.style.width = '80px';
        sizeInput.addEventListener('input', (e) => {
            this.currentBrushSize = parseInt(e.target.value);
        });
        sizeContainer.appendChild(sizeInput);
        
        const sizeValue = document.createElement('span');
        sizeValue.textContent = '20px';
        sizeValue.style.minWidth = '30px';
        sizeInput.addEventListener('input', (e) => {
            sizeValue.textContent = e.target.value + 'px';
        });
        sizeContainer.appendChild(sizeValue);
        
        controls.appendChild(sizeContainer);
        
        // Color control
        const colorContainer = document.createElement('div');
        colorContainer.style.cssText = 'display: flex; align-items: center; gap: 5px;';
        
        const colorLabel = document.createElement('label');
        colorLabel.textContent = 'Color:';
        colorLabel.style.marginRight = '5px';
        colorContainer.appendChild(colorLabel);
        
        const colorInput = document.createElement('input');
        colorInput.type = 'color';
        colorInput.value = '#FF0000';
        colorInput.style.width = '40px';
        colorInput.style.height = '25px';
        colorInput.addEventListener('input', (e) => {
            this.currentBrushColor = e.target.value;
        });
        colorContainer.appendChild(colorInput);
        
        controls.appendChild(colorContainer);
        
        // Clear button
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear Canvas';
        clearBtn.style.padding = '5px 10px';
        clearBtn.style.border = '1px solid #666';
        clearBtn.style.borderRadius = '3px';
        clearBtn.style.backgroundColor = '#f0f0f0';
        clearBtn.addEventListener('click', () => {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.sendCanvasUpdate();
        });
        controls.appendChild(clearBtn);
        
        container.appendChild(controls);
        
        // Instructions
        const instructions = document.createElement('div');
        instructions.textContent = 'Click and drag to draw with the brush';
        instructions.style.marginTop = '10px';
        instructions.style.fontSize = '12px';
        instructions.style.color = '#666';
        container.appendChild(instructions);
        
        // Add event listeners
        this.setupEventListeners();
        
        return container;
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
        
        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0]);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    startDrawing(e) {
        this.isDrawing = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastX = e.clientX - rect.left;
        this.lastY = e.clientY - rect.top;
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        this.ctx.lineJoin = 'round';
        this.ctx.lineCap = 'round';
        this.ctx.lineWidth = this.currentBrushSize;
        this.ctx.strokeStyle = this.currentBrushColor;
        this.ctx.globalCompositeOperation = 'source-over';
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(currentX, currentY);
        this.ctx.stroke();
        
        this.lastX = currentX;
        this.lastY = currentY;
        
        // Send incremental updates
        this.sendCanvasUpdate();
    }

    stopDrawing() {
        this.isDrawing = false;
        // Final update when drawing stops
        this.sendCanvasUpdate();
    }

    async sendCanvasUpdate() {
        const canvasData = this.canvas.toDataURL('image/png');
        
        try {
            const response = await fetch('/cyh_painter/canvas_update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    unique_id: this.node.unique_id,
                    canvas_data: canvasData
                })
            });
            
            if (!response.ok) {
                console.error('Failed to send canvas update');
            }
        } catch (error) {
            console.error('Error sending canvas update:', error);
        }
    }

    async onCustomWidget(node, inputName, app) {
        this.node = node;
        const container = await this.setup();
        
        // Load initial image if provided
        if (node.inputs && node.inputs.image) {
            const img = new Image();
            img.onload = () => {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.drawImage(img, 0, 0);
            };
            img.src = node.inputs.image;
        }
        
        return container;
    }
}

// Register the widget
app.registerExtension({
    name: "comfy.cyh_interactive_painter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CYHInteractivePainterNode") {
            nodeType.prototype.onCustomWidget = CYHInteractivePainterNode.prototype.onCustomWidget;
        }
    }
});