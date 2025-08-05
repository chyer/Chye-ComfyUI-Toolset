# ComfyUI ASPLatent Node Visual Mockup

## Node Appearance in ComfyUI Interface

```
┌──────────────────────────────────────────┐
│            🎯 ASP Latent Generator        │
├──────────────────────────────────────────┤
│                                          │
│  Model Type:     [Flux        ▼]         │
│                                          │
│  Aspect Ratio:   [16:9        ▼]         │
│                                          │
│  Orientation:    [Portrait    ▼]         │
│                                          │
│  Multiplier:     [1.0        ] 📏        │
│                                          │
│  Batch Size:     [1          ] 📦        │
│                                          │
├──────────────────────────────────────────┤
│                                          │
│  📊 Resolution Preview: 928 × 1664       │
│                                          │
│                             ●─── LATENT │
└──────────────────────────────────────────┘
```

## Input Field Details

### Model Type Dropdown
```
┌─────────────────┐
│ Flux           │
│ Qwen Image     │
│ SDXL           │
└─────────────────┘
```

### Aspect Ratio Dropdown
```
┌─────────────────┐
│ 1:1 (Square)   │
│ 4:3 (Classic)  │
│ 3:2 (Photo)    │
│ 16:9 (Wide)    │
│ 21:9 (Cinema)  │
└─────────────────┘
```

### Orientation Dropdown
```
┌─────────────────┐
│ Portrait       │ ← Default
│ Landscape      │
└─────────────────┘
```

### Interactive Features

1. **Dynamic Resolution Preview**: Shows calculated width × height in real-time
2. **Model-Aware Tooltips**: Hover hints showing optimal use cases for each model
3. **Validation Indicators**: Visual feedback for valid/invalid multiplier values
4. **Smart Defaults**: Automatically selects portrait orientation and 1.0 multiplier

## Node Color Scheme
- **Header**: Gradient from purple to blue (#6B46C1 → #3B82F6)
- **Background**: Dark gray (#374151)
- **Text**: Light gray (#F3F4F6)
- **Inputs**: Dark blue accent (#1E3A8A)
- **Output**: Green accent (#059669)

## Workflow Integration Example

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ASP Latent    │    │   KSampler      │    │  VAE Decode     │
│   Generator     │────│                 │────│                 │
│                 │    │                 │    │                 │
│ Model: Flux     │    │                 │    │                 │
│ Ratio: 16:9     │    │                 │    │                 │
│ Orient: Portrait│    │                 │    │                 │
│ Multi: 1.0      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Real-time Preview Examples

### Flux Model Selected
```
Model: Flux → Aspect: 16:9 → Portrait → Multi: 1.0
📊 Resolution Preview: 768 × 1344
```

### Qwen Model Selected
```
Model: Qwen → Aspect: 4:3 → Portrait → Multi: 1.5
📊 Resolution Preview: 1710 × 2208 (adjusted to multiples of 32)
```

### SDXL Model Selected
```
Model: SDXL → Aspect: 1:1 → Landscape → Multi: 0.5
📊 Resolution Preview: 512 × 512