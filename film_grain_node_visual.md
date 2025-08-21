# CYH Film Grain Node Visual Representation

## Node Appearance in ComfyUI

```
┌─────────────────────────────────────────────────────────────┐
│ 🎬 CYH Post Process | Film Grain                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [IMAGE] ──────────────────────────────────────────────┐    │
│                                                     │    │
│  ┌─────────────────────────────────────────────────┐ │    │
│  │               INPUTS                            │ │    │
│  │                                                 │ │    │
│  │  image: (IMAGE)                                 │ │    │
│  │                                                 │ │    │
│  │  strength: 0.5 ────────────────────────────────┐ │ │    │
│  │  [0.0]───[■■■■■■■■■■■■■■■■■■■■■■■■■]───[1.0]   │ │ │    │
│  │                                                 │ │ │    │
│  │  iso: 400 ───────────────────────────────────┐ │ │ │    │
│  │  [100]───[■■■■■■■■■■■■■■■■■■■■■■■■■]───[6400]  │ │ │ │    │
│  │                                                 │ │ │ │    │
│  │  grain_size: 2.0 ────────────────────────────┐ │ │ │ │    │
│  │  [1.0]───[■■■■■■■■■■■■■■■■■■■■■■■■■]───[10.0]  │ │ │ │ │    │
│  │                                                 │ │ │ │    │
│  │  colored: ☐ False                              │ │ │ │    │
│  │                                                 │ │ │ │    │
│  └─────────────────────────────────────────────────┘ │ │    │
│                                                     │ │    │
│                                                     ▼ │    │
│  ┌─────────────────────────────────────────────────┐ │    │
│  │               OUTPUTS                           │ │    │
│  │                                                 │ │    │
│  │  IMAGE ────────────────────────────────────────┼─┘    │
│  │                                                 │      │
│  └─────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Node Parameters

### Inputs
- **image** (IMAGE): The input image to apply film grain to
- **strength** (FLOAT): Overall grain strength (0.0 to 1.0)
  - Default: 0.5
  - Controls the intensity of the grain effect
- **iso** (INT): ISO value to simulate (100 to 6400)
  - Default: 400
  - Higher values produce more pronounced grain, simulating high-ISO film
- **grain_size** (FLOAT): Size of grain particles (1.0 to 10.0)
  - Default: 2.0
  - Controls the physical size of the grain particles
- **colored** (BOOLEAN): Whether to apply colored grain
  - Default: False
  - When checked, applies different grain patterns to each color channel
  - When unchecked, applies monochrome grain to all channels

### Outputs
- **IMAGE**: The processed image with film grain applied

## Node Functionality

The Film Grain node simulates realistic photographic film grain by:

1. **Analyzing luminance** - Shadows receive more apparent grain than highlights
2. **Generating correlated noise** - Creates spatially correlated noise patterns that mimic real film grain
3. **Applying ISO-based scaling** - Higher ISO values produce more pronounced grain
4. **Optional colored grain** - Can apply either monochrome or colored grain patterns

## Usage Examples

### Subtle Film Grain
```
strength: 0.2
iso: 200
grain_size: 1.5
colored: False
```
Perfect for adding a subtle film-like texture to digital images.

### Heavy Film Grain
```
strength: 0.8
iso: 3200
grain_size: 4.0
colored: True
```
Creates a strong, noticeable grain effect similar to high-speed color film.

### Classic Black & White Film
```
strength: 0.5
iso: 400
grain_size: 2.5
colored: False
```
Simulates the look of traditional black and white film stock.