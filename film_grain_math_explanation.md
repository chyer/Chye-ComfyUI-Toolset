# Mathematical Explanation of Film Grain Node Parameters

## Overview
The film grain node uses several mathematical concepts to simulate realistic photographic film grain. Each parameter plays a specific role in the final effect.

## Parameter Mathematics

### 1. Strength Parameter
```python
strength: 0.0 to 1.0
```

**Mathematical Role**: Global multiplier for grain intensity

**Formula Application**:
```
final_grain = grain_pattern * iso_factor * strength * shadow_boost
```

**Explanation**:
- Acts as a linear multiplier on the final grain pattern
- Value of 0.0 results in no grain (original image)
- Value of 1.0 applies full grain intensity
- Provides direct user control over the overall visibility of grain

### 2. ISO Parameter
```python
iso: 100 to 6400
```

**Mathematical Role**: Simulates film sensitivity through logarithmic scaling

**Formula Application**:
```
iso_factor = log10(iso / 100)
```

**Explanation**:
- Uses logarithmic scaling to simulate how real film grain responds to ISO
- ISO 100 = log10(1) = 0 (no additional grain from ISO)
- ISO 400 = log10(4) ≈ 0.602 (moderate grain increase)
- ISO 1600 = log10(16) ≈ 1.204 (significant grain)
- ISO 6400 = log10(64) ≈ 1.806 (heavy grain)
- This logarithmic relationship mimics how photographic film grain increases with sensitivity

### 3. Grain Size Parameter
```python
grain_size: 1.0 to 10.0
```

**Mathematical Role**: Controls spatial correlation of noise through Gaussian filtering

**Formula Application**:
```
sigma = max(0.5, grain_size / 10.0)
correlated_noise = gaussian_filter(noise, sigma=sigma)
```

**Explanation**:
- Converts grain size to sigma value for Gaussian filter
- Smaller values (1.0-2.0) create fine, sharp grain
- Larger values (8.0-10.0) create coarse, soft grain
- The Gaussian filter creates spatial correlation in the noise pattern
- Higher sigma values increase the correlation radius, making grain clump together

### 4. Colored Parameter
```python
colored: Boolean (True/False)
```

**Mathematical Role**: Determines grain application pattern across color channels

**Monochrome Mode (colored=False)**:
```
for c in range(image_np.shape[3]):
    result[0,:,:,c] = image_np[0,:,:,c] + final_grain
```

**Colored Mode (colored=True)**:
```
grain_r = generate_correlated_noise(shape, grain_size)
grain_g = generate_correlated_noise(shape, grain_size)
grain_b = generate_correlated_noise(shape, grain_size)

result[0,:,:,0] = image_np[0,:,:,0] + grain_r
result[0,:,:,1] = image_np[0,:,:,1] + grain_g
result[0,:,:,2] = image_np[0,:,:,2] + grain_b
```

**Explanation**:
- Monochrome: Same grain pattern applied to all RGB channels
- Colored: Independent grain patterns for each channel
- Colored grain simulates the chromatic noise found in color film
- Independent patterns create more realistic color film grain appearance

## Supporting Mathematics

### Luminance Calculation
```
luminance = 0.299 * R + 0.587 * G + 0.114 * B
```

**Explanation**:
- Uses standard RGB to luminance conversion weights
- These weights are based on human eye sensitivity to different colors
- Green contributes most to perceived brightness, red less, blue least

### Shadow Boost Calculation
```
shadow_boost = 1.0 + (1.0 - luminance) * 0.5
```

**Explanation**:
- Creates a multiplier based on image darkness
- Bright areas (luminance ≈ 1.0): shadow_boost ≈ 1.0 (no boost)
- Dark areas (luminance ≈ 0.0): shadow_boost ≈ 1.5 (50% boost)
- Simulates how film grain is more visible in shadow areas
- The 0.5 factor controls the maximum boost intensity

### Gaussian Filter for Correlated Noise
```
correlated_noise = gaussian_filter(noise, sigma=sigma)
```

**Mathematical Background**:
- Applies a 2D Gaussian kernel to random noise
- Kernel weights follow: G(x,y) = (1/2πσ²) * e^(-(x²+y²)/(2σ²))
- Creates spatial correlation by averaging nearby noise values
- Larger sigma values increase the correlation distance

### Noise Generation
```
noise = np.random.normal(0, 1, shape)
```

**Mathematical Background**:
- Generates values from normal distribution with μ=0, σ=1
- Probability density function: f(x) = (1/√(2π)) * e^(-x²/2)
- Creates the base random pattern for grain simulation

## Complete Mathematical Flow

1. **Base Noise Generation**:
   ```
   noise = N(0,1)  # Normal distribution
   ```

2. **Spatial Correlation**:
   ```
   sigma = grain_size / 10.0
   grain_pattern = GaussianFilter(noise, sigma)
   ```

3. **Luminance Analysis**:
   ```
   luminance = 0.299*R + 0.587*G + 0.114*B
   shadow_boost = 1.0 + (1.0 - luminance) * 0.5
   ```

4. **ISO Scaling**:
   ```
   iso_factor = log10(iso / 100)
   ```

5. **Final Grain Calculation**:
   ```
   final_grain = grain_pattern * iso_factor * strength * shadow_boost
   ```

6. **Application to Image**:
   - Monochrome: Same grain to all channels
   - Colored: Independent grain per channel

## Mathematical Properties

### Linearity
- Strength parameter provides linear control
- Combined effects are multiplicative

### Logarithmic Response
- ISO parameter uses logarithmic scaling
- Mimics real film response characteristics

### Spatial Correlation
- Gaussian filtering creates realistic grain clumping
- Prevents the "digital noise" look of uncorrelated pixels

### Channel Independence
- Colored mode uses independent noise patterns
- Creates realistic chromatic variation in color film grain

This mathematical approach creates film grain that closely resembles the characteristics of real photographic film, with proper scaling, spatial correlation, and luminance-dependent visibility.