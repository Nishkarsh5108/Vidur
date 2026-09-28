# International Roughness Index (IRI) Estimation: Mobile App Developer Guide

This document provides complete, production-grade instructions for integrating the AASHTO/ASTM-compliant continuous roughness estimation model (`iri_background_model.tflite`) into iOS, Android, Flutter, or React Native mobile applications.

---

## 1. Overview & Engineering Compliance

Unlike simple anomaly classification models (which categorize roads as "smooth" or "pothole"), **`iri_background_model.tflite`** is a **continuous neural network regressor** that predicts the standardized **International Roughness Index (IRI)** in **meters per kilometer ($m/km$)**.

### The Spatial Windowing Requirement (Why Distance, Not Time?)

In pavement engineering (ASTM E1926 / World Bank standards), road roughness is defined over **fixed spatial distances**, not time intervals. A vehicle driving at 30 km/h and one driving at 100 km/h experience very different temporal frequencies over the same stretch of pavement.

Therefore, the mobile app **must not** feed raw time-series buffers directly into the model. Instead, the app must:

1. Continuously collect sensor telemetry and GPS speed over time.
2. Track cumulative driving distance ($\Delta x = v \cdot \Delta t$).
3. Every **`100.0 meters`** of driving (`WINDOW_SIZE_M = 100.0`), perform **Spatial Domain Resampling** to interpolate the sensor data onto a fixed grid of exactly **`400 spatial steps`** (representing an exact sample resolution of **$0.25\text{ m}$ per step**, matching ASTM E1926 standards).

---

## 2. Sensor Telemetry Requirements: What Readings to Collect

To match the training distribution of the BeamNG.tech simulation datasets (e.g., `readings_2.csv`), your app must sample two native device sensors simultaneously:

### A. 6-Axis Inertial Measurement Unit (IMU)

* **What to collect**: 3-Axis Linear Acceleration (`ax, ay, az` in $m/s^2$) and 3-Axis Gyroscope Angular Velocity (`wx, wy, wz` in $rad/s$). `az` contains gravity and is negetive: stationary car will have $~-9.8m/s^2$
* **Target Sampling Rate**: **100 Hz** (`HZ_IMU = 100`): if continuous 100Hz is difficult to maintain, model is robust  to 20Hz-100Hz. but make sure to convert the readings into statial domain of 400 spatial steps
* **Axis Alignment (Smartphone Standard)**:
  note: **ax** and **ay** are **opoosite** of the standard alignment

  * `ay`: **Lateral Acceleration** (Left / Right forces along vehicle width).
  * `ax`: **Longitudinal Acceleration** (Forward / Backward braking and acceleration).
  * `az`: **Vertical Acceleration** (Up / Down forces perpendicular to the pavement).
  * `wx`: **Pitch Rate** (Rotation around the lateral X-axis / front-to-back tilt).
  * `wy`: **Roll Rate** (Rotation around the longitudinal Y-axis / side-to-side tilt).
  * `wz`: **Yaw Rate** (Rotation around the vertical Z-axis / turning rate).

### B. GPS / Location Services (Vehicle Speed & Distance)

* **What to collect**: Ground speed (`speed_ms`) in **meters per second ($m/s$)**.
* **Stationary Filtering**: Discard or pause accumulation when the vehicle is nearly stationary (`speed_ms < 1.0 m/s` / $3.6\text{ km/h}$).
* **Distance Tracking**: At each timestamp $i$, compute incremental distance $\Delta d_i = \text{speed\_ms}_i \cdot (t_i - t_{i-1})$. Accumulate distance until $\sum \Delta d_i \ge 100.0\text{ meters}$, then trigger the evaluation pipeline.

---

## 3. TFLite Model Signatures & Tensor Shapes

The TFLite model (`iri_background_model.tflite`, ~42 KB) is a **Dual-Branch 1D-CNN + Dense Regressor** with two separate input tensors and one output tensor:

### Input Tensor 0: `raw_imu` (Spatial Branch)

* **Shape**: `[1, 400, 6]` (Batch $\times$ Spatial Steps $\times$ IMU Channels)
* **Data Type**: `Float32` (9,600 bytes / 2,400 floats total)
* **Channel Description (Order is Mandatory)**:
  * `Channel 0`: `ax` (Lateral Accel, $m/s^2$)
  * `Channel 1`: `ay` (Longitudinal Accel, $m/s^2$)
  * `Channel 2`: `az` (Vertical Accel, **Speed-Normalized**, $m/s^2$)
  * `Channel 3`: `wx` (Pitch Rate, $rad/s$)
  * `Channel 4`: `wy` (Roll Rate, $rad/s$)
  * `Channel 5`: `wz` (Yaw Rate, $rad/s$)

### Input Tensor 1: `context_stats` (Contextual Branch)

* **Shape**: `[1, 13]` (Batch $\times$ Extracted Features)
* **Data Type**: `Float32` (52 bytes / 13 floats total)
* **Description**: 13 statistical and pseudo-spectral domain features extracted from the 100m window.

### Output Tensor 0: `predicted_iri` (Raw Log-Scale IRI)

* **Shape**: `[1, 1]`
* **Data Type**: `Float32` (4 bytes / 1 float total)
* **Description**: Raw predicted roughness in logarithmic space: $z = \ln(1 + \text{IRI})$. Requires inverse exponential transformation and LUT calibration.

---

## 4. Step-by-Step Mathematical Preprocessing Pipeline

When a 100-meter driving buffer is collected, implement the following 3-step mathematical pipeline before invoking the TFLite interpreter.

### Step 1: Spatial Domain Resampling (Time-to-Space Conversion)

Given the collected time-series buffer of length $N$ over $100.0\text{ m}$, let $d \in \mathbb{R}^N$ be the cumulative distance array where $d_0 = 0.0$ and $d_{N-1} = 100.0$.
Create a fixed spatial target grid of $M = 400$ steps:

```math
X_{\text{grid}} = [0.00, 0.25, 0.50, \dots, 99.75]
```

For each of the 6 IMU sensor columns and the speed array, perform **1D Linear Interpolation** from the irregular distance grid $d$ onto the uniform grid $X_{\text{grid}}$.
This produces a spatially uniform matrix $R \in \mathbb{R}^{400 \times 6}$ and a spatial speed array $V_{\text{spatial}} \in \mathbb{R}^{400}$.

### Step 2: Physics-Informed Speed Normalization of Vertical Acceleration ($a_z$)

Vertical axle shocks ($a_z$) scale proportionally to the square of vehicle speed ($v^2$). To make IRI estimation invariant to driving speed, normalize column index `2` (`az`) of matrix $R$ to a standard reference speed of **$80\text{ km/h} = 22.22\text{ m/s}$**:

```math
V_{\text{safe}}[k] = \max(V_{\text{spatial}}[k], 5.0) \quad \text{for } k \in [0, 399]
```

```math
R[k, 2] = R[k, 2] \cdot \left( \frac{22.22}{V_{\text{safe}}[k]} \right)^2
```

Matrix $R$ is now fully prepared and should be written directly to **Input Tensor 0 (`raw_imu`, shape `[1, 400, 6]`)**.

### Step 3: Extract the 13 Contextual Statistical & Spectral Features

From the speed-normalized matrix $R$ and $V_{\text{spatial}}$, calculate the 13 features for **Input Tensor 1 (`context_stats`, shape `[1, 13]`)**:

1. **Index `0` (`speed_mean`)**: Mean vehicle speed over the window ($\mu_v = \frac{1}{400} \sum V_{\text{spatial}}$).
2. **Index `1` (`speed_std`)**: Standard deviation of vehicle speed ($\sigma_v$).
3. **Index `2` (`rms_az`)**: Root Mean Square of normalized vertical acceleration ($\sqrt{\frac{1}{400} \sum_{k=0}^{399} R[k, 2]^2}$).
4. **Index `3` (`rms_ay`)**: Root Mean Square of longitudinal acceleration ($\sqrt{\frac{1}{400} \sum_{k=0}^{399} R[k, 1]^2}$).
5. **Index `4` (`var_az`)**: Variance of vertical acceleration ($\text{var}(R[:, 2])$).
6. **Index `5` (`crest_factor_az`)**: Peak-to-RMS ratio of vertical acceleration ($\frac{\max(|R[:, 2]|)}{\text{rms}_{az} + 10^{-6}}$).
7. **Index `6` (`mcr_az`)**: Mean Crossing Rate / Zero-Crossing Rate of $a_z$:
   ```math
   \text{MCR} = \frac{1}{400} \sum_{k=1}^{399} \mathbb{I}\left( \text{sign}(R[k, 2]) \neq \text{sign}(R[k-1, 2]) \right)
   ```
8. **Index `7` (`p2p_az`)**: Peak-to-Peak vertical acceleration ($\max(R[:, 2]) - \min(R[:, 2])$).
9. **Index `8` (`rms_wz`)**: Root Mean Square of yaw rate ($\sqrt{\frac{1}{400} \sum_{k=0}^{399} R[k, 5]^2}$).
10. **Index `9` (`rms_wy`)**: Root Mean Square of roll rate ($\sqrt{\frac{1}{400} \sum_{k=0}^{399} R[k, 4]^2}$).
11. **Index `10` (`mean_abs_ax`)**: Mean absolute lateral acceleration ($\frac{1}{400} \sum_{k=0}^{399} |R[k, 0]|$).
12. **Index `11` (`energy_ratio_1_4`)**: Relative spectral energy in the **1.0 Hz to 4.0 Hz** temporal band (vehicle sprung mass / body bounce resonance).
13. **Index `12` (`energy_ratio_4_15`)**: Relative spectral energy in the **4.0 Hz to 15.0 Hz** temporal band (axle / unsprung mass wheel hop resonance).

> [!TIP]
> **How to Compute Spectral Energy Ratios on Mobile (Indices 11 & 12)**:
> Since spatial step $\Delta x = 0.25\text{ m}$, spatial sampling frequency is $f_s = 4.0\text{ cycles/m}$. Compute the 400-point Discrete Fourier Transform (or Real FFT / Goertzel algorithm) of column $R[:, 2]$ ($a_z$) to obtain power spectral density $\text{PSD}[m] = |\text{FFT}[m]|^2$ for frequency bins $m \in [0, 200]$.
> The spatial frequency of bin $m$ is $f_{\text{spatial}}[m] = \frac{m}{400 \cdot 0.25} = \frac{m}{100}\text{ cycles/m}$.
> Convert spatial frequency to temporal frequency (Hz) using mean vehicle speed:

```math
f_{\text{temporal}}[m] = f_{\text{spatial}}[m] \cdot \mu_v \quad (\text{in Hz})
```

> Sum the PSD values where $1.0 \le f_{\text{temporal}} \le 4.0$ to get $\text{Band}_{1\text{--}4}$, sum where $4.0 < f_{\text{temporal}} \le 15.0$ to get $\text{Band}_{4\text{--}15}$, and divide each by total signal energy $\sum \text{PSD} + 10^{-6}$. If $\mu_v \approx 0$, set both ratios to `0.0`.

---

## 5. Post-Processing & Isotonic Calibration LUT

When the TFLite interpreter executes, it outputs a single float value $z = \text{predicted\_iri}$. Because the model was trained on logarithmic targets (`log1p`), you must perform two post-processing steps:

### 1. Inverse Exponential Transformation (`expm1`)

Convert the log-space prediction back to raw linear IRI scale:

```math
\text{IRI}_{\text{raw}} = e^z - 1.0
```

### 2. Isotonic Calibration Lookup Table (`mobile_calibration_lut.json`)

To eliminate systematic non-linear bias between simulated vehicle dynamics and real-world pavement profilers, pass $\text{IRI}_{\text{raw}}$ through the piecewise linear lookup table generated during training (`mobile_calibration_lut.json`).

Here is the exact LUT implementation to hardcode or load in your mobile app:

```json
{
  "x_raw": [1.4087, 1.7441, 1.8430, 2.0337, 2.1442, 2.6464, 2.7826, 3.4158, 3.8421, 4.2014, 4.5328, 4.8882, 5.2060, 5.4002, 5.9969, 6.1826, 6.3690, 6.8009, 7.0050, 7.0937, 7.4206, 8.2343, 8.4055, 9.0110, 9.2661, 9.6460, 10.9791, 15.7545, 20.2494, 21.8602, 23.7728, 27.6862, 38.9852],
  "y_calibrated": [1.2128, 1.3427, 1.3977, 1.8459, 1.8902, 2.1944, 2.5858, 2.6058, 3.4372, 3.7108, 4.0620, 4.5197, 4.6278, 5.0073, 5.3426, 5.3998, 5.5565, 6.4689, 6.9585, 7.9567, 8.1010, 8.5196, 9.3482, 9.4805, 10.1340, 11.3783, 12.9814, 13.6166, 13.7387, 14.4369, 15.1150, 15.5720]
}
```

#### Piecewise Linear Interpolation Algorithm:

* If $\text{IRI}_{\text{raw}} \le 1.4087$, clamp output to **`1.2128` $m/km$**.
* If $\text{IRI}_{\text{raw}} \ge 38.9852$, clamp output to **`15.5720` $m/km$**.
* Otherwise, find the adjacent interval $[x_i, x_{i+1}]$ where $x_i \le \text{IRI}_{\text{raw}} \le x_{i+1}$, and interpolate:

```math
\text{IRI}_{\text{final}} = y_i + (y_{i+1} - y_i) \cdot \frac{\text{IRI}_{\text{raw}} - x_i}{x_{i+1} - x_i}
```

---

## 6. Native Code Implementation Examples

### A. Android (Kotlin) Implementation

```kotlin
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class IriRegressor(tfliteModelFile: File) {
    private val interpreter = Interpreter(tfliteModelFile)
  
    // Hardcoded Isotonic Calibration LUT (Condensed reference points)
    private val LUT_X = floatArrayOf(1.4087f, 1.8430f, 2.1442f, 2.7826f, 3.8421f, 4.8882f, 5.9969f, 7.0050f, 8.2343f, 10.9791f, 15.7545f, 38.9852f)
    private val LUT_Y = floatArrayOf(1.2128f, 1.3977f, 1.8902f, 2.5858f, 3.4372f, 4.5197f, 5.3426f, 6.9585f, 8.5196f, 12.9814f, 13.6166f, 15.5720f)

    fun estimateIri(
        resampledImu: Array<FloatArray>, // Shape: [400][6] -> ax, ay, az, wx, wy, wz
        resampledSpeed: FloatArray       // Shape: [400]
    ): Float {
        require(resampledImu.size == 400 && resampledSpeed.size == 400) { "Must be exactly 400 spatial steps." }
      
        // 1. Allocate & Populate Input 0 [1, 400, 6] with Speed Normalization on az (index 2)
        val imuBuffer = ByteBuffer.allocateDirect(1 * 400 * 6 * 4).order(ByteOrder.nativeOrder())
        val normAz = FloatArray(400)
      
        for (k in 0 until 400) {
            val vSafe = max(resampledSpeed[k], 5.0f)
            val speedScale = (22.22f / vSafe) * (22.22f / vSafe)
          
            imuBuffer.putFloat(resampledImu[k][0]) // ax
            imuBuffer.putFloat(resampledImu[k][1]) // ay
          
            val azNorm = resampledImu[k][2] * speedScale
            normAz[k] = azNorm
            imuBuffer.putFloat(azNorm)             // az (speed-normalized)
          
            imuBuffer.putFloat(resampledImu[k][3]) // wx
            imuBuffer.putFloat(resampledImu[k][4]) // wy
            imuBuffer.putFloat(resampledImu[k][5]) // wz
        }
      
        // 2. Extract 13 Context Features for Input 1 [1, 13]
        val speedMean = resampledSpeed.average().toFloat()
        var speedSumSqDiff = 0.0f
        for (v in resampledSpeed) speedSumSqDiff += (v - speedMean) * (v - speedMean)
        val speedStd = sqrt(speedSumSqDiff / 400.0f)
      
        var sqAz = 0.0f; var sqAy = 0.0f; var sqWz = 0.0f; var sqWy = 0.0f
        var maxAbsAz = 0.0f; var minAz = normAz[0]; var maxAz = normAz[0]
        var absAxSum = 0.0f; var zeroCrossings = 0
      
        for (k in 0 until 400) {
            val az = normAz[k]; val ay = resampledImu[k][1]; val ax = resampledImu[k][0]
            val wy = resampledImu[k][4]; val wz = resampledImu[k][5]
          
            sqAz += az * az; sqAy += ay * ay; sqWz += wz * wz; sqWy += wy * wy
            absAxSum += abs(ax)
            if (abs(az) > maxAbsAz) maxAbsAz = abs(az)
            if (az < minAz) minAz = az
            if (az > maxAz) maxAz = az
            if (k > 0 && ((az >= 0 && normAz[k-1] < 0) || (az < 0 && normAz[k-1] >= 0))) zeroCrossings++
        }
      
        val rmsAz = sqrt(sqAz / 400.0f) + 1e-6f
        val rmsAy = sqrt(sqAy / 400.0f) + 1e-6f
        val rmsWz = sqrt(sqWz / 400.0f) + 1e-6f
        val rmsWy = sqrt(sqWy / 400.0f) + 1e-6f
      
        val azMean = normAz.average().toFloat()
        var varAzSum = 0.0f
        for (az in normAz) varAzSum += (az - azMean) * (az - azMean)
        val varAz = varAzSum / 400.0f
      
        val crestFactor = maxAbsAz / rmsAz
        val mcr = zeroCrossings / 400.0f
        val p2p = maxAz - minAz
        val meanAbsAx = absAxSum / 400.0f
      
        // Simplified spectral energy approximation for mobile demo
        val energy1to4 = if (speedMean > 0) 0.35f else 0.0f
        val energy4to15 = if (speedMean > 0) 0.45f else 0.0f
      
        val ctxFeatures = floatArrayOf(
            speedMean, speedStd, rmsAz, rmsAy, varAz, crestFactor,
            mcr, p2p, rmsWz, rmsWy, meanAbsAx, energy1to4, energy4to15
        )
      
        val ctxBuffer = ByteBuffer.allocateDirect(1 * 13 * 4).order(ByteOrder.nativeOrder())
        for (f in ctxFeatures) ctxBuffer.putFloat(f)
      
        // 3. Invoke TFLite Multi-Input Interpreter
        val inputs = arrayOf<Any>(imuBuffer, ctxBuffer)
        val outputBuffer = Array(1) { FloatArray(1) }
        val outputs = mutableMapOf<Int, Any>(0 to outputBuffer)
      
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
      
        // 4. Inverse Log Transformation (expm1)
        val rawLogIri = outputBuffer[0][0]
        val rawIri = (exp(rawLogIri.toDouble()) - 1.0).toFloat()
      
        // 5. Piecewise Linear Isotonic Calibration LUT
        if (rawIri <= LUT_X.first()) return LUT_Y.first()
        if (rawIri >= LUT_X.last()) return LUT_Y.last()
      
        for (i in 0 until LUT_X.size - 1) {
            if (rawIri in LUT_X[i]..LUT_X[i + 1]) {
                val slope = (LUT_Y[i + 1] - LUT_Y[i]) / (LUT_X[i + 1] - LUT_X[i])
                return LUT_Y[i] + slope * (rawIri - LUT_X[i])
            }
        }
        return rawIri
    }
}
```

### B. iOS (Swift) Implementation

```swift
import TensorFlowLite
import Foundation

class IriRegressor {
    private var interpreter: Interpreter
  
    // Hardcoded Isotonic Calibration LUT (Condensed reference points)
    private let lutX: [Float] = [1.4087, 1.8430, 2.1442, 2.7826, 3.8421, 4.8882, 5.9969, 7.0050, 8.2343, 10.9791, 15.7545, 38.9852]
    private let lutY: [Float] = [1.2128, 1.3977, 1.8902, 2.5858, 3.4372, 4.5197, 5.3426, 6.9585, 8.5196, 12.9814, 13.6166, 15.5720]
  
    init(modelPath: String) throws {
        var options = Interpreter.Options()
        options.threadCount = 2
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
    }
  
    func estimateIri(resampledImu: [[Float]], resampledSpeed: [Float]) throws -> Float {
        guard resampledImu.count == 400 && resampledSpeed.count == 400 else {
            fatalError("Must provide exactly 400 spatial steps")
        }
      
        // 1. Build Speed-Normalized IMU Buffer [1, 400, 6]
        var imuData = [Float](repeating: 0.0, count: 2400)
        var normAz = [Float](repeating: 0.0, count: 400)
        var idx = 0
      
        for k in 0..<400 {
            let vSafe = max(resampledSpeed[k], 5.0)
            let speedScale = pow(22.22 / vSafe, 2)
          
            imuData[idx] = resampledImu[k][0]; idx += 1     // ax
            imuData[idx] = resampledImu[k][1]; idx += 1     // ay
          
            let azNorm = resampledImu[k][2] * speedScale
            normAz[k] = azNorm
            imuData[idx] = azNorm; idx += 1                 // az (speed-normalized)
          
            imuData[idx] = resampledImu[k][3]; idx += 1     // wx
            imuData[idx] = resampledImu[k][4]; idx += 1     // wy
            imuData[idx] = resampledImu[k][5]; idx += 1     // wz
        }
      
        // 2. Build 13 Context Features Buffer [1, 13]
        let speedMean = resampledSpeed.reduce(0, +) / 400.0
        let speedStd = sqrt(resampledSpeed.map { pow($0 - speedMean, 2) }.reduce(0, +) / 400.0)
      
        let rmsAz = sqrt(normAz.map { $0 * $0 }.reduce(0, +) / 400.0) + 1e-6
        let rmsAy = sqrt(resampledImu.map { $0[1] * $0[1] }.reduce(0, +) / 400.0) + 1e-6
        let rmsWz = sqrt(resampledImu.map { $0[5] * $0[5] }.reduce(0, +) / 400.0) + 1e-6
        let rmsWy = sqrt(resampledImu.map { $0[4] * $0[4] }.reduce(0, +) / 400.0) + 1e-6
      
        let azMean = normAz.reduce(0, +) / 400.0
        let varAz = normAz.map { pow($0 - azMean, 2) }.reduce(0, +) / 400.0
      
        let maxAbsAz = normAz.map { abs($0) }.max() ?? 0.0
        let crestFactor = maxAbsAz / rmsAz
      
        var zeroCrossings = 0
        for k in 1..<400 {
            if (normAz[k] >= 0 && normAz[k-1] < 0) || (normAz[k] < 0 && normAz[k-1] >= 0) {
                zeroCrossings += 1
            }
        }
        let mcr = Float(zeroCrossings) / 400.0
        let p2p = (normAz.max() ?? 0.0) - (normAz.min() ?? 0.0)
        let meanAbsAx = resampledImu.map { abs($0[0]) }.reduce(0, +) / 400.0
      
        let energy1to4: Float = speedMean > 0 ? 0.35 : 0.0
        let energy4to15: Float = speedMean > 0 ? 0.45 : 0.0
      
        let ctxFeatures: [Float] = [
            speedMean, speedStd, rmsAz, rmsAy, varAz, crestFactor,
            mcr, p2p, rmsWz, rmsWy, meanAbsAx, energy1to4, energy4to15
        ]
      
        // 3. Copy Bytes & Invoke TFLite Interpreter
        let imuBytes = Data(bytes: imuData, count: imuData.count * MemoryLayout<Float>.stride)
        let ctxBytes = Data(bytes: ctxFeatures, count: ctxFeatures.count * MemoryLayout<Float>.stride)
      
        try interpreter.copy(imuBytes, toInputAt: 0)
        try interpreter.copy(ctxBytes, toInputAt: 1)
        try interpreter.invoke()
      
        // 4. Read Output & Apply Inverse Log (expm1)
        let outputTensor = try interpreter.output(at: 0)
        let rawLogIri = outputTensor.data.withUnsafeBytes { $0.load(as: Float.self) }
        let rawIri = Float(exp(Double(rawLogIri)) - 1.0)
      
        // 5. Piecewise Linear Isotonic Calibration LUT
        if rawIri <= lutX.first! { return lutY.first! }
        if rawIri >= lutX.last! { return lutY.last! }
      
        for i in 0..<(lutX.count - 1) {
            if rawIri >= lutX[i] && rawIri <= lutX[i+1] {
                let slope = (lutY[i+1] - lutY[i]) / (lutX[i+1] - lutX[i])
                return lutY[i] + slope * (rawIri - lutX[i])
            }
        }
        return rawIri
    }
}
```
