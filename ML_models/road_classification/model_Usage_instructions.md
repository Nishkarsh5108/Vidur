# Road Quality & Pothole Classification: Mobile App Developer Guide

This document provides complete, production-grade instructions for integrating the Physics-Informed Road Classification TensorFlow Lite (`.tflite`) models into iOS, Android, Flutter, or React Native mobile applications.

---

## 1. Model Selection: Float16 vs. Float32

Two TFLite models are provided in `ml_model/road_classification/`. Choose the appropriate version based on your target device capabilities and app size budget:

| Model File | Size | Precision | Target Hardware | Why Choose This? |
| :--- | :--- | :--- | :--- | :--- |
| `road_vision_final_float16.tflite` | **~226 KB** | FP16 (Half) | **Recommended Default**<br>(Mobile GPUs, Apple Neural Engine, Android NNAPI / NPU) | 50% smaller binary size, significantly faster execution on hardware accelerators, minimal accuracy drop (<0.1%). |
| `road_vision_final_float32.tflite` | **~446 KB** | FP32 (Full) | Older CPU-Only Devices | Absolute reference accuracy. Use as a fallback if the device CPU does not support FP16 acceleration. |

---

## 2. Telemetry Requirements: What Sensor Readings to Collect

To classify road quality accurately, the mobile app must continuously sample two native device sensors:

### A. Accelerometer (Vertical Z-Axis Shocks)
* **What to collect**: Linear acceleration along the axis perpendicular to the road surface ($a_z$), measured in $m/s^2$.
* **Sampling Rate**: Target **50 Hz to 100 Hz**.
* **Window Size**: The model requires exactly **`128 consecutive samples`** per inference pass (representing ~1.5 to 2.5 seconds of driving depending on sample rate).
* **Gravity & Orientation Calibration**:
  * The mobile phone must be rigidly mounted inside the vehicle (e.g., windshield dashcam cradle or phone mount).
  * Use the device's **Sensor Fusion / Rotation Vector** to project the raw 3D accelerometer vector $(x, y, z)$ into Earth-frame vertical acceleration, isolating true road shocks from phone tilt.
  * Subtract static Earth gravity ($9.81 m/s^2$) so that a stationary car reads $a_z \approx 0.0 m/s^2$.

### B. GPS / Location Services (Vehicle Speed)
* **What to collect**: Ground speed in **meters per second ($m/s$)**.
* **Window Aggregation**: Compute the **mean speed** over the exact duration of the 128-sample accelerometer window. If GPS is temporarily unavailable, fallback to the last known valid speed.

---

## 3. TFLite Input & Output Signature

Unlike standard image models, this architecture uses a **Dual-Input Multi-Modal Signature**. You must pass two separate input tensors to the TFLite interpreter and extract one output tensor:

### Input Tensor 0: `vibration` (Waveform Features)
* **Shape**: `[1, 2, 128]` (Batch $\times$ Channels $\times$ Time Samples)
* **Data Type**: `Float32` (1024 bytes / 256 floats total)
* **Channel Description**:
  * **Channel 0**: Rectified Magnitude ($|a_z|$ normalized differential acceleration).
  * **Channel 1**: Jerk / Sharpness (first gradient of the normalized differential acceleration).

### Input Tensor 1: `context` (Physical ECE Metadata)
* **Shape**: `[1, 4]` (Batch $\times$ Features)
* **Data Type**: `Float32` (16 bytes / 4 floats total)
* **Feature Order**: `[VehID, Speed_ms, RMS, Crest_Factor]` (Z-score standardized).

### Output Tensor 0: `road_grade` (Classification Logits)
* **Shape**: `[1, 4]`
* **Data Type**: `Float32` (16 bytes / 4 floats total)
* **Class Mapping**:
  * `0`: **Excellent** (Smooth pavement)
  * `1`: **Patches** (Minor surface roughness / repaired asphalt)
  * `2`: **Med Pothole** (Medium pothole anomaly)
  * `3`: **Big Pothole** (Severe hazard / deep pothole)

---

## 4. Step-by-Step Mathematical Preprocessing Pipeline

You cannot pass raw accelerometer readings directly into the TFLite interpreter. You must implement the following mathematical transformations in your native app code (Kotlin, Swift, Dart, etc.).

### Step 1: Compute Differential Acceleration (Jerk Approximation)
To isolate sharp road impacts from slow vehicle body rolls (accelerating/braking), compute the 1st-order difference of the 128-sample vertical acceleration buffer $A = [a_0, a_1, \dots, a_{127}]$:
```math
S[i] = \begin{cases} 
0 & \text{if } i = 0 \\
a_i - a_{i-1} & \text{if } i > 0 
\end{cases}
```

### Step 2: Robust Z-Score Standardization of the Waveform
Normalize the differential signal $S$ using its sample mean $\mu_S$ and standard deviation $\sigma_S$:
```math
\mu_S = \frac{1}{128} \sum_{i=0}^{127} S[i], \quad \sigma_S = \sqrt{\frac{1}{128} \sum_{i=0}^{127} (S[i] - \mu_S)^2} + 10^{-6}
```
```math
S_{\text{norm}}[i] = \frac{S[i] - \mu_S}{\sigma_S}
```

### Step 3: Populate the 2-Channel Vibration Input Buffer (`[1, 2, 128]`)
Allocate a flat `Float32` buffer of 256 elements ($2 \times 128$):
1. **Channel 0 (Indices `0` to `127`) — Rectified Magnitude**:
   ```math
   C_0[i] = |S_{\text{norm}}[i]|
   ```
2. **Channel 1 (Indices `128` to `255`) — Numerical Gradient (Central Difference)**:
   ```math
   C_1[i] = \begin{cases} 
   S_{\text{norm}}[1] - S_{\text{norm}}[0] & \text{if } i = 0 \\
   \frac{S_{\text{norm}}[i+1] - S_{\text{norm}}[i-1]}{2} & \text{if } 0 < i < 127 \\
   S_{\text{norm}}[127] - S_{\text{norm}}[126] & \text{if } i = 127 
   \end{cases}
   ```

### Step 4: Compute Physical ECE Features & Scale Context (`[1, 4]`)
Calculate the physical vibration energy metrics from $S_{\text{norm}}$:
```math
\text{RMS} = \sqrt{\frac{1}{128} \sum_{i=0}^{127} (S_{\text{norm}}[i])^2} + 10^{-6}, \quad \text{Crest Factor} = \frac{\max(|S_{\text{norm}}|)}{\text{RMS}}
```

Assemble the raw 4-element vector $V_{\text{raw}}$:
* `VehID`: Set to `0.0` (default/universal vehicle calibration).
* `Speed_ms`: Average GPS speed in $m/s$.
* `RMS`: Calculated RMS value above.
* `Crest_Factor`: Calculated Crest Factor above.

> [!IMPORTANT]
> **Context Standardization Without Python**: Neural network dense layers require standardized features. Because mobile apps cannot load scikit-learn `.pkl` files natively, extract the exact normalization constants from `context_scaler.pkl` once during your build using this 3-line Python script:
> ```python
> import joblib
> s = joblib.load("context_scaler.pkl")
> print("MEAN:", s.mean_.tolist())
> print("SCALE:", s.scale_.tolist())
> ```
> Then, hardcode those 4 `MEAN` and `SCALE` float constants in your mobile app and apply Z-score scaling:
```math
V_{\text{scaled}}[j] = \frac{V_{\text{raw}}[j] - \text{MEAN}[j]}{\text{SCALE}[j]} \quad \text{for } j \in \{0, 1, 2, 3\}
```

---

## 5. Post-Processing & Safety Thresholding

Once the TFLite interpreter executes, you will receive 4 logit values $[z_0, z_1, z_2, z_3]$. Follow these steps to generate user-facing road alerts:

### 1. Softmax Probability Conversion
Convert raw logits into class probabilities $P[c] \in [0, 1]$:
```math
P[c] = \frac{e^{z_c}}{\sum_{k=0}^{3} e^{z_k}}
```
Let $\text{PredClass} = \arg\max_c(P[c])$ and $\text{Confidence} = \max(P)$.

### 2. Safety Confidence Gating (False Alarm Prevention)
To prevent annoying false alarms when driving over minor road joints or bridge expansion grates, enforce the model's accuracy threshold:
* If the model predicts a hazard ($\text{PredClass} \ge 2$: **Med Pothole** or **Big Pothole**) **BUT** the confidence score is below **`0.82` (82%)**, you **must override/downgrade** the prediction back to **`0` (Excellent)**:
```python
if (predClass >= 2 and confidence < 0.82) {
    predClass = 0 // Override low-confidence hazard to smooth road
}
```

### 3. Temporal Smoothing (1.5s Majority Vote)
To prevent UI flickering during live driving, maintain a sliding FIFO queue of the last **15 window predictions** (~1.5 seconds of driving history). Display the **majority vote** class across that window to the user or telemetry backend.

---

## 6. Native Code Implementation Examples

### A. Android (Kotlin) Implementation
```kotlin
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class RoadClassifier(tfliteModelFile: File) {
    private val interpreter = Interpreter(tfliteModelFile)
    
    // Hardcoded constants from context_scaler.pkl
    private val SCALER_MEAN = floatArrayOf(0.0f, 12.5f, 0.98f, 3.45f) // Replace with exact scaler outputs
    private val SCALER_SCALE = floatArrayOf(1.0f, 4.2f, 0.31f, 1.12f)

    fun classify(rawAz: FloatArray, speedMs: Float): Int {
        require(rawAz.size == 128) { "Accelerometer window must be exactly 128 samples." }
        
        // 1. Differential Acceleration (Jerk)
        val sig = FloatArray(128)
        sig[0] = 0.0f
        for (i in 1 until 128) {
            sig[i] = rawAz[i] - rawAz[i - 1]
        }
        
        // 2. Z-Score Normalization
        val mean = sig.average().toFloat()
        var sumSq = 0.0f
        for (v in sig) sumSq += (v - mean) * (v - mean)
        val std = sqrt(sumSq / 128.0f) + 1e-6f
        
        val sigNorm = FloatArray(128)
        for (i in 0 until 128) sigNorm[i] = (sig[i] - mean) / std
        
        // 3. Populate Vibration Buffer [1, 2, 128]
        val vibBuffer = ByteBuffer.allocateDirect(1 * 2 * 128 * 4).order(ByteOrder.nativeOrder())
        // Channel 0: Rectified Magnitude
        for (i in 0 until 128) vibBuffer.putFloat(abs(sigNorm[i]))
        // Channel 1: Gradient
        vibBuffer.putFloat(sigNorm[1] - sigNorm[0])
        for (i in 1 until 127) vibBuffer.putFloat((sigNorm[i + 1] - sigNorm[i - 1]) / 2.0f)
        vibBuffer.putFloat(sigNorm[127] - sigNorm[126])
        
        // 4. Physical Features & Context Buffer [1, 4]
        var rmsSq = 0.0f
        var maxAbs = 0.0f
        for (v in sigNorm) {
            rmsSq += v * v
            if (abs(v) > maxAbs) maxAbs = abs(v)
        }
        val rms = sqrt(rmsSq / 128.0f) + 1e-6f
        val crestFactor = maxAbs / rms
        
        val rawContext = floatArrayOf(0.0f, speedMs, rms, crestFactor)
        val ctxBuffer = ByteBuffer.allocateDirect(1 * 4 * 4).order(ByteOrder.nativeOrder())
        for (j in 0 until 4) {
            ctxBuffer.putFloat((rawContext[j] - SCALER_MEAN[j]) / SCALER_SCALE[j])
        }
        
        // 5. Run Multi-Input Inference
        val inputs = arrayOf<Any>(vibBuffer, ctxBuffer)
        val outputLogits = Array(1) { FloatArray(4) }
        val outputs = mutableMapOf<Int, Any>(0 to outputLogits)
        
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        
        // 6. Post-Processing & Softmax
        val logits = outputLogits[0]
        val maxLogit = logits.maxOrNull() ?: 0.0f
        var expSum = 0.0f
        val probs = FloatArray(4)
        for (i in 0 until 4) {
            probs[i] = exp(logits[i] - maxLogit)
            expSum += probs[i]
        }
        for (i in 0 until 4) probs[i] /= expSum
        
        var predClass = 0
        var maxProb = probs[0]
        for (i in 1 until 4) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                predClass = i
            }
        }
        
        // 7. Safety Thresholding
        if (predClass >= 2 && maxProb < 0.82f) {
            predClass = 0 // Downgrade low-confidence pothole alarm
        }
        
        return predClass
    }
}
```

### B. iOS (Swift) Implementation
```swift
import TensorFlowLite
import Foundation

class RoadClassifier {
    private var interpreter: Interpreter
    
    // Hardcoded constants from context_scaler.pkl
    private let scalerMean: [Float] = [0.0, 12.5, 0.98, 3.45] // Replace with exact scaler outputs
    private let scalerScale: [Float] = [1.0, 4.2, 0.31, 1.12]
    
    init(modelPath: String) throws {
        var options = Interpreter.Options()
        options.threadCount = 2
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        try interpreter.allocateTensors()
    }
    
    func classify(rawAz: [Float], speedMs: Float) throws -> Int {
        guard rawAz.count == 128 else { fatalError("Need exactly 128 samples") }
        
        // 1. Differential Acceleration (Jerk)
        var sig = [Float](repeating: 0.0, count: 128)
        for i in 1..<128 { sig[i] = rawAz[i] - rawAz[i - 1] }
        
        // 2. Z-Score Normalization
        let mean = sig.reduce(0, +) / 128.0
        let sumSq = sig.map { pow($0 - mean, 2) }.reduce(0, +)
        let std = sqrt(sumSq / 128.0) + 1e-6
        let sigNorm = sig.map { ($0 - mean) / std }
        
        // 3. Build Vibration Buffer [1, 2, 128]
        var vibData = [Float](repeating: 0.0, count: 256)
        for i in 0..<128 { vibData[i] = abs(sigNorm[i]) }
        vibData[128] = sigNorm[1] - sigNorm[0]
        for i in 1..<127 { vibData[128 + i] = (sigNorm[i + 1] - sigNorm[i - 1]) / 2.0 }
        vibData[255] = sigNorm[127] - sigNorm[126]
        
        // 4. Build Context Buffer [1, 4]
        let rms = sqrt(sigNorm.map { $0 * $0 }.reduce(0, +) / 128.0) + 1e-6
        let maxAbs = sigNorm.map { abs($0) }.max() ?? 0.0
        let crestFactor = maxAbs / rms
        
        let rawContext: [Float] = [0.0, speedMs, rms, crestFactor]
        var scaledContext = [Float](repeating: 0.0, count: 4)
        for j in 0..<4 {
            scaledContext[j] = (rawContext[j] - scalerMean[j]) / scalerScale[j]
        }
        
        // 5. Copy Data & Invoke Interpreter
        let vibBytes = Data(bytes: vibData, count: vibData.count * MemoryLayout<Float>.stride)
        let ctxBytes = Data(bytes: scaledContext, count: scaledContext.count * MemoryLayout<Float>.stride)
        
        try interpreter.copy(vibBytes, toInputAt: 0)
        try interpreter.copy(ctxBytes, toInputAt: 1)
        try interpreter.invoke()
        
        // 6. Read Output & Softmax
        let outputTensor = try interpreter.output(at: 0)
        let logits = outputTensor.data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        
        let maxLogit = logits.max() ?? 0.0
        let exps = logits.map { exp($0 - maxLogit) }
        let expSum = exps.reduce(0, +)
        let probs = exps.map { $0 / expSum }
        
        var predClass = 0
        var maxProb = probs[0]
        for i in 1..<4 {
            if probs[i] > maxProb {
                maxProb = probs[i]
                predClass = i
            }
        }
        
        // 7. Safety Thresholding
        if predClass >= 2 && maxProb < 0.82 {
            predClass = 0
        }
        
        return predClass
    }
}
```
