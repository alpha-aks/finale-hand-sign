# Indian Sign Language Detection System - Pipeline Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Complete Pipeline Flow](#complete-pipeline-flow)
3. [Dataset Creation Process](#dataset-creation-process)
4. [Hand Detection Pipeline](#hand-detection-pipeline)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Classification Pipeline](#classification-pipeline)
7. [Output Generation Pipeline](#output-generation-pipeline)
8. [New Features Pipeline](#new-features-pipeline)

---

## System Architecture

The ISL Detection System follows a modular architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (Streamlit Dashboard: Training, Testing, Typing Mode)       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                  Application Logic Layer                     │
│  (Mode Selection, Data Management, State Management)         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌────────────┐  ┌─────────────┐
│Training      │  │Testing     │  │Typing       │
│Pipeline      │  │Pipeline    │  │Pipeline     │
└──────────────┘  └────────────┘  └─────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌────────────┐  ┌─────────────┐
│Detection     │  │Preproc.    │  │Classification
│Engine        │  │Engine      │  │Engine       │
│(MediaPipe)   │  │(Normalize) │  │(CNN)        │
└──────────────┘  └────────────┘  └─────────────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌────────────┐  ┌─────────────┐
│Text Output   │  │Text-to-    │  │Visualization
│Generation    │  │Speech      │  │(OpenCV,     │
│              │  │Synthesis   │  │Streamlit)   │
└──────────────┘  └────────────┘  └─────────────┘
```

---

## Complete Pipeline Flow

### 1. Video Capture Phase

**Input**: Real-time webcam feed or video file

```
Webcam (30 FPS) 
    ↓
Frame Capture (640×480 pixels)
    ↓
BGR Image in Memory
    ↓
Mirror Flip (for user perspective)
    ↓
Ready for Processing
```

### 2. Preprocessing Phase

**Conversion to RGB format for MediaPipe**

```python
image = cv.flip(image, 1)  # Mirror effect
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
```

### 3. Hand Detection Phase

**MediaPipe HandLandmarker Core Operation**

```
RGB Image (640×480)
    ↓
[MediaPipe Neural Network]
    ├─ Hand Detection Network (Detects hand bounding box)
    ├─ Hand Landmark Network (Detects 21 points per hand)
    └─ Hand Classifier (Determines Left/Right handedness)
    ↓
Output: For Each Hand
  ├─ 21 Landmarks (x, y, z coordinates: normalized 0-1)
  ├─ 21 Confidence Scores (0-1)
  ├─ Handedness (LEFT or RIGHT)
  └─ Detection Confidence (0-1)
    ↓
Multiple Hands Processed (if num_hands=2)
```

### 4. Landmark Extraction Phase

**Convert MediaPipe output to usable format**

```
Raw Landmarks (normalized x,y,z)
    ↓
For each of 21 landmarks:
    Convert x,y from [0-1] to pixel coordinates [0-640]×[0-480]
    ↓
21 landmark points in pixel space
    ↓
Store as: [[x0,y0], [x1,y1], ..., [x20,y20]]
```

### 5. Preprocessing/Normalization Phase

**Scale and normalize for model input**

```
Raw Landmark List (21 points × 2 coordinates = 42 values)
    ↓
Step 1: Wrist Centering
  - Find wrist position (landmark 0) as origin
  - Subtract wrist coordinates from all points
  - Result: Hand is position-invariant
    ↓
Step 2: Calculate Bounding Box
  - Find max/min x,y across all landmarks
  - Result: Gesture scale information
    ↓
Step 3: Flattening
  - Convert 21 (x,y) pairs to 1D array
  - Result: [x0, y0, x1, y1, ..., x20, y20]
    ↓
Step 4: Normalization
  - Find maximum absolute value
  - Divide all values by max value
  - Result: Values in range [-1, 1]
    ↓
Final Output: 42-element normalized vector
  [0.15, -0.22, 0.31, 0.18, ..., -0.05, 0.12]
```

### 6. Classification Phase

**Neural Network Prediction**

```
Input Vector (42 values)
    ↓
[Neural Network]
    ├─ Dense Layer 1: 42 → 128 neurons (ReLU activation)
    │  - Learn high-level hand shape features
    │  - ReLU: max(0, x) - enables non-linearity
    │
    ├─ Dropout Layer 1: 20% deactivation
    │  - Prevents overfitting
    │  - Improves generalization
    │
    ├─ Dense Layer 2: 128 → 64 neurons (ReLU activation)
    │  - Refine features
    │  - Extract gesture-specific patterns
    │
    ├─ Dropout Layer 2: 20% deactivation
    │  - Further regularization
    │
    └─ Output Layer: 64 → N neurons (Softmax)
       - N = number of gesture classes
       - Softmax: Creates probability distribution
       - Sum of all outputs = 1.0
    ↓
Probability Distribution: [P(hello)=0.85, P(thumbsup)=0.10, ...]
    ↓
Predicted Class: argmax(probabilities)
    ↓
Final Output: Gesture ID (e.g., 0 for "hello")
```

### 7. Point History Phase (Complex Gestures)

**Track finger movement over time**

```
For each frame:
    ├─ If gesture detected as "point" (ID=2)
    │  └─ Record landmark 8 (index finger tip) position
    └─ Else: Record [0, 0]
    ↓
Maintain History Buffer:
    - FIFO deque with max 16 frames
    - Contains last 16 finger tip positions
    ↓
When Buffer Full (16 frames):
    ↓
[Point History Preprocessing]
    - Normalize trajectory by image dimensions
    - Center on first point in history
    - Flatten to 32 values (16 points × 2)
    ↓
[Point History Classifier]
    - Detect motion gestures (swipe, circle, zigzag)
    - Output: Motion gesture ID
    ↓
Final Result: Combination of static + motion gesture
```

### 8. Output Generation Phase

```
Gesture ID (from Classifier)
    ↓
[Label Lookup]
    Load: keypoint_classifier_label.csv
    ↓
Gesture Text: gesture_classifier_labels[ID]
    Examples: "hello", "thumbsup", "goodbye"
    ↓
Confidence Score: max(probability_distribution)
    Range: [0.0, 1.0]
    ↓
Handedness: "LEFT" or "RIGHT"
    From MediaPipe output
    ↓
Create Output Dictionary:
    {
        "gesture": "hello",
        "confidence": 0.94,
        "handedness": "RIGHT",
        "timestamp": timestamp_ms
    }
```

### 9. Display Phase

```
Output Dictionary
    ↓
┌─────────────────────────────────────────┐
│ Display Information on Video Frame       │
├─────────────────────────────────────────┤
│ ├─ Bounding Rectangle (green box)       │
│ ├─ Hand Landmarks (circles + lines)     │
│ ├─ Gesture Text (top label)             │
│ ├─ Confidence Score (percentage)        │
│ ├─ Handedness (LEFT/RIGHT)              │
│ └─ FPS Counter                          │
└─────────────────────────────────────────┘
    ↓
[Text-to-Speech Engine]
    - Convert gesture text to speech
    - Play audio output (non-blocking)
    ↓
Display Frame to User
    - Streamlit: st.image()
    - OpenCV: cv.imshow()
```

---

## Dataset Creation Process

### The Challenge

**Problem**: No public ISL datasets available

### The Solution

**Created custom dataset collection pipeline**

```
1. DEFINE GESTURE
   User specifies: "hello", "thumbsup", etc.
    ↓
2. SETUP TRAINING MODE
   Dashboard shows real-time video preview
    ↓
3. CONFIGURE ENVIRONMENT
   Setup lighting, background, hand distance
    ↓
4. CAPTURE IMAGES
   For each frame where hand is detected:
    ├─ Hand area > min_threshold (e.g., 5000 pixels)
    ├─ Save image to: training_data/{gesture_name}/
    ├─ Auto-name: {gesture_name}_{number:04d}.jpg
    └─ Repeat until 100+ images collected
    ↓
5. VERIFY DATA
   Check image count and quality
    ↓
6. UPDATE CONFIG
   Update gesture_config.json with metadata
    {
        "hello": {
            "created": "2026-02-04T16:45:47",
            "image_count": 100,
            "last_updated": "2026-02-04T16:45:47"
        }
    }
    ↓
7. READY FOR TRAINING
   Dataset ready for model training
```

### Dataset Statistics

```
Gesture Collection Summary:
├─ hello: 100 images ✓
├─ thumbsup: 100 images ✓
├─ Total: 200 images
├─ Resolution: 640×480 pixels
├─ Format: JPEG
├─ Diversity: Multiple angles, distances, lighting
└─ Status: Ready for production

Data Organization:
training_data/
├── hello/
│   ├── hello_0001.jpg
│   ├── hello_0002.jpg
│   └── ... (100 images)
└── thumbsup/
    ├── thumbsup_0001.jpg
    └── ... (100 images)
```

---

## Hand Detection Pipeline

### MediaPipe Configuration

```python
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,                              # Single to Dual-hand
    min_hand_detection_confidence=0.7,        # Detection threshold
    min_tracking_confidence=0.5,              # Tracking threshold
    running_mode=vision.RunningMode.VIDEO     # For video consistency
)
```

### Detection Output Structure

```
For each detected hand:
├─ 21 Landmarks
│  ├─ Index 0: Wrist (base of hand)
│  ├─ Index 1-4: Thumb (base, middle, PIP, tip)
│  ├─ Index 5-8: Index finger (base, PIP, DIP, tip)
│  ├─ Index 9-12: Middle finger
│  ├─ Index 13-16: Ring finger
│  └─ Index 17-20: Pinky finger
│
├─ Each Landmark Contains:
│  ├─ x: Normalized horizontal position [0-1]
│  ├─ y: Normalized vertical position [0-1]
│  ├─ z: Normalized depth [0-1] (relative)
│  └─ presence: Confidence score [0-1]
│
├─ Handedness:
│  └─ label: "LEFT" or "RIGHT"
│     confidence: Detection confidence [0-1]
│
└─ Hand Bounding Box
   ├─ x_min, y_min, x_max, y_max (pixel coordinates)
   └─ Defines region of interest
```

---

## Preprocessing Pipeline

### Step-by-Step Normalization

```
INPUT: Raw MediaPipe Landmarks
┌────────────────────────────────────────┐
│ Landmark[0].x = 0.45, .y = 0.52        │
│ Landmark[1].x = 0.48, .y = 0.49        │
│ ... (19 more landmarks)                 │
└────────────────────────────────────────┘
        │
        ▼
STEP 1: Extract Pixel Coordinates
┌────────────────────────────────────────┐
│ pixel_x = int(0.45 * 640) = 288        │
│ pixel_y = int(0.52 * 480) = 250        │
│ ... (repeat for all 21 landmarks)      │
└────────────────────────────────────────┘
        │
        ▼
STEP 2: Wrist Centering
┌────────────────────────────────────────┐
│ base_x = landmark[0].x = 288            │
│ base_y = landmark[0].y = 250            │
│                                        │
│ For each landmark i:                   │
│   x_norm[i] = x[i] - base_x            │
│   y_norm[i] = y[i] - base_y            │
│ Result: Hand centered on origin        │
└────────────────────────────────────────┘
        │
        ▼
STEP 3: Flatten to 1D Array
┌────────────────────────────────────────┐
│ landmark_list = [x0, y0, x1, y1, ...,  │
│                  x20, y20]             │
│ Length: 42 values (21 × 2)             │
│ Range: -640 to 640 (pixel values)      │
└────────────────────────────────────────┘
        │
        ▼
STEP 4: Calculate Maximum Absolute Value
┌────────────────────────────────────────┐
│ max_value = max(|x0|, |y0|, ..., |y20|)│
│ If all zeros: max_value = 1 (avoid div0)
└────────────────────────────────────────┘
        │
        ▼
STEP 5: Normalize to [-1, 1]
┌────────────────────────────────────────┐
│ normalized[i] = landmark_list[i] /     │
│                  max_value             │
│ Result: All values in [-1, 1] range    │
│ Properties: Scale invariant            │
│             Position invariant         │
└────────────────────────────────────────┘
        │
        ▼
OUTPUT: Normalized 42-element Vector
┌────────────────────────────────────────┐
│ [0.15, -0.22, 0.31, 0.18, ...,        │
│  -0.05, 0.12]                          │
│ Ready for Neural Network Input         │
└────────────────────────────────────────┘
```

---

## Classification Pipeline

### Neural Network Architecture

```
Input Layer (42 features)
    │
    ▼
Dense Layer 1 (128 units)
    ├─ Activation: ReLU (f(x) = max(0, x))
    ├─ Purpose: Learn non-linear relationships
    ├─ Weights: 42 × 128 = 5,376 parameters
    └─ Bias: 128 parameters
    │
    ▼
Dropout Layer 1 (20% drop rate)
    ├─ Purpose: Prevent overfitting
    ├─ Action: Randomly deactivate 20% of neurons
    └─ Effect: Forces network to learn robust features
    │
    ▼
Dense Layer 2 (64 units)
    ├─ Activation: ReLU
    ├─ Purpose: Refine learned features
    ├─ Weights: 128 × 64 = 8,192 parameters
    └─ Bias: 64 parameters
    │
    ▼
Dropout Layer 2 (20% drop rate)
    ├─ Purpose: Further regularization
    └─ Action: Randomly deactivate neurons
    │
    ▼
Output Layer (N units, where N = number of classes)
    ├─ Activation: Softmax
    │  (converts to probability distribution)
    ├─ Mathematical: softmax(x_i) = e^x_i / Σe^x_j
    ├─ Output: [P(class_1), P(class_2), ..., P(class_N)]
    │  Example: [0.85, 0.10, 0.05]
    └─ Interpretation:
       ├─ 85% probability of "hello"
       ├─ 10% probability of "thumbsup"
       └─ 5% probability of other gesture
    │
    ▼
Predicted Class
    ├─ argmax(probabilities)
    ├─ Example: class_0 (corresponding to "hello")
    └─ Confidence: max(probabilities) = 0.85
```

### Training Process

```
1. DATA PREPARATION
   ├─ Load training images from training_data/
   ├─ Extract landmarks using MediaPipe
   ├─ Preprocess: Normalize to 42-element vectors
   ├─ Create labels: [0, 1, ..., N-1]
   └─ Split: 80% train, 20% validation

2. MODEL COMPILATION
   ├─ Optimizer: Adam (adaptive learning)
   ├─ Loss: Categorical Crossentropy
   └─ Metrics: Accuracy

3. TRAINING LOOP
   For each epoch (1-50):
       For each batch (size=32):
           ├─ Forward pass: Predict on batch
           ├─ Calculate loss: How wrong is prediction
           ├─ Backward pass: Calculate gradients
           └─ Update weights: Minimize loss
       │
       Validate on test set:
       ├─ Calculate validation accuracy
       ├─ Monitor for overfitting
       └─ Save best model

4. MODEL EXPORT
   ├─ HDF5 format: For fine-tuning
   ├─ TFLite format: For mobile/edge devices
   └─ Label CSV: Gesture name mapping
```

---

## Classification Pipeline

### Inference (Prediction) Process

```
Live Video Frame
    │
    ├─ Capture frame
    ├─ Detect hand landmarks (MediaPipe)
    ├─ Preprocess landmarks (normalize)
    │
    ▼
Classifier Inference
    │
    ├─ Load TFLite model
    ├─ Prepare input: 42-element vector
    ├─ Run inference: <30ms typically
    ├─ Get output probabilities
    │
    ▼
Post-Processing
    │
    ├─ Apply confidence threshold (e.g., 0.7)
    ├─ If confidence < threshold: Reject
    ├─ Else: Accept prediction
    │
    ▼
Gesture Output
    │
    ├─ Gesture ID: argmax(probabilities)
    ├─ Confidence: max(probabilities)
    ├─ Label: gesture_labels[ID]
    │
    ▼
Display & Speak
    │
    ├─ Draw on video: bounding box, landmarks
    ├─ Display gesture text
    ├─ Trigger text-to-speech
    ├─ Update UI with results
    │
    ▼
Next Frame
```

---

## Output Generation Pipeline

### Text Output

```
Gesture Classification Result
    ├─ Gesture ID: integer (0, 1, 2, ...)
    │
    ▼
Label CSV Lookup
    ├─ Load: keypoint_classifier_label.csv
    ├─ Format:
    │  0,hello
    │  1,thumbsup
    │  2,point
    │
    ├─ Access: labels[gesture_id]
    │
    ▼
Gesture Text Output
    ├─ Example: "hello"
    ├─ Confidence: 0.94
    ├─ Store in variable
    │
    ▼
Logging/CSV Storage (Optional)
    ├─ Log format: [gesture_id, ...landmark_data]
    ├─ File: model/keypoint_classifier/keypoint.csv
    └─ Used for additional training
```

### Speech Output

```
Gesture Text (e.g., "hello")
    │
    ▼
Duplicate Detection
    ├─ If same gesture as last (< 1 sec)
    │  └─ Skip (avoid repetitive speech)
    └─ Else: Continue
    │
    ▼
Text-to-Speech Queue
    │
    ├─ Push text to queue
    ├─ Queue managed by SpeechEngine thread
    │
    ▼
Threading (Non-blocking)
    │
    ├─ Daemon thread for speech synthesis
    ├─ PyTTSX3 initialization
    ├─ Set speech rate (150 WPM typical)
    ├─ Generate speech
    ├─ Play audio (system speaker)
    │
    ▼
Main Thread Continues
    ├─ Processing not blocked by speech
    ├─ Allows smooth gesture recognition
    │
    ▼
Audio Output
    ├─ Spoken gesture name
    ├─ Example: "Hello" spoken to user
    └─ Enhances accessibility
```

---

## New Features Pipeline

### Dual-Hand Detection Pipeline

```
CONFIGURATION CHANGE
    ├─ Before: num_hands=1
    └─ After: num_hands=2
        │
        ▼
VIDEO FRAME
    │
    ├─ Input to MediaPipe
    │
    ▼
DETECTION
    │
    ├─ Hand 1 Processing
    │  ├─ 21 landmarks
    │  ├─ Confidence scores
    │  ├─ Handedness: RIGHT
    │  │
    │  └─ Convert to pixels
    │     ├─ Bounding rectangle 1
    │     ├─ Normalize landmarks
    │     │
    │     └─ Classify
    │        ├─ Gesture 1: "hello"
    │        └─ Confidence: 0.92
    │
    ├─ Hand 2 Processing
    │  ├─ 21 landmarks
    │  ├─ Confidence scores
    │  ├─ Handedness: LEFT
    │  │
    │  └─ Convert to pixels
    │     ├─ Bounding rectangle 2
    │     ├─ Normalize landmarks
    │     │
    │     └─ Classify
    │        ├─ Gesture 2: "thumbsup"
    │        └─ Confidence: 0.88
    │
    ▼
OUTPUT
    │
    ├─ Display both hands with landmarks
    ├─ Show both gesture labels
    ├─ Left hand: "thumbsup"
    ├─ Right hand: "hello"
    │
    ▼
SPEECH
    │
    ├─ Speak both gestures
    └─ Example: "hello thumbsup"
```

### Typing Mode Pipeline

```
SETUP PHASE
    │
    ├─ Load trained gestures
    │  Example: ["hello", "thumbsup"]
    │
    ├─ Create gesture-to-character mapping
    │  Default: First letter of gesture
    │  hello → "H"
    │  thumbsup → "T"
    │
    ├─ Allow user to customize mapping
    │  User input: hello → "hello "
    │  User input: thumbsup → "👍"
    │
    ▼
LIVE TYPING PHASE
    │
    ├─ Initialize text buffer
    │  typed_text = ""
    │
    ├─ Initialize gesture tracking
    │  last_gesture = None
    │  gesture_hold_count = 0
    │  hold_threshold = 10 frames (~300ms)
    │
    ▼
VIDEO LOOP (for each frame)
    │
    ├─ Capture frame
    ├─ Detect hands
    ├─ Extract landmarks
    ├─ Preprocess
    │
    ├─ Classify gesture
    │  Result: detected_gesture (e.g., "hello")
    │
    ────────────────────────────────────────┐
    │ Check if gesture is recognized         │
    │                                         │
    ├─ If detected_gesture == last_gesture   │
    │  ├─ Increment gesture_hold_count++     │
    │  │                                      │
    │  ├─ If gesture_hold_count > threshold  │
    │  │  ├─ Get mapped character            │
    │  │  │  char = mapping[gesture]         │
    │  │  │  Example: "hello" → "H"         │
    │  │  │                                   │
    │  │  ├─ Add to typed text               │
    │  │  │  typed_text += char              │
    │  │  │  Example: "H"                    │
    │  │  │                                   │
    │  │  ├─ Check auto-speak setting        │
    │  │  │  if auto_speak:                  │
    │  │  │     speak_text(char)             │
    │  │  │     Thread: "H" spoken           │
    │  │  │                                   │
    │  │  └─ Reset counters                  │
    │  │     last_gesture = None             │
    │  │     gesture_hold_count = 0          │
    │  │                                      │
    │  └─ Display real-time text             │
    │     st.text_area(typed_text)           │
    │                                         │
    ├─ Else if gesture changed               │
    │  ├─ Update last_gesture                │
    │  └─ Reset hold counter                 │
    │                                         │
    └─────────────────────────────────────────┘
    │
    ▼
USER CONTROLS
    │
    ├─ Clear Button
    │  └─ typed_text = ""
    │
    ├─ Backspace Button
    │  └─ typed_text = typed_text[:-1]
    │
    ├─ Copy Button
    │  └─ Copy typed_text to clipboard
    │
    ├─ Speak Button
    │  └─ speak_text(typed_text)
    │
    └─ Stop Button (Exit typing mode)
    │
    ▼
OUTPUT DISPLAY
    │
    ├─ Text Area: Current typed text
    ├─ Character Map: Gesture → Character mapping
    ├─ Live Video: Hand detection + gesture
    ├─ Gesture Display: Currently recognized
    └─ Status: Typing in progress
```

### Text-to-Speech Pipeline

```
INITIALIZATION
    │
    ├─ Create queue (thread-safe)
    ├─ Create daemon thread
    ├─ Start thread (background)
    │
    ▼
SPEECH REQUEST
    │
    ├─ Text to be spoken: "hello"
    ├─ Add to queue: queue.put("hello")
    ├─ Return immediately (non-blocking)
    │
    ▼
BACKGROUND THREAD
    │
    ├─ Monitor queue continuously
    ├─ Block waiting for text
    │
    └─ When text received
       │
       ├─ Initialize pyttsx3 engine
       ├─ Set properties
       │  ├─ Rate: 150 words per minute
       │  ├─ Volume: 1.0 (max)
       │  └─ Voice: Default system voice
       │
       ├─ Queue text: engine.say("hello")
       ├─ Wait for completion: engine.runAndWait()
       ├─ Cleanup: engine.stop()
       │
       └─ Return to queue monitoring
    │
    ▼
AUDIO OUTPUT
    │
    ├─ System speaker plays: "hello"
    ├─ Main thread unaffected
    ├─ Gesture recognition continues
    │
    ▼
AUTO-SPEAK IN TYPING MODE
    │
    ├─ When character added:
    │  ├─ If auto_speak enabled
    │  │  ├─ speak_text(char) called
    │  │  ├─ Added to queue
    │  │  └─ Played in background
    │  │
    │  └─ User sees character appear
    │     User hears character spoken
    └─ Seamless integrated experience
```

---

## Performance Optimization

### Real-Time Optimization

```
≤100ms CONSTRAINT (for 30 FPS)
│
├─ Eye-to-Brain latency: ~50-100ms (human perception)
├─ System processing: <50ms target
│  ├─ Video capture: ~5ms
│  ├─ Hand detection: ~30ms
│  ├─ Preprocessing: ~5ms
│  ├─ Classification: <10ms
│  └─ Display: ~5ms
│
└─ Result: Imperceptible lag to user
```

### Memory Management

```
Frame Buffer: 1 frame (640×480×3 bytes) ≈ 1MB
Point History: 16 frames × 2 coordinates = 32 floats ≈ 256 bytes
Model Weights: ~2MB (TFLite quantized)
Total Typical: 50-200MB depending on Streamlit caching
```

---

## Error Handling

```
Error Scenarios & Recovery

1. Hand Not Detected
   ├─ MediaPipe confidence < threshold
   ├─ Action: Wait for valid detection
   └─ Status: "Show your hand" message

2. Low Confidence Gesture
   ├─ Classification confidence < 0.7
   ├─ Action: Skip (don't add character)
   └─ Status: "Gesture not recognized" warning

3. Camera Error
   ├─ Camera not accessible
   ├─ Fallback: Try alternate camera index
   └─ Status: Error dialog to user

4. Model Loading Error
   ├─ Model file missing
   ├─ Action: Auto-download from MediaPipe
   └─ Status: "Downloading model..." message
```

