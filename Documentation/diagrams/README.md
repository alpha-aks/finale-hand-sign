# Diagrams Directory

This directory contains all diagrams and visual representations for the ISL Detection System report.

## Required Diagrams

### 1. Data Flow Diagram (DFD)
- **File**: dfd.png or dfd.pdf
- **Description**: Shows data movement from input (video) to output (speech/text)
- **Contains**:
  - Video Input
  - Hand Detection (MediaPipe)
  - Preprocessing
  - Classification
  - Output Generation
  - Speech Synthesis

### 2. Entity-Relationship Diagram (ER)
- **File**: er_diagram.png or er_diagram.pdf
- **Description**: Database structure and relationships
- **Entities**:
  - Gesture
  - TrainingImage
  - Model
  - Session
  - Label

### 3. Use Case Diagram
- **File**: usecase_diagram.png or usecase_diagram.pdf
- **Description**: System interactions with users
- **Actors**: End User, Administrator
- **Use Cases**:
  - Train New Gesture
  - Test Recognition
  - Type Using Gestures
  - View Training Data
  - Generate Speech

### 4. Class Diagram
- **File**: class_diagram.png or class_diagram.pdf
- **Description**: Object-oriented system structure
- **Classes**:
  - KeyPointClassifier
  - PointHistoryClassifier
  - SimpleGestureRecognizer
  - SimpleHandDetector
  - SpeechEngine

### 5. Sequence Diagram
- **File**: sequence_diagram.png or sequence_diagram.pdf
- **Description**: Interaction timeline for gesture recognition
- **Flow**: User → Camera → Detection → Classification → Output

### 6. Activity Diagram
- **File**: activity_diagram.png or activity_diagram.pdf
- **Description**: Processing workflow and decision points
- **Activities**:
  - Video Capture
  - Hand Detection
  - Preprocessing
  - Classification
  - Decision (Confidence check)
  - Output Generation

## Performance Graphs

### 7. Accuracy Metrics
- **File**: accuracy_graph.png
- **Description**: Model accuracy across different gestures
- **Data**:
  - Gesture names vs. accuracy
  - Training vs. validation accuracy
  - Confidence scores

### 8. FPS Performance
- **File**: fps_performance.png
- **Description**: Real-time FPS measurement
- **Data**:
  - FPS over time
  - Latency breakdown
  - Resource usage

### 9. Training Loss Curve
- **File**: training_loss.png
- **Description**: Model loss during training
- **Data**:
  - Training loss vs. epochs
  - Validation loss curve
  - Convergence point

## Dashboard Screenshots

### 10. Training Mode
- **File**: training_mode.png
- **Description**: Screenshot of gesture training interface
- **Shows**: Camera feed, gesture name input, image count

### 11. Testing Mode
- **File**: testing_mode.png
- **Description**: Real-time gesture recognition display
- **Shows**: Live camera, predictions, confidence scores

### 12. Typing Mode
- **File**: typing_mode.png
- **Description**: Gesture-based character typing interface
- **Shows**: Typed text, character mapping, speech controls

## Additional Resources

### 13. System Architecture
- **File**: system_architecture.png
- **Description**: High-level system component diagram

### 14. Pipeline Flowchart
- **File**: pipeline_flowchart.png
- **Description**: Complete data pipeline visualization
- **Referenced in**: PIPELINE_OVERVIEW.md

### 15. Hand Landmarks Visualization
- **File**: hand_landmarks.png
- **Description**: 21-point hand landmark positions
- **Shows**: Landmark numbering and skeletal structure

## How to Add Diagrams

### Option 1: Using Draw.io
1. Create diagrams in draw.io
2. Export as PNG (transparent background recommended)
3. Save to this directory with appropriate naming

### Option 2: Using PlantUML
1. Create diagrams in PlantUML text format
2. Convert to PNG using plantuml compiler
3. Save to this directory

### Option 3: Using Lucidchart
1. Design diagrams in Lucidchart
2. Download as PNG
3. Save at 300 DPI for print quality

### Option 4: Using Graphviz
```bash
# Example for Data Flow Diagram
dot -Tpng dfd.dot -o dfd.png
```

## LaTeX Integration

All diagrams referenced in chapters:
- paths are relative: `diagrams/filename.png`
- recommended size: 0.9\textwidth (90% of page width)
- format: includegraphics with caption

Example in LaTeX:
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{diagrams/dfd.png}
\caption{Data Flow Diagram - System processes}
\label{fig:dfd}
\end{figure}
```

## Quality Standards

- **Resolution**: Minimum 300 DPI for print
- **Format**: PNG or PDF preferred
- **Colors**: Use black/white or color-blind friendly palette
- **Text**: Clear, readable font (Arial, Calibri, or similar)
- **Size**: Optimized for printing (A4 page)

## File Naming Convention

```
{type}_{number}_{brief_description}.png

Examples:
- diagram_01_data_flow.png
- diagram_02_er_model.png
- diagram_03_usecase.png
- diagram_04_class_structure.png
- diagram_05_sequence.png
- diagram_06_activity.png
- graph_01_accuracy_metrics.png
- graph_02_fps_performance.png
- screenshot_01_training_mode.png
- screenshot_02_testing_mode.png
- screenshot_03_typing_mode.png
```

## Current Status

- ⏳ Placeholder descriptions provided
- 📝 Ready to add actual diagrams
- 📊 LaTeX chapters reference these files
- 🎯 All diagram types documented

---

**Last Updated**: February 16, 2026
**Format**: PNG/PDF
**Integration**: Integrated with LaTeX chapters via includegraphics
