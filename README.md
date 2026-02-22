# Hand Gesture Recognition Dashboard

Hand gesture training and recognition dashboard for webcam input. This project provides a Streamlit UI to capture gesture images, store them by label, and recognize gestures in real time.

## Requirements
- Python 3.10+
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run enhanced_dashboard.py
```

Open the app at http://localhost:8501

## Training

1. Open Train New Gesture.
2. Enter a gesture name (example: thumbs_up).
3. Click Start Capturing and show the gesture.
4. Images are saved under training_data/<gesture_name>.

## Recognition

1. Open Test & Recognize.
2. Click Reload Training Data.
3. Start Testing and show a trained gesture.

## License

Copyright (c) 2026 Disha Jhaveri, MGM College.
<pre>
