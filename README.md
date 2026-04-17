# 🖐️ Hand Gesture Classification

Real-time hand gesture and motion recognition using MediaPipe and TensorFlow Lite.

## 🚀 Demo

<table style="width:100%; table-layout:fixed; border-collapse:collapse;">
  <tr>
    <td style="padding:8px; vertical-align:top; width:50%;">
      <h4>Gesture Classification</h4>
      <img src="./demo/gesture-classification.gif" style="width:100%; border-radius:8px;"></img>
    </td>
    <td style="padding:8px; vertical-align:top; width:50%;">
      <h4>History Classification</h4>
      <img src="./demo/history-classification.gif" style="width:100%; border-radius:8px;"></img>
    </td>
  </tr>
</table>

## 🎯 Features

- **8 Hand Gestures**: 👍 👎 ☝️ ✌️ ✊ ✋ 🤟 👌
- **4 Motion Types**: Still • Clockwise • Anticlockwise • Move
- **Dual Hand Detection**: Recognize 2 hands simultaneously
- **High Performance**: TF Lite models, ~30ms inference per frame
- **Real-time Video Processing**: 1280×720 @ 30 FPS

## 📦 Installation

### Windows

```bash
pip install -r requirements_win.txt
```

### WSL/Linux

```bash
pip install -r requirements_wsl.txt
```

## 🚀 Quick Start

### 1️⃣ Collect Training Data

```bash
python collect_data.py
```

- Press `S` to start/stop recording
- Press `Q` to quit
- Choose mode: Gestures or Motion sequences

### 2️⃣ Run Real-time Classification

```bash
python app.py
```

- Point camera at your hand
- Press `Q` to exit

## 📂 Project Structure

```
├── app.py                  # Real-time gesture classifier
├── collect_data.py         # Training data collection tool
├── src/
│   ├── config.py          # Configuration & labels
│   ├── classifier.py      # TF Lite model wrapper
│   ├── processor.py       # Landmark normalization
│   └── visualizer.py      # Visualization utilities
├── models/                # Pre-trained TF Lite models
├── data/                  # Training datasets
└── notebooks/             # Training scripts
```

## 🔧 Key Components

| Module      | Purpose                       |
| ----------- | ----------------------------- |
| `MediaPipe` | Hand landmark detection       |
| `TF Lite`   | Fast gesture/motion inference |
| `OpenCV`    | Video capture & rendering     |
| `NumPy`     | Landmark processing           |

## 📊 Model Architecture

- **Gesture Model**: Classifies static hand poses
- **Motion Model**: Analyzes 16-frame sequences for movement direction

## ⚙️ Configuration

Edit `src/config.py` to customize:

- Max hands: `MAX_HANDS = 2`
- Input resolution: `IMAGE_SHAPE = (1280, 720)`
- Model paths and labels

## 📝 Notes

- Requires webcam input
- Optimal lighting recommended
- Models run on CPU (GPU optional)

## 👤 Author

Made with 💻 and ☕ by [@udham2511]("https://www.github.com/udham2511")

## 🙏 Inspiration

Inspired by [@Kazuhito00](https://github.com/Kazuhito00) and their innovative hand gesture recognition projects.

## 📜 License

This project is open-source and available under the **MIT License**.
