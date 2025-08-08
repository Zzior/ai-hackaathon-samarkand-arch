# 🚦 TRAFFIC VISION
**Detection of accidents on the road using computer vision.**

---

## 📋 Description

TRAFFIC VISION is an AI-powered system that detects accidents and traffic anomalies from video feeds.  
This project leverages deep learning models (e.g., YOLO, CNNs) for real-time accident detection, with potential integration into smart city systems for enhanced traffic safety.

---

## 📦 Installation

**Install PyTorch**  
   Visit [https://pytorch.org](https://pytorch.org) and select the appropriate installation command for your system (OS, Python version, CUDA).

   Example:
   ```bash
   pip3 install torch torchvision
   ````

**Install other dependencies**

   ```bash
   pip install -r requirements.txt
   ```
---
## ⚙️ Configuration (`config.yaml`)

Main sections you can edit:

- **`source_info`** — video source and ROI.
- **`detection`** — YOLO model, tracker, detection params.
- **`show`** — on-screen display options.
- **`web_mov`** — web streaming settings.
- **`video_writer`** — save video to disk.
- **`notify`** — Telegram alerts (location, buffer, bot token, chat ID).

---

## 🚀 Run the Application

```bash
python3 src/main.py
```

---
## 🌍 Frontend repository
https://github.com/khurshed555/ai-hackathon-samarkand-arch-frontend