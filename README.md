<h1 align="center">HAVDEF – Hindi Audio-Visual Deepfake Defense</h1>
<p align="center">
  🚨 Real-time detection of AI-generated voice fraud calls in Hinglish using Deep Learning 🚨 <br>
  <strong>Python</strong> • <strong>TensorFlow</strong> • <strong>Flask</strong> • <strong>Signal Processing</strong> • <strong>Spectrogram CNNs</strong>
</p>

---

## 🎯 Project Overview

HAVDEF is an intelligent deepfake detection system that flags AI-generated voice scams during **real-time phone calls**. It focuses on **Hinglish**—a widely spoken Hindi-English mix in India—making the solution highly contextual and practical.

The system relies entirely on software-based methods (no hardware dependency), aiming to ensure wide device compatibility and robustness in noisy, multilingual environments. Trained on over **5000 real and synthetic voice samples**, it uses **spectrogram analysis and convolutional neural networks (CNNs)** to detect fraud with high accuracy.

---

## ✨ Key Features

- 🎙️ Real-time analysis of incoming voice streams
- 🧠 Deepfake detection using spectrogram-based CNN models
- 🔊 Robust against background noise and code-switching (Hinglish)
- 📱 Designed for low-resource, mobile-friendly deployment
- 📊 Trained on a diverse dataset of 5000+ voice samples

---

## 🧩 System Workflow

1. **User Phone Call Input**  
   Incoming voice audio is captured from the phone call.

2. **Audio Signal Capture**  
   Real-time buffering and signal extraction.

3. **Preprocessing & Noise Filtering**  
   Applies denoising, silence trimming, and normalization.

4. **Spectrogram Generation**  
   Converts audio signal into spectrogram images using Librosa.

5. **Deep Learning Model (CNN)**  
   TensorFlow CNN model classifies the spectrogram as real or AI-generated.

6. **Prediction Output**  
   Model outputs "Human" or "Deepfake" with confidence scores.

7. **Real-Time Alert System**  
   Alerts user instantly via UI/API in case of deepfake detection.

---

## 🛠️ Tech Stack

| Category           | Technologies Used                                      |
|-------------------|--------------------------------------------------------|
| Language          | Python                                                 |
| Backend           | Flask                                                  |
| Machine Learning  | TensorFlow, NumPy, Scikit-learn                        |
| Audio Processing  | Librosa, OpenSMILE                                     |
| Visualization     | Matplotlib, Seaborn                                    |
| Deployment        | Docker (Planned), REST API                             |

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/shivane1/HAVDEF.git
cd HAVDEF

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
