<h1 align="center">HAVDEF – Hindi Audio-Visual Deepfake Defense</h1>
<p align="center">
  🚨 Real-time detection of AI-generated voice fraud calls in Hinglish using Deep Learning 🚨 <br>
  <strong>Python</strong> • <strong>TensorFlow</strong> • <strong>Flask</strong> • <strong>Signal Processing</strong> • <strong>Spectrogram CNNs</strong>
</p>

---

## 🎯 Project Overview

HAVDEF is a deepfake detection system built to identify AI-generated voice fraud in **real-time phone calls**. It targets scams in **Hinglish**—a hybrid of Hindi and English—commonly spoken across India. By analyzing voice patterns through spectrograms and neural networks, HAVDEF aims to prevent AI-driven scams before they can cause harm.

> 🔐 Focus: Mobile-first, software-only solution to ensure device-independence, multilingual support, and high real-world applicability.

---

## ✨ Key Features

- 🎙️ Real-time voice stream analysis
- 🧠 Spectrogram-based CNN deepfake detection
- 🌐 Multilingual (Hinglish) and noise-resilient support
- 📱 Optimized for mobile and low-resource deployment
- 📊 Dataset: 5000+ real & synthetic Hinglish samples

---

## 🛠️ Tech Stack

| Category           | Technologies Used                                      |
|-------------------|--------------------------------------------------------|
| Language          | Python                                                 |
| Backend           | Flask                                                  |
| Machine Learning  | TensorFlow, NumPy, Scikit-learn                        |
| Audio Processing  | Librosa, OpenSMILE                                     |
| Visualization     | Matplotlib, Seaborn                                    |
| Deployment        | Docker (Planned), REST APIs                            |

---






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










