# 😊 Facial Emotion Recognition

CNN-based real-time facial emotion recognition built with **PyTorch** and deployed on **Streamlit Cloud**.

Detects: Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise

---

## Local setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Install CUDA PyTorch (for NVIDIA GPU training)
```bash
# Check your CUDA version first
nvidia-smi

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install -r requirements.txt
```

### 3. Place your dataset
