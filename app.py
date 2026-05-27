import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="centered",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 128
DEVICE   = torch.device("cpu")   # Streamlit Cloud has no GPU

# 8 classes — matches your dataset (includes contempt)
EMOTION_META = {
    "angry":    ("😠", "#FF4500"),
    "contempt": ("😒", "#8B4513"),
    "disgust":  ("🤢", "#228B22"),
    "fear":     ("😨", "#8A2BE2"),
    "happy":    ("😄", "#FFD700"),
    "neutral":  ("😐", "#808080"),
    "sad":      ("😢", "#4169E1"),
    "surprise": ("😲", "#FF69B4"),
}

# ─────────────────────────────────────────────
#  MODEL  (must match train.py exactly)
# ─────────────────────────────────────────────
class EmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ─────────────────────────────────────────────
#  INFERENCE TRANSFORM
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpt        = torch.load("model/emotion_model.pth", map_location=DEVICE)
    num_classes = ckpt.get("num_classes", 8)
    classes     = ckpt.get("classes", sorted(EMOTION_META.keys()))
    m           = EmotionCNN(num_classes=num_classes).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, classes


@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def predict_face(face_bgr, model, classes):
    pil    = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    tensor = infer_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx     = int(np.argmax(probs))
    emotion = classes[idx]
    emoji, color = EMOTION_META.get(emotion, ("❓", "#888888"))
    return emotion, emoji, color, float(probs[idx]), probs


def run_detection(image_rgb, model, cascade, classes):
    img_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img_bgr,   cv2.COLOR_BGR2GRAY)
    faces     = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    annotated = img_bgr.copy()
    results   = []

    for (x, y, w, h) in faces:
        crop                                 = img_bgr[y:y+h, x:x+w]
        emotion, emoji, hex_col, conf, probs = predict_face(crop, model, classes)

        r2, g2, b2 = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
        bgr        = (b2, g2, r2)

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), bgr, 2)

        # Label pill
        label       = f"{emotion.upper()}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, y - th - 14), (x + tw + 10, y), bgr, -1)
        cv2.putText(annotated, label, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        results.append({
            "emotion": emotion, "emoji": emoji,
            "color":   hex_col, "conf":  conf, "probs": probs,
        })

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), results, len(faces)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history        = []
if "live_mode"      not in st.session_state: st.session_state.live_mode      = False
if "snapshot_count" not in st.session_state: st.session_state.snapshot_count = 0

# ─────────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────────
try:
    model, classes = load_model()
    cascade        = load_cascade()
except FileNotFoundError:
    st.error("❌ `model/emotion_model.pth` not found — run `python train.py` first.")
    st.stop()
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("😊 Real-time Emotion Recognition")
st.caption("Takes a snapshot from your camera and predicts your facial emotion.")

status_col, toggle_col = st.columns([3, 1])
with status_col:
    st.success(f"✅ Model ready — {len(classes)} emotions")
with toggle_col:
    live = st.toggle("📷 Live mode", value=st.session_state.live_mode)
    st.session_state.live_mode = live

st.divider()

# ─────────────────────────────────────────────
#  CAMERA + RESULT  (side by side)
# ─────────────────────────────────────────────
cam_col, result_col = st.columns([1, 1], gap="medium")

with cam_col:
    st.subheader("Camera")
    if st.session_state.live_mode:
        st.info("📸 Live mode — keep clicking Take photo for continuous updates.", icon="ℹ️")

    frame = st.camera_input(
        label="Take a photo",
        key=f"cam_{st.session_state.snapshot_count}",
        label_visibility="collapsed",
    )

    if st.session_state.live_mode and frame is not None:
        time.sleep(0.3)
        st.session_state.snapshot_count += 1
        st.rerun()

with result_col:
    st.subheader("Result")

    if frame is None:
        st.markdown(
            """
            <div style='
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 52px 24px;
                text-align: center;
                color: #888;
                font-size: 14px;
            '>
                📷<br><br>Take a photo to see<br>the emotion prediction
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        image     = Image.open(frame).convert("RGB")
        img_array = np.array(image)

        with st.spinner("Analysing face…"):
            annotated, results, n_faces = run_detection(
                img_array, model, cascade, classes
            )

        st.image(annotated, use_column_width=True)

        if n_faces == 0:
            st.warning("No face detected — try better lighting or move closer.")
        else:
            for r in results:
                st.markdown(
                    f"""
                    <div style='
                        background: {r["color"]}22;
                        border: 1.5px solid {r["color"]};
                        border-radius: 10px;
                        padding: 10px 16px;
                        margin-bottom: 10px;
                    '>
                        <span style='font-size: 28px'>{r["emoji"]}</span>&nbsp;
                        <strong style='font-size: 18px; color: {r["color"]}'>
                            {r["emotion"].capitalize()}
                        </strong>
                        <span style='color: #888; font-size: 13px'>
                            &nbsp;·&nbsp; {r["conf"]*100:.1f}% confidence
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            # Append to history
            for r in results:
                st.session_state.history.append({
                    "emotion":   r["emotion"],
                    "emoji":     r["emoji"],
                    "conf":      r["conf"],
                    "timestamp": time.strftime("%H:%M:%S"),
                })

# ─────────────────────────────────────────────
#  CONFIDENCE BREAKDOWN
# ─────────────────────────────────────────────
if frame is not None and "results" in dir() and results:
    st.divider()
    st.subheader("Confidence breakdown")
    for r in results:
        for j, cls in enumerate(classes):
            prob         = float(r["probs"][j]) if j < len(r["probs"]) else 0.0
            emoji, color = EMOTION_META.get(cls, ("❓", "#888"))
            filled       = int(prob * 20)
            bar = (
                f"<span style='color:{color}'>{'█' * filled}</span>"
                f"<span style='opacity:0.15'>{'█' * (20 - filled)}</span>"
            )
            st.markdown(
                f"{emoji} **{cls.capitalize()}** &nbsp; {bar} &nbsp; `{prob*100:.1f}%`",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
#  SESSION HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.divider()

    hist_col, clear_col = st.columns([4, 1])
    with hist_col:
        st.subheader(f"Session history  ({len(st.session_state.history)} snapshots)")
    with clear_col:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Frequency summary badges
    from collections import Counter
    freq         = Counter(h["emotion"] for h in st.session_state.history)
    summary_cols = st.columns(len(freq))
    for col, (emo, count) in zip(summary_cols, freq.most_common()):
        emoji, color = EMOTION_META.get(emo, ("❓", "#888"))
        col.markdown(
            f"""
            <div style='
                text-align: center;
                background: {color}18;
                border: 1px solid {color}66;
                border-radius: 8px;
                padding: 8px 4px;
            '>
                <div style='font-size: 22px'>{emoji}</div>
                <div style='font-size: 11px; color: #888'>
                    {emo}<br><strong>{count}×</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Last 10 entries
    st.markdown("&nbsp;")
    for h in st.session_state.history[-10:][::-1]:
        emoji, color = EMOTION_META.get(h["emotion"], ("❓", "#888"))
        st.markdown(
            f"<span style='color:#888; font-size:12px'>{h['timestamp']}</span>"
            f" &nbsp; {emoji} **{h['emotion'].capitalize()}**"
            f" &nbsp; <span style='color:#888; font-size:13px'>{h['conf']*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption("PyTorch CNN · OpenCV Haar Cascade · Streamlit Cloud")