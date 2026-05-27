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
IMG_SIZE = 48
DEVICE   = torch.device("cpu")

EMOTION_META = {
    "angry":    ("😠", "#FF4500"),
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
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
#  INFERENCE TRANSFORM
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ─────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpt    = torch.load("model/emotion_model.pth", map_location=DEVICE)
    m       = EmotionCNN().to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    classes = ckpt.get("classes", list(EMOTION_META.keys()))
    return m, classes


@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

# ─────────────────────────────────────────────
#  CORE INFERENCE
# ─────────────────────────────────────────────
def predict_face(face_bgr, model, classes):
    pil    = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    tensor = infer_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    idx     = int(np.argmax(probs))
    emotion = classes[idx]
    emoji, color = EMOTION_META.get(emotion, ("❓", "#888"))
    return emotion, emoji, color, float(probs[idx]), probs


def run_detection(image_rgb, model, cascade, classes):
    """Detect faces, annotate frame, return results."""
    img_bgr   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img_bgr,   cv2.COLOR_BGR2GRAY)
    faces     = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    annotated = img_bgr.copy()
    results   = []

    for (x, y, w, h) in faces:
        crop                             = img_bgr[y:y+h, x:x+w]
        emotion, emoji, hex_col, conf, probs = predict_face(crop, model, classes)
        r2, g2, b2 = int(hex_col[1:3],16), int(hex_col[3:5],16), int(hex_col[5:7],16)
        bgr        = (b2, g2, r2)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), bgr, 2)

        # Label background pill
        label     = f"{emotion.upper()}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, y - th - 14), (x + tw + 10, y), bgr, -1)
        cv2.putText(
            annotated, label,
            (x + 5, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        results.append({
            "emotion": emotion, "emoji": emoji,
            "color": hex_col, "conf": conf, "probs": probs,
        })

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), results, len(faces)

# ─────────────────────────────────────────────
#  SESSION STATE  (tracks history + live toggle)
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []        # list of {emotion, conf, timestamp}
if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "snapshot_count" not in st.session_state:
    st.session_state.snapshot_count = 0

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
st.caption("Uses your device camera — snapshots are processed instantly on Streamlit Cloud.")

status_col, toggle_col = st.columns([3, 1])
with status_col:
    model_ok = st.empty()
    model_ok.success("✅ Model ready")
with toggle_col:
    live = st.toggle("📷 Live mode", value=st.session_state.live_mode)
    st.session_state.live_mode = live

st.divider()

# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
cam_col, result_col = st.columns([1, 1], gap="medium")

# ── Camera input ──────────────────────────────
with cam_col:
    st.subheader("Camera")

    if st.session_state.live_mode:
        st.info(
            "📸 Live mode on — keep clicking **Take photo** for continuous updates.",
            icon="ℹ️",
        )

    frame = st.camera_input(
        label="Take a photo",
        key=f"cam_{st.session_state.snapshot_count}",
        help="Click the button below the preview to capture",
        label_visibility="collapsed",
    )

    if st.session_state.live_mode and frame is not None:
        # Auto-increment key so camera resets immediately after capture
        time.sleep(0.3)
        st.session_state.snapshot_count += 1
        st.rerun()

# ── Result panel ──────────────────────────────
with result_col:
    st.subheader("Result")

    if frame is None:
        st.markdown(
            """
            <div style='
                border: 2px dashed var(--color-border-secondary);
                border-radius: 12px;
                padding: 48px 24px;
                text-align: center;
                color: var(--color-text-secondary);
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

        with st.spinner("Analysing…"):
            annotated, results, n_faces = run_detection(
                img_array, model, cascade, classes
            )

        st.image(annotated, use_column_width=True)

        if n_faces == 0:
            st.warning("No face detected — try better lighting or move closer.")
        else:
            for i, r in enumerate(results):
                # Dominant emotion badge
                st.markdown(
                    f"""
                    <div style='
                        background: {r["color"]}22;
                        border: 1.5px solid {r["color"]};
                        border-radius: 10px;
                        padding: 10px 16px;
                        margin-bottom: 10px;
                    '>
                        <span style='font-size:28px'>{r["emoji"]}</span>&nbsp;
                        <strong style='font-size:18px; color:{r["color"]}'>
                            {r["emotion"].capitalize()}
                        </strong>
                        <span style='color:var(--color-text-secondary); font-size:13px'>
                            &nbsp;·&nbsp;{r["conf"]*100:.1f}% confidence
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Push to history
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
            bar          = (
                f"<span style='color:{color}'>{'█' * filled}</span>"
                f"<span style='opacity:0.2'>{'█' * (20 - filled)}</span>"
            )
            st.markdown(
                f"{emoji} **{cls.capitalize()}** &nbsp; {bar} &nbsp; `{prob*100:.1f}%`",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────
#  EMOTION HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.divider()

    hist_header, clear_col = st.columns([4, 1])
    with hist_header:
        st.subheader(f"Session history  ({len(st.session_state.history)} snapshots)")
    with clear_col:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Emotion frequency summary
    from collections import Counter
    freq = Counter(h["emotion"] for h in st.session_state.history)
    summary_cols = st.columns(len(freq))
    for col, (emo, count) in zip(summary_cols, freq.most_common()):
        emoji, color = EMOTION_META.get(emo, ("❓", "#888"))
        col.markdown(
            f"""
            <div style='
                text-align:center;
                background:{color}18;
                border:1px solid {color}66;
                border-radius:8px;
                padding:8px 4px;
            '>
                <div style='font-size:22px'>{emoji}</div>
                <div style='font-size:11px;color:var(--color-text-secondary)'>
                    {emo}<br><strong>{count}×</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Recent captures table
    st.markdown("&nbsp;")
    recent = st.session_state.history[-10:][::-1]
    for h in recent:
        emoji, color = EMOTION_META.get(h["emotion"], ("❓", "#888"))
        st.markdown(
            f"<span style='color:var(--color-text-secondary);font-size:12px'>"
            f"{h['timestamp']}</span> &nbsp; "
            f"{emoji} **{h['emotion'].capitalize()}** &nbsp; "
            f"<span style='color:var(--color-text-secondary);font-size:13px'>"
            f"{h['conf']*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

st.divider()
st.caption("PyTorch CNN · OpenCV Haar Cascade · Streamlit Cloud")