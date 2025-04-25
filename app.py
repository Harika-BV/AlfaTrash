#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# Streamlit Smart-Trash Classifier
#   • offline-first (no internet after pip install)
#   • uses TorchScript INT8 model (dynamic or static quantised)
#   • webcam capture or file upload
#   • runs on CPU – Apple Silicon, Intel, Windows, Linux
# ────────────────────────────────────────────────────────────────

import streamlit as st
from PIL import Image
import torch, torchvision.transforms as T
import pathlib
torch.backends.quantized.engine = "qnnpack"      # ← add this

# ── 4. Streamlit page layout ───────────────────────────────────
st.set_page_config(page_title="Smart-Trash Offline Classifier",
                   page_icon="🗑️",
                   layout="centered")
# ── 1. Config ───────────────────────────────────────────────────
MODEL_FILE  = pathlib.Path("workspace/model_int8_dyn.pt")   # rename if you used FX static
LABELS_FILE = pathlib.Path("workspace/labels.txt")          # one class per line
IMG_SIZE    = 224
CONF_THRES  = 0.70                                # 70 % threshold

# ── 2. Cache model & transforms ─────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_and_preproc():
    model = torch.jit.load(str(MODEL_FILE), map_location="cpu")
    model.eval()

    labels = LABELS_FILE.read_text().strip().splitlines()

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]),
    ])
    return model, labels, tf

model, LABELS, preprocess = load_model_and_preproc()

# ── 3. Helper to predict one PIL.Image ──────────────────────────
def classify(img: Image.Image):
    x = preprocess(img.convert("RGB")).unsqueeze(0)         # 1×3×224×224
    with torch.no_grad():
        out = model(x)
        probs = out.softmax(dim=1)[0].tolist()
    idx   = int(max(range(len(probs)), key=probs.__getitem__))
    conf  = float(probs[idx])
    return LABELS[idx], conf


st.title("🗑️ Smart-Trash Classifier — offline")

st.write("""
Take a picture or upload an image to find out whether the item belongs in
**DRY**, **WET**, or **REJECT**.  
_All computation stays on-device; you can turn off Wi-Fi after the first load._
""")

tab_cam, tab_up = st.tabs(["📷 Camera", "🖼️ Upload"])

# ---- Camera capture ----
with tab_cam:
    frame = st.camera_input("Capture the object")
    if frame is not None:
        img = Image.open(frame)
        label, conf = classify(img)
        st.image(img, caption="Captured", use_container_width=True)
        if conf >= CONF_THRES:
            st.success(f"**{label.upper()}**  ({conf*100:.1f} %)")
        else:
            st.warning(f"Unsure ({conf*100:.1f} %).  Try another angle.")

# ---- File upload ----
with tab_up:
    file = st.file_uploader("Drop an image", type=["jpg", "jpeg", "png"])
    if file is not None:
        img = Image.open(file)
        label, conf = classify(img)
        st.image(img, caption=file.name, use_container_width=True)
        st.success(f"**{label.upper()}**  ({conf*100:.1f} %)")

st.caption("Model : MobileNetV3-Small  •  INT8 TorchScript  •  CPU-only")
