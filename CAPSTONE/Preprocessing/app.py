import streamlit as st
import os
import json
import time
import cv2
import torch
import numpy as np

from datetime import datetime
from PIL import Image

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DeepFishVision | Industrial Vision System",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_PATH = (
    r"C:\Users\veman\OneDrive\Desktop"
    r"\Open CV New Slot\CAPSTONE"
)

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Outputs",
    "models"
)


# ============================================================
# THEME / DESIGN TOKENS
# ============================================================

ACCENT = "#00E5FF"        # cyan — primary accent
ACCENT_2 = "#7C4DFF"      # violet — secondary accent
OK_COLOR = "#00FFA3"      # green — pass / high confidence
WARN_COLOR = "#FFB020"    # amber — caution / mid confidence
BAD_COLOR = "#FF4D6D"     # red — fail / low confidence
BG_PANEL = "rgba(255, 255, 255, 0.035)"
BORDER = "rgba(0, 229, 255, 0.25)"


# ============================================================
# CUSTOM CSS — INDUSTRIAL / HUD STYLE
# ============================================================

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', monospace;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 10% 0%, rgba(0,229,255,0.06), transparent 40%),
            radial-gradient(circle at 90% 10%, rgba(124,77,255,0.07), transparent 45%),
            #05070A;
    }}

    /* ---------- HEADER ---------- */
    .hud-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 26px;
        border: 1px solid {BORDER};
        border-radius: 10px;
        background: {BG_PANEL};
        margin-bottom: 6px;
    }}

    .main-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 34px;
        font-weight: 900;
        letter-spacing: 2px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}

    .subtitle {{
        font-size: 13px;
        color: #8CA0B3;
        letter-spacing: 1px;
        margin-top: 4px;
        text-transform: uppercase;
    }}

    .status-pill {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 999px;
        border: 1px solid rgba(0,255,163,0.4);
        background: rgba(0,255,163,0.08);
        font-size: 12px;
        letter-spacing: 1px;
        color: {OK_COLOR};
        text-transform: uppercase;
        white-space: nowrap;
    }}

    .status-dot {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: {OK_COLOR};
        box-shadow: 0 0 8px {OK_COLOR};
        animation: pulse 1.6s infinite;
    }}

    @keyframes pulse {{
        0%   {{ opacity: 1;   transform: scale(1);   }}
        50%  {{ opacity: 0.4; transform: scale(1.3); }}
        100% {{ opacity: 1;   transform: scale(1);   }}
    }}

    /* ---------- SECTION LABELS ---------- */
    .section-label {{
        font-family: 'Orbitron', sans-serif;
        font-size: 15px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {ACCENT};
        border-left: 3px solid {ACCENT};
        padding-left: 10px;
        margin: 26px 0 14px 0;
    }}

    /* ---------- PANELS / CARDS ---------- */
    .hud-panel {{
        border: 1px solid {BORDER};
        background: {BG_PANEL};
        border-radius: 10px;
        padding: 16px 18px;
    }}

    .result-box {{
        padding: 22px;
        border-radius: 12px;
        border: 1px solid {BORDER};
        background: {BG_PANEL};
        text-align: center;
        position: relative;
        overflow: hidden;
    }}

    .result-box::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
    }}

    .result-label {{
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #8CA0B3;
        margin-bottom: 8px;
    }}

    .species {{
        font-family: 'Orbitron', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: #F2F6FA;
        letter-spacing: 1px;
    }}

    .confidence-value {{
        font-family: 'Orbitron', sans-serif;
        font-size: 30px;
        font-weight: 900;
    }}

    /* confidence bar */
    .conf-track {{
        width: 100%;
        height: 10px;
        border-radius: 6px;
        background: rgba(255,255,255,0.08);
        margin-top: 12px;
        overflow: hidden;
    }}

    .conf-fill {{
        height: 100%;
        border-radius: 6px;
    }}

    /* pipeline step chips */
    .pipeline-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 6px;
    }}

    .pipeline-chip {{
        font-size: 12px;
        padding: 7px 13px;
        border-radius: 6px;
        border: 1px solid rgba(0,255,163,0.35);
        background: rgba(0,255,163,0.06);
        color: {OK_COLOR};
        letter-spacing: 0.5px;
    }}

    /* sidebar */
    section[data-testid="stSidebar"] {{
        background: #070A0F;
        border-right: 1px solid {BORDER};
    }}

    .sidebar-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        color: {ACCENT};
        letter-spacing: 1px;
    }}

    .module-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
        padding: 6px 0;
        color: #C4D2DE;
        border-bottom: 1px dashed rgba(255,255,255,0.08);
    }}

    .module-ok {{
        color: {OK_COLOR};
        font-size: 11px;
        letter-spacing: 1px;
    }}

    .metric-chip {{
        border: 1px solid {BORDER};
        background: {BG_PANEL};
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }}

    .metric-chip .k {{
        font-size: 11px;
        color: #8CA0B3;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .metric-chip .v {{
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        color: #F2F6FA;
    }}

    /* dataframe / uploader tweaks */
    [data-testid="stFileUploaderDropzone"] {{
        border: 1.5px dashed {BORDER} !important;
        background: {BG_PANEL} !important;
        border-radius: 10px !important;
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================

def confidence_color(confidence_pct: float) -> str:
    if confidence_pct >= 85:
        return OK_COLOR
    if confidence_pct >= 60:
        return WARN_COLOR
    return BAD_COLOR


def confidence_verdict(confidence_pct: float) -> str:
    if confidence_pct >= 85:
        return "HIGH CONFIDENCE — ACCEPT"
    if confidence_pct >= 60:
        return "MODERATE — REVIEW RECOMMENDED"
    return "LOW CONFIDENCE — MANUAL CHECK REQUIRED"


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown(
        f"""
        <div class="hud-header">
            <div>
                <div class="main-title">⌬ DEEPFISHVISION</div>
                <div class="subtitle">
                    Industrial Underwater Species Recognition
                    &nbsp;·&nbsp; OpenCV Pre-Processing
                    &nbsp;·&nbsp; Vision Transformer Inference
                </div>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div> SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    return model, processor


@st.cache_data
def load_classes():
    class_file = os.path.join(MODEL_PATH, "class_names.json")
    with open(class_file, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("⚠ MODEL ARTIFACTS NOT FOUND AT CONFIGURED PATH. Check MODEL_PATH and re-deploy.")
    st.stop()

model, processor = load_model()
class_names = load_classes()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown('<div class="sidebar-title">⌬ DEEPFISHVISION</div>', unsafe_allow_html=True)
    st.caption("Computer Vision Inference Console")

    st.markdown("---")

    st.markdown(
        '<div class="metric-chip"><div class="k">Species Classes</div>'
        f'<div class="v">{len(class_names)}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="metric-chip"><div class="k">Model Architecture</div>'
        '<div class="v">ViT-Base</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="metric-chip"><div class="k">Session Time</div>'
        f'<div class="v">{datetime.now().strftime("%H:%M:%S")}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("**PIPELINE MODULES**")

    modules = [
        "OpenCV Enhancement (CLAHE)",
        "Gaussian Denoising",
        "Vision Transformer Encoder",
        "Species Classification Head",
        "Confidence Scoring",
        "Result Visualization"
    ]

    for m in modules:
        st.markdown(
            f'<div class="module-row">{m} <span class="module-ok">● ACTIVE</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    with st.expander("⚙ Advanced Settings"):
        conf_threshold = st.slider(
            "Confidence alert threshold (%)",
            min_value=0, max_value=100, value=60
        )
        top_k = st.slider(
            "Top-K predictions to display",
            min_value=3, max_value=min(10, len(class_names)), value=5
        )

    st.caption("DeepFishVision · Industrial Build")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown('<div class="section-label">01 · INPUT FEED</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an underwater fish image for analysis",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if original_image is None:
        st.error("⚠ UNABLE TO DECODE IMAGE. File may be corrupted or unsupported.")
        st.stop()

    # --------------------------------------------------------
    # OpenCV Enhancement
    # --------------------------------------------------------
    resized = cv2.resize(original_image, (224, 224))
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge((enhanced_l, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # --------------------------------------------------------
    # Pipeline status chips
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="pipeline-row">
            <div class="pipeline-chip">✔ IMAGE DECODED</div>
            <div class="pipeline-chip">✔ RESIZED 224×224</div>
            <div class="pipeline-chip">✔ GAUSSIAN DENOISE</div>
            <div class="pipeline-chip">✔ CLAHE CONTRAST ENHANCEMENT</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-label">02 · VISUAL COMPARISON</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.caption("RAW INPUT")
        st.image(
            cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="hud-panel">', unsafe_allow_html=True)
        st.caption("OPENCV ENHANCED (CLAHE)")
        st.image(
            enhanced_rgb,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # ViT Prediction
    # --------------------------------------------------------
    st.markdown('<div class="section-label">03 · MODEL INFERENCE</div>', unsafe_allow_html=True)

    with st.spinner("Running Vision Transformer inference..."):
        start_time = time.time()

        pil_image = Image.fromarray(enhanced_rgb)
        inputs = processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        k = min(top_k, len(class_names))
        top_probabilities, top_indices = torch.topk(probabilities, k=k)

        inference_ms = (time.time() - start_time) * 1000

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------
    best_index = top_indices[0][0].item()
    best_species = class_names[best_index]
    best_confidence = top_probabilities[0][0].item() * 100
    color = confidence_color(best_confidence)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Predicted Species</div>
                <div class="species">{best_species}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col2:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Confidence Score</div>
                <div class="confidence-value" style="color:{color};">{best_confidence:.2f}%</div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{best_confidence:.1f}%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with result_col3:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Inference Time</div>
                <div class="confidence-value" style="color:{ACCENT};">{inference_ms:.0f} ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if best_confidence < conf_threshold:
        st.warning(f"⚠ {confidence_verdict(best_confidence)} — score below the {conf_threshold}% operator threshold.")
    else:
        st.success(f"✔ {confidence_verdict(best_confidence)}")

    # --------------------------------------------------------
    # TOP-K PREDICTIONS
    # --------------------------------------------------------
    st.markdown('<div class="section-label">04 · RANKED PREDICTIONS</div>', unsafe_allow_html=True)

    top_data = []
    for i in range(len(top_indices[0])):
        index = top_indices[0][i].item()
        probability = top_probabilities[0][i].item() * 100
        top_data.append({
            "Rank": i + 1,
            "Species": class_names[index],
            "Confidence (%)": round(probability, 2)
        })

    table_col, chart_col = st.columns([1, 1])

    with table_col:
        st.dataframe(
            top_data,
            use_container_width=True,
            hide_index=True
        )

    with chart_col:
        st.bar_chart(
            {item["Species"]: item["Confidence (%)"] for item in top_data}
        )

    st.caption(
        f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        f"Model: ViT-Base · Classes: {len(class_names)} · Build: Industrial"
    )

else:
    st.markdown(
        f"""
        <div class="hud-panel" style="text-align:center; padding:40px;">
            <div style="font-size:15px; color:#8CA0B3; letter-spacing:1px;">
                AWAITING INPUT — Upload an underwater fish image to begin classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )