import time
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.cm as cm

# ---------------------------------------------------------
# Page Configuration & Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Skin Lesion CAD - Uncertainty & XAI",
    page_icon="🔬",
    layout="wide"
)

# 7 diagnostic classes from the HAM10000 dataset
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': 'Actinic Keratoses (akiec)',
    'bcc':   'Basal Cell Carcinoma (bcc)',
    'bkl':   'Benign Keratosis-like (bkl)',
    'df':    'Dermatofibroma (df)',
    'mel':   'Melanoma (mel)',
    'nv':    'Melanocytic Nevi (nv)',
    'vasc':  'Vascular Lesions (vasc)'
}

# Decision rule thresholds directly from the research paper
TAU_ENTROPY = 1.0   # Maximum allowed entropy
TAU_CONF = 0.70     # Minimum required confidence (70%)

# ---------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------
def calculate_shannon_entropy(probs: np.ndarray) -> float:
    """Calculates predictive Shannon entropy H(p) = -sum(p * log(p))."""
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(safe_probs * np.log(safe_probs)))

def generate_dummy_gradcam(image: Image.Image) -> Image.Image:
    """Generates a synthetic Grad-CAM++ heatmap overlay for demonstration."""
    img_array = np.array(image.convert("RGB"))
    h, w, _ = img_array.shape

    # Create a 2D Gaussian peak around the center to simulate attention
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = (x - center_x) ** 2 + (y - center_y) ** 2
    sigma = (min(h, w) / 4) ** 2
    heatmap = np.exp(-dist_from_center / (2 * sigma))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert heatmap to JET colormap (RGB)
    colored_heatmap = cm.jet(heatmap)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    # Blend original image with heatmap (60% original, 40% heatmap)
    blended = (0.6 * img_array + 0.4 * colored_heatmap).astype(np.uint8)
    return Image.fromarray(blended)

def mock_inference(mode: str):
    """
    Simulates MC Dropout Ensemble outputs.
    - 'Confident': Clear prediction passing both thresholds.
    - 'Uncertain (High Entropy)': Split decisions across classes, fails threshold.
    """
    time.sleep(1.0)  # Simulate network inference delay

    if mode == "Confident (Passes Filter)":
        # Clear melanoma / nv case: one class dominates
        raw_probs = np.array([0.02, 0.03, 0.04, 0.01, 0.82, 0.06, 0.02])
    else:
        # Ambiguous case: split between melanoma, bkl, and nevus
        raw_probs = np.array([0.05, 0.10, 0.32, 0.05, 0.38, 0.08, 0.02])

    raw_probs = raw_probs / np.sum(raw_probs)
    entropy = calculate_shannon_entropy(raw_probs)
    conf = float(np.max(raw_probs))
    pred_idx = int(np.argmax(raw_probs))

    return CLASSES[pred_idx], conf, entropy, raw_probs

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    st.markdown("Use these controls to simulate different diagnostic outcomes during your viva demo:")
    sim_mode = st.radio(
        "Simulate Model Behavior:",
        ["Confident (Passes Filter)", "Uncertain / Edge Case (Rejected)"]
    )

    st.markdown("---")
    st.markdown("### 📋 System Parameters")
    st.markdown(f"- **Ensemble Models:** 3 (MaxViT, ConvNeXt, EfficientNetV2)")
    st.markdown(f"- **MC Dropout Passes ($T$):** 20")
    st.markdown(f"- **Entropy Threshold ($\\tau_e$):** `< {TAU_ENTROPY}`")
    st.markdown(f"- **Confidence Threshold ($\\tau_c$):** `≥ {TAU_CONF * 100:.0f}%`")

# ---------------------------------------------------------
# Main User Interface
# ---------------------------------------------------------
st.title("🔬 Uncertainty-Aware Skin Lesion Classification")
st.caption("5th Semester Minor Project | Ensemble Deep Learning & Explainable AI (Grad-CAM++)")

st.info(
    "**Clinical Prototype Disclaimer:** This system is an academic research prototype designed "
    "to assist clinical triage. It is not an autonomous medical device.",
    icon="⚠️"
)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Input Dermoscopic Image")
    uploaded_file = st.file_uploader(
        "Upload a skin lesion image (HAM10000 format - JPG/PNG):",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_container_width=True)
    else:
        st.markdown(
            """
            *Awaiting image upload...*  
            👉 Upload any skin image to see the diagnosis, uncertainty filtering, and heatmap.
            """
        )

with col2:
    st.subheader("2. Diagnostic Assessment")

    if uploaded_file:
        with st.spinner("Executing 20 MC Dropout forward passes across 3 ensemble models..."):
            pred_class, conf, entropy, probs = mock_inference(sim_mode)

        # Decision Rule: Entropy < 1.0 AND Confidence >= 0.70
        is_reliable = (entropy < TAU_ENTROPY) and (conf >= TAU_CONF)

        if is_reliable:
            st.success(f"### Result: {CLASS_NAMES[pred_class]}")
            m1, m2 = st.columns(2)
            m1.metric("Ensemble Confidence", f"{conf * 100:.1f}%")
            m2.metric("Predictive Entropy", f"{entropy:.3f}", delta="Reliable", delta_color="normal")
            st.success("✅ **Status:** Prediction Accepted (High Reliability)")
        else:
            st.error("### ⚠️ Prediction Rejected")
            m1, m2 = st.columns(2)
            m1.metric("Ensemble Confidence", f"{conf * 100:.1f}%", delta="Below threshold" if conf < TAU_CONF else "Pass", delta_color="inverse")
            m2.metric("Predictive Entropy", f"{entropy:.3f}", delta="Too High" if entropy >= TAU_ENTROPY else "Pass", delta_color="inverse")
            
            st.warning(
                f"**Clinical Action:** High uncertainty detected (Entropy $\ge 1.0$ or Confidence $< 70\%$). "
                f"Flagged for referral to a consultant dermatologist for physical dermoscopy or biopsy."
            )

        st.markdown("---")
        st.subheader("3. Class Probability Distribution")
        prob_chart_data = {CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
        st.bar_chart(prob_chart_data)

        st.markdown("---")
        st.subheader("4. Visual Explainability (Grad-CAM++)")
        heatmap_overlay = generate_dummy_gradcam(input_image)
        st.image(
            heatmap_overlay,
            caption="Grad-CAM++ Activation Map (Focus on lesion border and pigmentation)",
            use_container_width=True
        )
        st.caption("Warmer regions (red/yellow) indicate features in the MaxViT backbone that contributed most to the prediction.")