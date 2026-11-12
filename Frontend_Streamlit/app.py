
import hashlib
import json
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
import hashlib
import json
from io import BytesIO

from hoi_adapter import predict_image, get_model_resources


def get_visual_interpretation(action, object_name):
    """
    Convert the model's HOI label into a cautious,
    human-readable description.

    This does not create a new model prediction.
    """

    action = action.lower().replace("_", " ").strip()
    object_name = object_name.lower().replace("_", " ").strip()

    if action == "hold" and object_name == "cell phone":
        return (
            "The person is holding a cell phone. "
            "This may indicate phone use, but talking cannot "
            "be confirmed from this prediction alone."
        )

    return f"The detected interaction is: person {action} {object_name}."

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HOI Vision Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .metric-card {
        background: #151f32;
        border: 1px solid #29364b;
        border-radius: 12px;
        padding: 17px;
        min-height: 105px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #60a5fa;
        font-size: 25px;
        font-weight: 750;
    }

    .section-heading {
        color: #f1f5f9;
        font-size: 19px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .prediction-card {
        background: #151f32;
        border: 1px solid #29364b;
        border-radius: 11px;
        padding: 13px;
        margin-bottom: 10px;
    }

    .prediction-title {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 650;
    }

    .prediction-detail {
        color: #94a3b8;
        font-size: 12px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563eb;
        border: none;
        color: white;
        border-radius: 9px;
        font-weight: 650;
        min-height: 44px;
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 8px;
    }

    [data-testid="stImage"] img {
        border-radius: 10px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "hoi_result" not in st.session_state:
    st.session_state.hoi_result = None

if "hoi_result_hash" not in st.session_state:
    st.session_state.hoi_result_hash = None

if "hoi_history" not in st.session_state:
    st.session_state.hoi_history = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🔍 HOI VISION")
    st.caption("Computer Vision Studio")

    st.divider()

    st.markdown("### Detection Settings")
    object_threshold = 0.50
    max_candidates=100

    # object_threshold = st.slider(
    #     "Object threshold",
    #     # min_value=0.10,
    #     # max_value=0.90,
    #     value=0.50,
    #     step=0.05,
    #     help="Controls which detected objects are considered for HOI candidate generation.",
    # )

    # max_candidates = st.slider(
    #     "Maximum candidates",
    #     min_value=10,
    #     max_value=200,
    #     value=100,
    #     step=10,
    #     help="Maximum number of HOI candidates considered by the graph pipeline.",
    # )

    # st.divider()

    st.markdown("### Model Information")

    with st.expander("About QPIC"):
        st.write(
            "QPIC is the primary human-object interaction detection "
            "model. It identifies people, objects and their potential "
            "interactions in an image."
        )

    with st.expander("About Hybrid GNN"):
        st.write(
            "The Hybrid GNN performs graph-based reasoning over "
            "HOI candidates and produces scores used to rank "
            "the predicted interactions."
        )

    with st.expander("About the System"):
        st.write(
            "The application combines QPIC detection with trained "
            "Hybrid GNN scoring. Inference currently runs on the CPU."
        )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

header_left, header_right = st.columns([4, 1])

with header_left:
    st.markdown(
        '<div class="main-title">HOI Vision Studio</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Human–Object Interaction Detection | QPIC + Hybrid GNN</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right; padding-top:15px; color:#34d399; '
        'font-weight:700;">● SYSTEM READY</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MODEL STATUS
# --------------------------------------------------

st.markdown('<div class="section-heading">Model Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Detection Model</div>
        <div class="metric-value">QPIC</div>
        <div class="prediction-detail">ResNet-50 backbone</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Interaction Model</div>
        <div class="metric-value">Hybrid GNN</div>
        <div class="prediction-detail">Trained checkpoint</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Inference Device</div>
        <div class="metric-value">CPU</div>
        <div class="prediction-detail">PyTorch inference</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# INPUT WORKSPACE
# --------------------------------------------------

st.markdown('<div class="section-heading">Detection Workspace</div>',
            unsafe_allow_html=True)

input_mode = st.radio(
    "Select image source",
    ["Upload Image", "Live Camera"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_file = None

if input_mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing a person and objects.",
    )

else:

    st.info(
        "Allow camera access in your browser, position the scene, "
        "and capture an image."
    )

    camera_col, empty_col = st.columns([1, 2])

    with camera_col:
        uploaded_file = st.camera_input("Capture Image")


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image_bytes = uploaded_file.getvalue()
        st.caption(
            f"Captured image size: {len(image_bytes) / (1024 * 1024):.2f} MB"
        )
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        st.markdown('<div class="section-heading">Input Preview</div>',
                    unsafe_allow_html=True)

        preview_col, details_col = st.columns([1, 2])

        with preview_col:
            st.image(
                image,
                caption="Input image",
                width=320,
            )

        with details_col:
            st.markdown("**Image Information**")
            st.write(f"Width: {image.width} pixels")
            st.write(f"Height: {image.height} pixels")
            st.write(f"Source: {input_mode}")

            st.markdown("**Active Settings**")
            st.write(f"Object threshold: {object_threshold:.2f}")
            st.write(f"Maximum candidates: {max_candidates}")

        st.divider()

        if st.button(
            "▶ Run HOI Detection",
            type="primary",
            use_container_width=True,
        ):

            st.session_state.hoi_result = None
            st.session_state.hoi_result_hash = None

            try:

                with st.spinner(
                    "Processing image with QPIC + Hybrid GNN..."
                ):

                    result = predict_image(
                        image,
                        object_threshold=object_threshold,
                        max_candidates=max_candidates,
                    )

                st.session_state.hoi_result = result
                st.session_state.hoi_result_hash = image_hash

                st.session_state.hoi_history.insert(
                    0,
                    {
                        "source": input_mode,
                        "prediction_count": len(result["predictions"]),
                        "top_action": (
                            result["predictions"][0]["action"]
                            if result["predictions"]
                            else "None"
                        ),
                        "time": result["inference_seconds"],
                    },
                )

                st.session_state.hoi_history = (
                    st.session_state.hoi_history[:10]
                )

            except RuntimeError as exc:

                st.error(f"Inference failed: {exc}")

                if "No compatible HOI candidates" in str(exc):
                    st.info(
                        "Try improving object visibility or reducing "
                        "the object threshold. Lower thresholds may "
                        "also introduce false detections."
                    )

            except Exception as exc:
                st.error(f"Unexpected inference error: {exc}")

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        result = st.session_state.hoi_result

        if result is not None:

            if st.session_state.hoi_result_hash != image_hash:

                st.info(
                    "The current image differs from the last processed "
                    "image. Run detection again."
                )

            else:

                st.divider()

                st.markdown(
                    '<div class="section-heading">Detection Results</div>',
                    unsafe_allow_html=True,
                )

                predictions = result.get("predictions", [])

                top_prediction = (
                    predictions[0] if predictions else None
                )

                best_action = (
                    top_prediction["action"]
                    if top_prediction else "None"
                )

                best_score = (
                    top_prediction["gnn_score"]
                    if top_prediction else 0
                )

                # RESULT METRICS

                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    st.metric(
                        "Interactions",
                        len(predictions),
                    )

                with r2:
                    st.metric(
                        "Top Action",
                        best_action.title(),
                    )

                with r3:
                    st.metric(
                        "Top GNN Score",
                        f"{best_score:.4f}",
                    )

                with r4:
                    st.metric(
                        "Inference Time",
                        f"{result['inference_seconds']:.2f}s",
                    )

                # IMAGE COMPARISON

                st.markdown(
                    '<div class="section-heading">Visual Analysis</div>',
                    unsafe_allow_html=True,
                )

                original_col, annotated_col = st.columns(2)

                with original_col:
                    st.markdown("**Original Image**")
                    st.image(
                        image,
                        width=320,
                    )

                with annotated_col:
                    st.markdown("**Annotated Image**")
                    st.image(
                        result["annotated_image"],
                        width=320,
                    )

                # RANKED PREDICTIONS

                st.markdown(
                    '<div class="section-heading">Ranked Interactions</div>',
                    unsafe_allow_html=True,
                )

                if predictions:

                    for item in predictions:
                        interpretation = get_visual_interpretation(
                            item["action"],
                            item["object"],
                        )

                        with st.container(border=True):
                            left, right = st.columns([4, 1])

                            with left:
                                st.markdown(
                                    f"**#{item['rank']} | MODEL PREDICTION**"
                                )

                                st.markdown(
                                    f"Person → {item['action'].title()} "
                                    f"→ {item['object'].title()}"
                                )

                                st.caption(
                                    f"QPIC: {item['qpic_score']:.4f} | "
                                    f"Object: {item['object_score']:.4f}"
                                )

                                st.markdown("**Visual interpretation**")

                                st.write(interpretation)

                            with right:
                                st.metric(
                                    "GNN Score",
                                    f"{item['gnn_score']:.3f}",
                                )

                    # PREDICTION TABLE

                    with st.expander("View complete prediction table"):

                        prediction_df = pd.DataFrame(predictions)

                        st.dataframe(
                            prediction_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    # SCORE VISUALIZATION

                    with st.expander("Explore prediction scores"):

                        score_df = pd.DataFrame(predictions)

                        score_df = score_df.set_index("rank")[
                            ["gnn_score", "qpic_score"]
                        ]

                        st.bar_chart(score_df)

                else:
                    st.warning("No interaction predictions returned.")

                # DOWNLOADS

                st.markdown(
                    '<div class="section-heading">Export Results</div>',
                    unsafe_allow_html=True,
                )

                download_col1, download_col2 = st.columns(2)

                with download_col1:

                    image_buffer = BytesIO()

                    result["annotated_image"].save(
                        image_buffer,
                        format="JPEG",
                        quality=95,
                    )

                    st.download_button(
                        "⬇ Download Annotated Image",
                        data=image_buffer.getvalue(),
                        file_name="hoi_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )

                with download_col2:

                    json_data = json.dumps(
                        result["payload"],
                        indent=4,
                    )

                    st.download_button(
                        "⬇ Download Prediction JSON",
                        data=json_data,
                        file_name="hoi_predictions.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                st.caption(result["score_note"])

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "Unable to read this image. Please upload or capture a valid image."
        )

else:

    st.session_state.hoi_result = None
    st.session_state.hoi_result_hash = None


# --------------------------------------------------
# RECENT ACTIVITY
# --------------------------------------------------

if st.session_state.hoi_history:

    st.divider()

    with st.expander("Recent Detection Activity"):

        history_df = pd.DataFrame(st.session_state.hoi_history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:12px;">
        HOI Vision Studio | QPIC + Hybrid GNN | Capstone Project
    </div>
    """,
    unsafe_allow_html=True,
)
