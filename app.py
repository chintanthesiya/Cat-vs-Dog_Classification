from pathlib import Path
import base64
import pickle

import cv2
import numpy as np
import streamlit as st
from PIL import Image


IMAGE_SIZE = (256, 256)
MODEL_PATH = Path(__file__).with_name("model.pkl")
BACKGROUND_IMAGE = Path(__file__).parent / "archive (8)" / "train" / "dogs" / "dog.10030.jpg"
CAT_REFERENCE = Path(__file__).with_name("test_cat.jpg")
DOG_REFERENCE = Path(__file__).with_name("dog1.jpg")


st.set_page_config(page_title="Cat or Dog", page_icon="🐾", layout="centered")

background_url = ""
if BACKGROUND_IMAGE.exists():
    encoded_background = base64.b64encode(BACKGROUND_IMAGE.read_bytes()).decode("ascii")
    background_url = f"url(data:image/jpeg;base64,{encoded_background})"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root { --ink: #17221c; --muted: #66736a; --leaf: #39745a; --mint: #e5f0e8; --paper: #fbfaf5; --line: #dce5dd; --coral: #e8795d; }
    .stApp { background-color: var(--paper); background-image: __BACKGROUND_IMAGE__; background-size: cover; background-position: center; background-attachment: fixed; color: var(--ink); }
    .stApp:before { content: ''; position: fixed; inset: 0; background: rgba(251,250,245,.86); pointer-events: none; z-index: 0; }
    .block-container { position: relative; z-index: 1; max-width: 1180px; padding: 1.5rem 2rem 2.5rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    p, label, [data-testid="stMarkdownContainer"] { font-family: 'DM Sans', sans-serif; }
    .hero { position: relative; padding: 1.25rem 1.4rem 1.1rem; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.9); box-shadow: 0 18px 50px rgba(39, 71, 52, .12); margin-bottom: .8rem; overflow: hidden; }
    .hero:after { content: '🐾'; position: absolute; right: 2.2rem; bottom: -.8rem; font-size: 6rem; opacity: .08; transform: rotate(-15deg); }
    .eyebrow { display: inline-block; color: var(--leaf); background: var(--mint); border-radius: 999px; padding: .4rem .75rem; font: 700 .72rem 'DM Sans', sans-serif; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.5rem, 5vw, 4rem); line-height: .92; margin: .7rem 0 .45rem; max-width: 600px; }
    .hero p { color: var(--muted); font-size: 1rem; margin: 0; max-width: 500px; }
    .section-heading { display: flex; align-items: baseline; gap: .7rem; margin: .85rem 0 .55rem; }
    .section-number { color: var(--coral); font: 700 .78rem 'DM Sans', sans-serif; letter-spacing: .08em; }
    .section-heading h3 { margin: 0; font-size: 1.25rem; }
    .upload-card { border: 1px dashed #a9bdad; border-radius: 14px; padding: .55rem; background: rgba(255,255,255,.78); }
    .meta { display: flex; justify-content: space-between; gap: 1rem; border-top: 1px solid var(--line); margin-top: .8rem; padding: .8rem .3rem .1rem; color: var(--muted); font-size: .86rem; }
    .meta strong { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    .stage { border: 1px solid var(--line); border-radius: 14px; background: white; padding: 1rem; height: 100%; }
    .stage-label { color: var(--muted); font: 700 .7rem 'DM Sans', sans-serif; letter-spacing: .11em; text-transform: uppercase; margin-bottom: .65rem; }
    .shape { color: var(--leaf); background: var(--mint); border-radius: 8px; padding: .7rem .8rem; font: 700 .9rem 'Space Grotesk', sans-serif; }
    .result { background: linear-gradient(135deg, #39745a, #285340); border-radius: 14px; padding: 1.7rem; color: white; margin-top: 1rem; box-shadow: 0 15px 35px rgba(39, 83, 64, .2); }
    .result .label { color: #cce5d3; font: 700 .76rem 'DM Sans', sans-serif; letter-spacing: .1em; text-transform: uppercase; }
    .result h2 { font-size: 3rem; margin: .45rem 0; }
    .result p { color: #e2f0e4; margin: 0; }
    .side-pet { border-radius: 18px; border: 1px solid var(--line); overflow: hidden; background: white; box-shadow: 0 18px 40px rgba(39, 71, 52, .12); }
    .side-pet-label { color: var(--muted); text-align: center; font: 700 .7rem 'DM Sans', sans-serif; letter-spacing: .1em; text-transform: uppercase; padding: .6rem .4rem .7rem; }
    .side-pet img { display: block; aspect-ratio: 1 / 1.15; object-fit: cover; }
    div[data-testid="stFileUploader"] { border: 0; background: transparent; }
    div.stButton > button { border-radius: 999px; min-height: 3rem; font-family: 'DM Sans', sans-serif; font-weight: 700; }
    @media (max-width: 800px) { .side-pet { display: none; } }
    @media (max-width: 640px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding: 1.25rem 1.15rem; } .hero:after { right: .5rem; } }
    </style>
    """.replace("__BACKGROUND_IMAGE__", background_url),
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl was not found beside app.py")

    import keras  # noqa: F401

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def prepare_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb_image = np.asarray(image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    resized_image = cv2.resize(bgr_image, IMAGE_SIZE)
    model_input = resized_image.reshape(1, 256, 256, 3)
    return resized_image, model_input


def predict(model, model_input: np.ndarray) -> tuple[str, float]:
    try:
        raw_prediction = model.predict(model_input, verbose=0)
    except TypeError:
        raw_prediction = model.predict(model_input)

    scores = np.asarray(raw_prediction, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("The model returned an empty prediction")
    if scores.size == 1:
        dog_probability = float(np.clip(scores[0], 0.0, 1.0))
        label = "Dog" if dog_probability >= 0.5 else "Cat"
        confidence = dog_probability if label == "Dog" else 1 - dog_probability
        return label, confidence

    probabilities = scores[:2]
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        probabilities = np.exp(probabilities - probabilities.max())
        probabilities /= probabilities.sum()
    class_index = int(np.argmax(probabilities))
    return ("Cat", "Dog")[class_index], float(probabilities[class_index])


try:
    model = load_model()
except Exception as error:
    model = None
    st.error(f"Could not load the model: {error}")

prediction = None
left_pet, main_content, right_pet = st.columns([.72, 1.7, .72], gap="large")
with left_pet:
    if CAT_REFERENCE.exists():
        st.markdown('<div class="side-pet">', unsafe_allow_html=True)
        st.image(CAT_REFERENCE, width="stretch")
        st.markdown('<div class="side-pet-label">Cat reference</div></div>', unsafe_allow_html=True)
with main_content:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Local vision model · ready</div>
          <h1>Cat or dog?</h1>
          <p>Upload a photo and let the model make the call.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a cat or dog image.",
    )
    predict_clicked = st.button(
        "Predict cat or dog",
        type="primary",
        width="stretch",
        disabled=not uploaded_file or model is None,
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="section-heading"><span class="section-number">01</span><h3>Your image</h3></div>', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", width="stretch")

        if predict_clicked:
            with st.spinner("Preparing image..."):
                _, model_input = prepare_image(image)

            with st.spinner("Predicting..."):
                label, confidence = predict(model, model_input)
            prediction = (label, confidence)
    else:
        st.info("Upload a JPG or PNG image to begin.")
with right_pet:
    if DOG_REFERENCE.exists():
        st.markdown('<div class="side-pet">', unsafe_allow_html=True)
        st.image(DOG_REFERENCE, width="stretch")
        st.markdown('<div class="side-pet-label">Dog reference</div></div>', unsafe_allow_html=True)
    if prediction:
        label, confidence = prediction
        st.markdown('<div class="section-heading"><span class="section-number">02</span><h3>Prediction</h3></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result"><div class="label">Prediction</div>'
            f'<h2>{label}</h2><p>Confidence: {confidence:.1%}</p></div>',
            unsafe_allow_html=True,
        )

