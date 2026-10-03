"""
Stress & Depression Detector – Deep Learning-Based Detection of Stress and Depression from Text
Streamlit front end for a Bi-LSTM text classifier (Final Year Project).

Run:  streamlit run app.py
Needs Streamlit >= 1.40 (uses keyed containers for card styling).
"""

import logging
import pickle
from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import clean_text

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/stress_model.h5")
TOKENIZER_PATH = Path("models/tokenizer.pkl")  # saved during training (see notes)
MAX_WORDS = 10_000
MAX_LEN = 150
THRESHOLD = 0.5
MAX_CHARS = 1500

logger = logging.getLogger("psytext")

EXAMPLES = {
    "Overwhelmed": "I can't keep up anymore. Every day feels heavier than the last and I'm exhausted all the time.",
    "Hopeful": "Had a great weekend hiking with friends. Feeling rested and looking forward to the week ahead!",
    "Anxious": "My heart races every time I think about tomorrow's deadline. I can't sleep and I can't focus.",
}

st.set_page_config(
    page_title="Stress & Depression Detector",
    
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Styling  (charcoal · lime · soft grey)
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --charcoal: #232623;
    --charcoal-2: #2F332F;
    --charcoal-3: #454A45;
    --lime: #C5F135;
    --lime-dark: #A9D41C;
    --grey-bg: #ECEEEB;
    --grey-card: #F7F8F6;
    --grey-line: #D9DCD7;
    --grey-text: #6B716B;
    --coral: #FF7F5C;
}

html, body, [class*="css"], .stApp, textarea, button {
    font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}

/* App shell */
.stApp { background: var(--grey-bg); }
header[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
.block-container { max-width: 1180px; padding: 2rem 1.5rem 3rem; }

/* Brand bar */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.2rem; padding: 0 .2rem;
}
.brand { display: flex; align-items: center; gap: .7rem; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 12px;
    background: var(--charcoal);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    box-shadow: inset 0 -4px 0 var(--lime);
}
.brand-name { font-size: 1.3rem; font-weight: 800; color: var(--charcoal); letter-spacing: -.02em; line-height: 1; }
.brand-name span { background: var(--lime); padding: 0 .3rem; border-radius: 6px; }
.brand-tag { font-size: .82rem; font-weight: 600; color: var(--grey-text); text-align: right; max-width: 340px; line-height: 1.35; }

/* Hero */
.hero {
    background: var(--charcoal);
    border-radius: 24px;
    padding: 2.6rem 2.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -70px; top: -70px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: var(--lime);
    opacity: .95;
}
.hero-badge {
    display: inline-block;
    background: rgba(197,241,53,.14);
    color: var(--lime);
    font-weight: 700;
    font-size: .8rem;
    padding: .35rem .8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #fff;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -.02em;
    line-height: 1.12;
    margin: 0 0 .8rem;
    max-width: 640px;
    padding: 0;
}
.hero p {
    color: #B9BEB8;
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 560px;
    margin: 0;
}

.hero p.hero-sub {
    color: #fff;
    font-size: 1.2rem;
    font-weight: 700;
    max-width: 520px;
    margin: 0 0 .8rem;
}

/* Panel headings */
.panel-title { font-size: 1.15rem; font-weight: 800; color: var(--charcoal); margin: 0 0 .25rem; }
.panel-sub   { font-size: .9rem; color: var(--grey-text); margin: 0 0 1rem; }

/* Input card (st.container(key="input_card")) */
.st-key-input_card {
    background: var(--grey-card);
    border: 1px solid var(--grey-line);
    border-radius: 20px;
    padding: 1.6rem 1.6rem 1.4rem;
}
.st-key-input_card textarea {
    background: #fff !important;
    color: var(--charcoal) !important;
    border: 1.5px solid var(--grey-line) !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: 1rem !important;
    min-height: 220px;
}
.st-key-input_card textarea:focus {
    border-color: var(--charcoal) !important;
    box-shadow: 0 0 0 4px rgba(197,241,53,.55) !important;
}
.st-key-input_card textarea::placeholder { color: #9AA09A; }
.st-key-input_card [data-baseweb="textarea"],
.st-key-input_card [data-baseweb="base-input"] { background: transparent !important; border: none !important; }
.st-key-input_card [data-testid="stTextAreaRootElement"] { border: none !important; }
.try-label { font-size: .85rem; font-weight: 700; color: var(--grey-text); margin: .9rem 0 .4rem; }

/* Primary button */
.st-key-input_card button[kind="primary"],
.st-key-input_card button[data-testid="stBaseButton-primary"] {
    background: var(--lime);
    color: var(--charcoal);
    border: none;
    border-radius: 12px;
    font-weight: 800;
    font-size: 1rem;
    padding: .7rem 1.2rem;
    width: 100%;
    transition: background .15s ease, transform .1s ease;
}
.st-key-input_card button[kind="primary"]:hover,
.st-key-input_card button[data-testid="stBaseButton-primary"]:hover {
    background: var(--lime-dark);
    color: var(--charcoal);
}
.st-key-input_card button[kind="primary"]:active { transform: scale(.985); }
.st-key-input_card button[kind="primary"]:focus-visible { outline: 3px solid var(--charcoal); outline-offset: 2px; }

/* Example + clear buttons */
.st-key-input_card button[kind="secondary"],
.st-key-input_card button[data-testid="stBaseButton-secondary"] {
    background: #fff;
    color: var(--charcoal);
    border: 1.5px solid var(--grey-line);
    border-radius: 999px;
    font-weight: 600;
    font-size: .85rem;
    padding: .25rem .9rem;
    width: 100%;
}
.st-key-input_card button[kind="secondary"]:hover,
.st-key-input_card button[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--charcoal);
    color: var(--charcoal);
    background: #fff;
}

/* Result card */
.result {
    background: var(--charcoal);
    border-radius: 20px;
    padding: 1.8rem;
    color: #fff;
}
.result.calm   { --accent: var(--lime); }
.result.stress { --accent: var(--coral); }
.result-chip {
    display: inline-flex; align-items: center; gap: .45rem;
    background: rgba(255,255,255,.08);
    color: var(--accent);
    font-weight: 700; font-size: .85rem;
    padding: .35rem .85rem; border-radius: 999px;
}
.result-chip i { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); display: inline-block; }
.result h2 { color: #fff; font-size: 1.7rem; font-weight: 800; letter-spacing: -.01em; margin: 1rem 0 .3rem; padding: 0; }
.result .lead { color: #B9BEB8; font-size: .95rem; line-height: 1.55; margin: 0 0 1.4rem; }

.score-row { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: .6rem; }
.score-big { font-size: 3.2rem; font-weight: 800; color: var(--accent); line-height: 1; letter-spacing: -.03em; }
.score-label { font-size: .85rem; color: #B9BEB8; font-weight: 600; text-align: right; }

.meter { position: relative; height: 12px; background: var(--charcoal-3); border-radius: 999px; }
.meter-fill { height: 100%; border-radius: 999px; background: var(--accent); }
.meter-tick { position: absolute; left: 50%; top: -5px; width: 2px; height: 22px; background: #fff; opacity: .55; }
.meter-scale { display: flex; justify-content: space-between; font-size: .75rem; color: #8E948E; margin-top: .5rem; }

.stats { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin-top: 1.5rem; }
.stat { background: var(--charcoal-2); border-radius: 14px; padding: .9rem 1rem; }
.stat b { display: block; font-size: 1.15rem; color: #fff; font-weight: 800; }
.stat span { font-size: .8rem; color: #9DA39D; }

.notice { margin-top: 1.2rem; font-size: .8rem; line-height: 1.5; color: #9DA39D; border-top: 1px solid var(--charcoal-3); padding-top: 1rem; }

/* Empty state */
.empty {
    border: 2px dashed #C3C7C1;
    border-radius: 20px;
    padding: 3rem 1.8rem;
    text-align: center;
    color: var(--grey-text);
    background: rgba(247,248,246,.6);
}
.empty .icon { font-size: 2.4rem; margin-bottom: .6rem; }
.empty h3 { color: var(--charcoal); font-size: 1.1rem; font-weight: 800; margin: 0 0 .3rem; padding: 0; }
.empty p { font-size: .92rem; margin: 0; line-height: 1.55; }

/* Footer */
.footer { text-align: center; color: var(--grey-text); font-size: .8rem; margin-top: 2rem; line-height: 1.6; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 12px; }

@media (max-width: 760px) {
    .brand-tag { display: none; }
    .brand-name { font-size: 1.05rem; }
    .hero { padding: 1.8rem 1.4rem; }
    .hero h1 { font-size: 1.8rem; }
    .hero::after { width: 160px; height: 160px; right: -60px; top: -60px; }
    .score-big { font-size: 2.6rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_trained_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_tokenizer():
    """Load the tokenizer fitted on the TRAINING data, if available."""
    if TOKENIZER_PATH.exists():
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    return None


model = load_trained_model()
saved_tokenizer = load_tokenizer()


# ─────────────────────────────────────────────────────────────
# Logic
# ─────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {"error": "empty_after_clean"}

    if saved_tokenizer is not None:
        tokenizer, used_saved = saved_tokenizer, True
    else:
        # Fallback: results are unreliable because word indices won't match
        # the ones the model was trained on. Developer-only warning (console).
        logger.warning("models/tokenizer.pkl not found; fitting tokenizer on input text.")
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts([cleaned])
        used_saved = False

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])

    return {
        "prob": prob,
        "is_stress": prob > THRESHOLD,
        "words": len(text.split()),
        "used_saved_tokenizer": used_saved,
    }


def confidence_label(prob: float) -> str:
    c = max(prob, 1 - prob)
    if c >= 0.85:
        return "High"
    if c >= 0.65:
        return "Moderate"
    return "Low"


def set_example(key: str):
    st.session_state.user_text = EXAMPLES[key]
    st.session_state.result = None


def clear_all():
    st.session_state.user_text = ""
    st.session_state.result = None


def render_result(r: dict):
    prob = r["prob"]
    pct = round(prob * 100, 1)
    if r["is_stress"]:
        cls, chip = "stress", "Indicators detected"
        title = "Stress is likely present"
        lead = ("This text contains language often linked with stress, anxiety or low mood.")
    else:
        cls, chip = "calm", "No indicators detected"
        title = "This text reads as calm"
        lead = ("This text doesn't show the language patterns usually linked with stress.")

    html = (
        f'<div class="result {cls}">'
        f'<span class="result-chip"><i></i>{chip}</span>'
        f'<h2>{title}</h2>'
        f'<p class="lead">{lead}</p>'
        f'<div class="score-row">'
        f'<div class="score-big">{pct:.0f}%</div>'
        f'<div class="score-label">stress level</div>'
        f'</div>'
        f'<div class="meter"><div class="meter-fill" style="width:{pct}%"></div>'
        f'<div class="meter-tick"></div></div>'
        f'<div class="meter-scale"><span>Low</span><span>High</span></div>'
        f'<div class="stats">'
        f'<div class="stat"><b>{confidence_label(prob)}</b><span>Confidence</span></div>'
        f'<div class="stat"><b>{r["words"]}</b><span>Words checked</span></div>'
        f'</div>'
        f'<p class="notice">This detector is an automated check, not a medical diagnosis. '
        f'If you or someone you know is struggling, please talk to a mental health '
        f'professional or a local crisis service.</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────
st.session_state.setdefault("user_text", "")
st.session_state.setdefault("result", None)

st.markdown(
    '<div class="topbar">'
    '<div class="brand">'
    '<div class="brand-mark">🧠</div>'
    '<div class="brand-name">Stress &amp; Depression <span>Detector</span></div>'
    '</div>'
    '<div class="brand-tag">Deep Learning-Based Detection of Stress and Depression from Text</div>'
    '</div>'
    '<div class="hero">'
    '<span class="hero-badge">Instant results</span>'
    '<h1>Is stress showing up in your words?</h1>'
    '<p>Paste a post, comment or message. The detector reads the language and shows '
    'how strongly it points to stress.</p>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1], gap="large")

# ── Input column ──
with left:
    with st.container(key="input_card"):
        st.markdown(
            '<p class="panel-title">Your text</p>'
            '<p class="panel-sub">Write or paste a post. Longer text gives a more stable result.</p>',
            unsafe_allow_html=True,
        )
        st.text_area(
            "Text to analyse",
            key="user_text",
            height=230,
            max_chars=MAX_CHARS,
            placeholder="Type or paste text here…",
            label_visibility="collapsed",
        )

        st.markdown('<p class="try-label">Or try an example</p>', unsafe_allow_html=True)
        ex_cols = st.columns(len(EXAMPLES) + 1)
        for col, name in zip(ex_cols, EXAMPLES):
            col.button(name, key=f"ex_{name}", on_click=set_example, args=(name,))
        ex_cols[-1].button("Clear", key="clear", on_click=clear_all)

        st.write("")
        analyze = st.button("Analyze text", type="primary", key="analyze")

    if analyze:
        text = st.session_state.user_text
        if not text.strip():
            st.session_state.result = None
            st.warning("Enter some text first, then select Analyze text.")
        else:
            with st.spinner("Analysing…"):
                st.session_state.result = predict(text)

# ── Result column ──
with right:
    result = st.session_state.result
    if result is None:
        st.markdown(
            '<div class="empty">'
            '<div class="icon">📝</div>'
            '<h3>Your result will appear here</h3>'
            '<p>Add some text on the left and select <b>Analyze text</b> to see '
            'your stress level.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif result.get("error") == "empty_after_clean":
        st.warning("There was no usable text left after cleaning. Try a longer or different post.")
    else:
        render_result(result)

st.markdown(
    '<p class="footer">Stress &amp; Depression Detector · Final Year Project<br>'
    'Results are automated estimates, not a medical diagnosis, and do not replace '
    'professional advice.</p>',
    unsafe_allow_html=True,
)