import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

@st.cache_resource
def load_assets():

    model = load_model("next_word_lstm.h5")

    with open("tokenizer.pickle","rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_assets()


# -------------------------------------------------
# Max Sequence Length
# -------------------------------------------------

def get_max_sequence_length(model):

    shape = model.input_shape

    if len(shape) > 1:
        return shape[1] + 1

    return 20


MAX_SEQUENCE_LENGTH = get_max_sequence_length(model)


# -------------------------------------------------
# Reverse Vocabulary
# -------------------------------------------------

index_word = {}

for word,index in tokenizer.word_index.items():
    index_word[index] = word


# -------------------------------------------------
# Prediction Function
# -------------------------------------------------

def predict_next_word(model, tokenizer, text, max_sequence_length):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return None,None,None

    if len(token_list) >= max_sequence_length:

        token_list = token_list[-(max_sequence_length-1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_length-1,
        padding="pre"
    )

    prediction = model.predict(token_list,verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_word = index_word.get(predicted_index,None)

    confidence = float(prediction[predicted_index])

    top5_index = np.argsort(prediction)[-5:][::-1]

    top5=[]

    for idx in top5_index:

        word=index_word.get(idx)

        if word:

            top5.append(

                (
                    word,
                    float(prediction[idx])
                )

            )

    return predicted_word,confidence,top5


# -------------------------------------------------
# Session State
# -------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history=[]

if "prediction" not in st.session_state:

    st.session_state.prediction=None

if "confidence" not in st.session_state:

    st.session_state.confidence=0

if "top5" not in st.session_state:

    st.session_state.top5=[]


# -------------------------------------------------
# Sample Prompts
# -------------------------------------------------

SAMPLE_PROMPTS=[

    "to be or not",

    "the king is",

    "i have a",

    "my lord",

    "what is the",

    "there is no",

    "shall i",

    "you are"

]


# ============================================================
# CSS
# ============================================================

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html,body,[class*="css"]{

font-family:'Poppins',sans-serif;

}

.stApp{

background:linear-gradient(135deg,#0f172a,#1e293b,#111827);

color:white;

}

section[data-testid="stSidebar"]{

background:#111827;

border-right:1px solid rgba(255,255,255,.08);

}

.main-title{

font-size:48px;

font-weight:800;

margin-bottom:5px;

background:linear-gradient(90deg,#00DBDE,#FC00FF);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}

.sub-title{

font-size:18px;

color:#cbd5e1;

margin-bottom:30px;

}

.card{

background:rgba(255,255,255,.05);

padding:20px;

border-radius:20px;

border:1px solid rgba(255,255,255,.08);

backdrop-filter:blur(15px);

box-shadow:0 8px 30px rgba(0,0,0,.35);

}

.metric{

font-size:35px;

font-weight:700;

color:#38bdf8;

}

.metric-title{

font-size:14px;

color:#cbd5e1;

margin-bottom:5px;

}

.result{

background:linear-gradient(135deg,#2563eb,#4f46e5);

padding:25px;

border-radius:20px;

text-align:center;

margin-top:15px;

box-shadow:0 8px 30px rgba(0,0,0,.35);

}

.result h1{

font-size:55px;

margin:0;

}

.result p{

font-size:18px;

margin-top:10px;

}

.footer{

text-align:center;

margin-top:50px;

color:#94a3b8;

font-size:14px;

}

div.stButton > button{

width:100%;

height:50px;

border:none;

border-radius:12px;

background:linear-gradient(90deg,#06b6d4,#3b82f6);

color:white;

font-weight:700;

font-size:18px;

}

div.stButton > button:hover{

background:linear-gradient(90deg,#3b82f6,#06b6d4);

}

</style>

""",unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧠 Model Information")

st.sidebar.markdown("---")

st.sidebar.metric(

"Vocabulary",

len(tokenizer.word_index)

)

st.sidebar.metric(

"Context Length",

MAX_SEQUENCE_LENGTH-1

)

st.sidebar.metric(

"Architecture",

"LSTM"

)

st.sidebar.markdown("---")

st.sidebar.write(

"""
### Sample Prompts

- to be or not

- the king is

- i have a

- my lord

- what is the

"""
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Model Loaded Successfully")

# ============================================================
# HERO
# ============================================================

st.markdown(

"""
<div class='main-title'>

📝 Next Word Predictor

</div>

<div class='sub-title'>

Predict the next word using a Deep Learning LSTM Model trained on Shakespeare's Hamlet.

</div>

""",

unsafe_allow_html=True

)

# ============================================================
# TOP METRICS
# ============================================================

c1,c2,c3=st.columns(3)

with c1:

    st.markdown(

    f"""

    <div class='card'>

    <div class='metric-title'>

    Vocabulary Size

    </div>

    <div class='metric'>

    {len(tokenizer.word_index)}

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

with c2:

    st.markdown(

    f"""

    <div class='card'>

    <div class='metric-title'>

    Context Window

    </div>

    <div class='metric'>

    {MAX_SEQUENCE_LENGTH-1}

    </div>

    </div>

    """,

    unsafe_allow_html=True

    )

with c3:

    st.markdown(

    """

    <div class='card'>

    <div class='metric-title'>

    Neural Network

    </div>

    <div class='metric'>

    LSTM

    </div>

    </div>

    """,

    unsafe_allow_html=True

)

st.write("")

# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([2.2,1])

# ============================================================
# LEFT PANEL
# ============================================================

with left:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("✍️ Enter Your Prompt")

    prompt = st.text_input(
        "",
        placeholder="Example : to be or not to be",
        value="",
        key="prompt"
    )

    st.caption(f"Words : {len(prompt.split())}")

    st.write("")

    st.markdown("#### 🔥 Sample Prompts")

    c1,c2,c3,c4 = st.columns(4)

    if c1.button("To Be"):
        st.session_state.prompt="to be or not"

    if c2.button("King"):
        st.session_state.prompt="the king is"

    if c3.button("Lord"):
        st.session_state.prompt="my lord"

    if c4.button("What"):
        st.session_state.prompt="what is the"

    st.write("")

    col1,col2=st.columns([3,1])

    with col1:

        predict=st.button(
            "🚀 Predict Next Word",
            use_container_width=True
        )

    with col2:

        clear=st.button(
            "🗑 Clear",
            use_container_width=True
        )

    if clear:

        st.session_state.prediction=None
        st.session_state.confidence=0
        st.session_state.top5=[]
        st.rerun()

    if predict:

        if prompt.strip()=="":

            st.warning("Please enter some text.")

        else:

            with st.spinner("Predicting..."):

                word,conf,top5 = predict_next_word(
                    model,
                    tokenizer,
                    prompt.lower().strip(),
                    MAX_SEQUENCE_LENGTH
                )

            if word is None:

                st.error("Prediction not available.")

            else:

                st.session_state.prediction=word
                st.session_state.confidence=conf
                st.session_state.top5=top5

                st.session_state.history.insert(
                    0,
                    (
                        prompt,
                        word
                    )
                )

                if len(st.session_state.history)>10:

                    st.session_state.history.pop()

    st.markdown("</div>",unsafe_allow_html=True)

# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("📊 Prediction")

    if st.session_state.prediction:

        st.markdown(

            f"""

            <div class='result'>

            <h1>

            {st.session_state.prediction}

            </h1>

            <p>

            Predicted Next Word

            </p>

            </div>

            """,

            unsafe_allow_html=True

        )

        st.write("")

        st.write("Confidence")

        st.progress(st.session_state.confidence)

        st.success(
            f"{st.session_state.confidence*100:.2f}%"
        )

    else:

        st.info("No prediction yet.")

    st.markdown("</div>",unsafe_allow_html=True)

# ============================================================
# TOP 5 WORDS
# ============================================================

st.write("")

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("🏆 Top 5 Predictions")

if len(st.session_state.top5)>0:

    rank=1

    for word,prob in st.session_state.top5:

        c1,c2=st.columns([5,2])

        with c1:

            st.write(f"**{rank}. {word}**")

        with c2:

            st.progress(prob)

            st.caption(f"{prob*100:.2f}%")

        rank+=1

else:

    st.info("No prediction available.")

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# PREDICTION HISTORY
# ============================================================

st.write("")

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("🕘 Recent Predictions")

if len(st.session_state.history) == 0:

    st.info("No prediction history available.")

else:

    for i,(prompt,prediction) in enumerate(st.session_state.history,1):

        col1,col2=st.columns([5,2])

        with col1:

            st.markdown(

                f"""
                **{i}.**

                Prompt

                > {prompt}

                """,

                unsafe_allow_html=True

            )

        with col2:

            st.success(prediction)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# ============================================================
# MODEL INFORMATION
# ============================================================

st.write("")

left,right=st.columns(2)

with left:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("📚 Model Details")

    st.markdown(f"""

**Architecture**

LSTM Neural Network

---

**Vocabulary Size**

{len(tokenizer.word_index)}

---

**Context Length**

{MAX_SEQUENCE_LENGTH-1}

---

**Prediction**

Next Word Prediction

---

**Framework**

TensorFlow + Keras

""")

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True
    )

    st.subheader("💡 Tips")

    st.markdown("""

✅ Use lowercase text

✅ Use Shakespeare-like sentences

✅ Short prompts give better results

✅ Unknown words reduce accuracy

✅ Click Predict after typing

""")

    st.markdown("</div>",unsafe_allow_html=True)

# ============================================================
# SIMPLE ANALYTICS
# ============================================================

st.write("")

st.markdown(
    "<div class='card'>",
    unsafe_allow_html=True
)

st.subheader("📈 Statistics")

col1,col2,col3=st.columns(3)

with col1:

    st.metric(

        "Vocabulary",

        len(tokenizer.word_index)

    )

with col2:

    st.metric(

        "Predictions",

        len(st.session_state.history)

    )

with col3:

    if st.session_state.confidence:

        st.metric(

            "Last Confidence",

            f"{st.session_state.confidence*100:.2f}%"

        )

    else:

        st.metric(

            "Last Confidence",

            "0%"

        )

st.markdown("</div>",unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.write("")
st.write("")

st.markdown("---")

st.markdown("""

<div class="footer">

<h3>📝 Next Word Predictor</h3>

Built using ❤️ Streamlit • TensorFlow • Keras

Developed by <b>Aman Singh</b>

</div>

""",unsafe_allow_html=True)