import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load model, vectorizer, and label encoder
model = joblib.load('netflix_rating_model.pkl')
vectorizer = joblib.load('description_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ------------------🎛️ Theme Toggle ------------------
theme = st.sidebar.radio("🎨 Select Theme", ["Dark Mode", "Light Mode"])

# ------------------💄 Dynamic CSS Styling ------------------
if theme == "Dark Mode":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #141414;
            color: white;
        }

        section[data-testid="stSidebar"] {
            background-color: #1c1c1c !important;
        }

        /* Sidebar & Labels */
        .stSidebar, .css-1v0mbdj, .css-1cpxqw2, .stRadio > label, label, p {
            color: white !important;
        }

        /* Input Text + Borders */
        .stTextInput>div>input,
        .stTextArea>div>textarea,
        .stNumberInput>div>input,
        .stSelectbox>div>div,
        .stSelectbox>div>div>div,
        input, textarea, select {
            background-color: #222 !important;
            color: white !important;
            border: 1px solid #e50914 !important;
        }

        /* Placeholder text in text area/input */
        ::placeholder {
            color: #bbbbbb !important;
            opacity: 1 !important;
        }

        /* Dropdown values */
        div[role="combobox"] * {
            color: white !important;
        }

        /* Success messages */
        .stAlert-success {
            background-color: #222 !important;
            border-left: 5px solid #e50914 !important;
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #e50914;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.6rem 1.2rem;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #b00610;
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )



# ------------------🔴 Netflix Logo & Title ------------------
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg"
             style="width:130px; margin-bottom:-10px; filter: drop-shadow(0 0 5px #e50914);" />
        <h1 style='color:#e50914; text-shadow: 1px 1px 2px black;'>Netflix Rating Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------🧠 Description ------------------
st.write("🎞️ *Describe a show or movie and let the model guess the rating!*")

# ------------------📥 User Inputs ------------------
description = st.text_area("📜 Enter the description", placeholder="e.g. Horror, action, teen romance...")
release_year = st.number_input("📅 Release Year", min_value=1900, max_value=2025, value=2020)
duration = st.number_input("⏱️ Duration (minutes)", min_value=1, max_value=300, value=90)
show_type = st.selectbox("🎥 Type", ["Movie", "TV Show"])

# ------------------🎯 Predict Button ------------------
if st.button("🎯 Predict Rating"):
    if description.strip() == "":
        st.warning("⚠️ Please enter a description.")
    else:
        desc_vector = vectorizer.transform([description])
        type_encoded = 1 if show_type == "Movie" else 0
        numeric = np.array([[release_year, duration, type_encoded]])
        final_input = hstack([desc_vector, numeric])

        pred = model.predict(final_input)
        predicted_rating = label_encoder.inverse_transform([pred[0]])[0]

        st.success(f"🍿 **Predicted Rating: {predicted_rating}**")
