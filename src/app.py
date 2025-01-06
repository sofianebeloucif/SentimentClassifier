import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

import pandas as pd
from preprocess import preprocess

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================
# App Header
# =============================
st.markdown("---")
st.title("🧠 Emotion Detection App")
st.markdown("""
This app analyzes the emotion conveyed in a given text.  
Data Source: [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).  
**Developed by Sofiane Beloucif.**
""")
st.markdown("---")

# =============================
# Load Resources (Models and Encoders)
# =============================
@st.cache_resource
def load_resources():
    encoder = pickle.load(open('../utils/encoder.pkl', 'rb'))
    cv = pickle.load(open('../utils/CountVectorizer.pkl', 'rb'))
    model = tf.keras.models.load_model('/models/naive_model.h5')
    return encoder, cv, model

encoder, cv, model = load_resources()

# =============================
# User Input Section
# =============================
st.sidebar.markdown("## User Input")
st.sidebar.markdown("---")

def get_user_input():
    text = st.sidebar.text_area("Enter your sentence below:", height=150)
    return text

user_text = get_user_input()

# =============================
# Emotion Prediction Section
# =============================
st.markdown("## 🎯 Emotion Prediction")
if not user_text.strip():
    st.write("🔹 **Please enter a sentence in the sidebar to detect its emotion.**")
else:
    preprocessed_text = preprocess(user_text)
    array = cv.transform([preprocessed_text]).toarray()
    pred = model.predict(array)
    a = np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]
    st.success(f"The detected emotion is: **{prediction}**")

st.markdown("---")

# =============================
# Visualization: Bar Chart
# =============================
st.markdown("## 📊 Words Required for Accurate Prediction")
st.markdown("""
This section illustrates how many words were necessary for the model to correctly predict the emotion in a dataset sample of 1,000 sentences.
""")

@st.cache_data
def load_result_data():
    return pd.read_csv('../data/result_rows.csv')

result_rows = load_result_data()

# Prepare data for the bar chart
chart_data = result_rows['pred_correct_at_words'].value_counts().reset_index()
chart_data.columns = ['Words Count', 'Frequency']
chart_data = chart_data.sort_values('Words Count')

# Display bar chart
st.bar_chart(chart_data.set_index('Words Count'))

st.markdown("---")

# =============================
# Footer
# =============================
st.markdown("""
### 💻 About this App  
Developed by **Sofiane Beloucif**.  
You can view the code and give a ⭐️ at [GitHub Repository](https://github.com/sofianebeloucif/SentimentClassifier).  
""")
