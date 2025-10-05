import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from keras.models import load_model
from config import get_config
from graph import ECG_model, zeropad, zeropad_output_shape
import os

class_names = ['N', 'V', '/', 'A', 'F', '~']

def preprocess_signal(signal, input_size=256):
    signal = np.nan_to_num(signal)
    if len(signal) > input_size:
        signal = signal[:input_size]
    elif len(signal) < input_size:
        signal = np.pad(signal, (0, input_size - len(signal)), 'constant')
    signal = scale(signal)
    return signal

# Load config and model
config = get_config()

try:
    model = ECG_model(config)
    model.load_weights("models/MLII-latest.keras")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="ECG Classifier", layout="centered")
st.title("🩺 ECG Signal Classifier")
st.markdown("Upload an ECG signal to classify the heartbeat type.")

uploaded_file = st.file_uploader("Upload ECG file (.csv)", type=["csv"])
sample_option = st.selectbox("Or choose a sample signal:", ["None", "Sample 1", "Sample 2"])

# Load signal
if uploaded_file:
    signal = np.loadtxt(uploaded_file, delimiter=",")
elif sample_option != "None":
    sample_path = f"samples/{sample_option}.csv"
    if not os.path.exists(sample_path):
        st.error(f"Sample file '{sample_option}.csv' not found in /samples folder.")
        st.stop()
    signal = np.loadtxt(sample_path, delimiter=",")
else:
    st.info("Please upload a signal or choose a sample to begin.")
    st.stop()

# Plot signal
st.subheader("📈 ECG Signal")
fig, ax = plt.subplots()
ax.plot(signal)
ax.set_title("Raw ECG Signal")
st.pyplot(fig)

# Predict
processed = preprocess_signal(signal, input_size=config.input_size)
prediction = model.predict(processed.reshape(1, -1, 1))
predicted_class = class_names[np.argmax(prediction)]

st.success(f"Suggested Diagnosis: **{predicted_class}**")
st.markdown(f"Confidence: `{np.max(prediction):.2f}`")

# Explanation in English only
explanations = {
    'N': 'Normal beat',
    'V': 'Ventricular beat',
    '/': 'Paced beat',
    'A': 'Atrial beat',
    'F': 'Fusion beat',
    '~': 'Noisy signal'
}

st.markdown(f"**Explanation:** {explanations.get(predicted_class, 'Unknown type')}")
