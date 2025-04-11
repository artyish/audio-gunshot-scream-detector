import streamlit as st
import os
import time
import random
from datetime import datetime
import librosa
import matplotlib.pyplot as plt
import numpy as np


def download_model():
    file_id = "1GYWZ3sgidwDOSOd2xIBXqKeIIWbJjsWj"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "maybeFinalModel123.keras"

    if not os.path.exists(output):
        with st.spinner("Downloading model..."):
            gdown.download(url, output, quiet=False)
            st.success("Model downloaded successfully!")
    return output

model_path = download_model()
model = tf.keras.models.load_model(model_path)

from predictEmergency import predict

duration = 3
samplerate = 44100
chunksize = duration * samplerate


st.title("Gunshot Detection Demo")

st.markdown("Listen to preloaded sounds and predict if it's a gunshot.")

audio_folder = "audio_files"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith((".wav", ".mp3"))]

selected_file = st.selectbox("Choose an audio to play", audio_files)

file_path = os.path.join(audio_folder, selected_file)
st.audio(file_path)
audio, sr = librosa.load(file_path, sr=samplerate)
audio = audio.flatten()
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
prediction , predicted_class , class_labels = predict(audio)


confidence = prediction[0, predicted_class]*100
msg = f"""
        **Prediction Result**

        - **Predicted Class**: `{class_labels[predicted_class].upper()}`
        - **Confidence**: `{confidence:.2f}%`
        - **Time**: `{current_time}`
        """
fig, ax = plt.subplots(figsize=(8, 3))
S = librosa.feature.melspectrogram(y=audio, sr=samplerate)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, sr=samplerate, x_axis='time', y_axis='mel', ax=ax)
ax.set_title('Mel Spectrogram')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
if(predicted_class == 1):
    st.success(msg)
    st.write(f'Prediction confidence for {class_labels[0].upper()} is {prediction[0,0]*100:.2f}%')
    st.pyplot(fig)
elif(predicted_class == 0):
    if(confidence < 70):
        st.warning(msg)
        st.pyplot(fig)               
    else:
        st.error(msg)
        st.pyplot(fig)
else:
    if(confidence < 70):
        st.warning(msg)
        st.write(f'Prediction confidence for {class_labels[0].upper()} is {prediction[0,0]*100:.2f}%')
        st.pyplot(fig)
    else:
        st.error(msg)
        st.write(f'Prediction confidence for {class_labels[0].upper()} is {prediction[0,0]*100:.2f}%') 
        st.pyplot(fig)
