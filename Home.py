import streamlit as st
import sounddevice as sd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
from datetime import datetime
import gdown
import tensorflow as tf

st.title("🛡️ EchoSense")

duration = 3
samplerate = 44100
chunksize = duration * samplerate


st.markdown(
    '<h4 style="color: grey;">Sensing Danger Through Sound🎙️</h4>',
    unsafe_allow_html=True
)

# Expandable section for privacy
st.markdown("""
    This application is designed with a mission to enhance women's and children's safety in real-time. In a world where incidents of violence and emergencies can happen unexpectedly, immediate detection and response are critical.

    Our AI-powered audio classification system listens for high-risk sounds like screams and gunshots, helping identify potential emergencies early. By recognizing such sounds automatically, it has the potential to alert guardians or authorities — even when victims may be unable to reach out for help.

    ---
    **🔒 Privacy First**
    
    - No audio is recorded or stored.
    - All processing happens **locally on your device**.
    - Microphone access is granted **only with your permission**, and you can disable it anytime.
    - Detection logs (like “scream” or “gunshot”) are stored **only on your device**, and shared **only if you choose**.
    
    We believe technology should serve humanity — and do so responsibly. Empowering communities while respecting individual privacy is a powerful step in the right direction.
    """)

st.markdown("""
        ---
        """)
st.markdown("#####   ⚙️ Experience the Current Build of Our Sound Detection System")

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


col1, col2 = st.columns(2)

if 'recording' not in st.session_state:
    st.session_state.recording = False

with col1:
    if st.button("🎬 Start / Stop Recording"):
        st.session_state.recording = not st.session_state.recording

if st.session_state.recording:
    with col1:
        st.success("Recording...")

    with st.spinner("Recording..."):
        while st.session_state.recording:
            with col1:
                st.write("Recording 3 seconds...")

            audio = sd.rec(int(chunksize), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()

            with col2:
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
                    st.pyplot(fig)
                elif(predicted_class == 0):
                    predicted_class = 2
                    msg = f"""
                        **Prediction Result**

                        - **Predicted Class**: `{class_labels[predicted_class].upper()}`
                        - **Confidence**: `{confidence:.2f}%`
                        - **Time**: `{current_time}`
                        """
                    if(confidence < 70):
                        st.warning(msg)
                        st.pyplot(fig)               
                    else:
                        st.error(msg)
                        st.pyplot(fig)
                else:
                    if(confidence < 70):
                        st.warning(msg)
                        st.pyplot(fig)
                    else:
                        st.error(msg)
                        st.pyplot(fig)

