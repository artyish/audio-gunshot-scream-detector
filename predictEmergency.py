from tensorflow.keras.models import load_model
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa

model = load_model("maybeFinalModel123.keras")

sr = 44100
duration = 3

def predict(aud):    
    print("recording started")

    #audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32') # here we start the recording
    #sd.wait() # wait until recording gets finished
    #audio = audio.flatten()
    #write('recorded_audio.wav', sr, audio)
    #print(audio)

    #print("recording finished")
    audio = aud
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_normalized = (S_dB + 80)/(+80-0)  
    input_vector = S_dB_normalized.flatten()
    input_vector = input_vector.reshape(1, -1)

    
    prediction = model.predict(input_vector)
    print("Class probabilities:", prediction)
    class_labels = ["gunshot", "normal", "scream"]
    predicted_class = np.argmax(prediction)
    print(f'The predicted class is : {class_labels[predicted_class].upper()} with a confidence rate of : {prediction[0, predicted_class]*100:.4f}%')

    return(prediction , predicted_class , class_labels)




