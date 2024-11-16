import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.image import resize

# Load the pre-trained model
model = tf.keras.models.load_model('Trained_model.keras')

# List of genre classes
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_melspectrogram(audio_data, sr, chunk_duration=4, overlap_duration=2, target_shape=(150, 150)):
    chunk_samples = chunk_duration * sr
    overlap_samples = overlap_duration * sr
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    mel_spectrograms = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
        mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mel_spectrograms.append(mel_spectrogram_resized)

    return np.array(mel_spectrograms)

def predict_genre(mel_spectrograms):
    predictions = model.predict(mel_spectrograms)
    avg_prediction = np.mean(predictions, axis=0)
    predicted_index = np.argmax(avg_prediction)
    return classes[predicted_index]

def plot_melspectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))  # Adjust the figure size to be medium
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()
    st.pyplot(plt)

# Setting a custom theme through Streamlit's config.toml or directly in the app
st.set_page_config(
    page_title="Music Genre Classification",
    # layout="wide",
    menu_items={
        'About': "This app uses deep learning to predict the genre of a music track."
    }
)

# Example of injecting custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1510915361894-db8b60106cb1?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        color: white;
    }
    .stFileUploader label {
        color: white !important;
    }
    .st-btn {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
    }
    .custom-heading {
        color: white;
        font-size: 3.75em;
        font-weight: bold;
        text-align: center;
    }
    .custom-heading1 {
        color: white;
        text-align: center;
    }
    .stMarkdown {
        color: white;
    }
    .css-10trblm {
        color: white !important;
    }
    .css-1r6slb0 {
        color: white !important;
    }
    .centered-plot {
        display: flex;
        justify-content: center;
    }
    .label-input{
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use markdown for the heading to apply custom CSS
st.markdown('<h1 class="custom-heading">Music Genre Classification</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="custom-heading1">Upload an audio file to predict its genre.</h2>', unsafe_allow_html=True)
st.markdown('<p class="label-input">Choose your audio file!</p>', unsafe_allow_html=True)



uploaded_file = st.file_uploader('', type=["wav"])

if uploaded_file is not None:
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    
    st.audio(uploaded_file, format='audio/wav')
    st.write(f"Sample Rate: {sr}")
    st.write(f"Duration: {librosa.get_duration(y=audio_data, sr=sr):.2f} seconds")

    # Center the plot using CSS
    st.markdown('<div class="centered-plot">', unsafe_allow_html=True)
    
    # Display the Mel-spectrogram of the uploaded audio
    plot_melspectrogram(audio_data, sr)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Extract Mel-spectrogram features
    mel_spectrograms = extract_melspectrogram(audio_data, sr)

    # Predict the genre
    predicted_genre = predict_genre(mel_spectrograms)

    st.write(f"Predicted Genre: **{predicted_genre}**")
