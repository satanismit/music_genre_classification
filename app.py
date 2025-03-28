import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from tensorflow.image import resize

# Load model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("Trained_model.h5")

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None, None, None

    data = []
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data), audio_data, sample_rate

# Predict genre
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    return unique_elements[np.argmax(counts)]

# Genre Information Dictionary
genre_info = {
    "blues": "🎸 Blues music is known for its soulful melodies and expressive guitar solos.",
    "classical": "🎼 Classical music includes orchestral and symphonic compositions by famous composers like Beethoven and Mozart.",
    "country": "🤠 Country music is recognized for its storytelling lyrics and acoustic instruments.",
    "disco": "🕺 Disco is a 70s dance music genre with funky basslines and upbeat rhythms.",
    "hiphop": "🎤 Hip-Hop features rhythmic beats and spoken lyrics, originating from urban culture.",
    "jazz": "🎷 Jazz is known for improvisation, swing rhythms, and brass instruments.",
    "metal": "🤘 Heavy Metal features powerful guitar riffs and aggressive vocals.",
    "pop": "🎵 Pop music is mainstream, catchy, and widely loved.",
    "reggae": "🌴 Reggae is a Jamaican music style with deep bass and offbeat rhythms.",
    "rock": "🎸 Rock music features strong beats, electric guitars, and dynamic performances."
}

# Sidebar Navigation
st.sidebar.title("🎵 Music Genre Classifier")
app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "📂 Upload & Predict", "📊 Visualization", "ℹ️ About"])

# Home Page
if app_mode == "🏠 Home":
    st.title("🎶 Welcome to the Music Genre Classifier!")
    st.write("Upload an audio file and let AI predict its genre with high accuracy.")

    # How it Works
    st.subheader("💡 How It Works?")
    st.write("""
    1️⃣ **Upload an audio file** 🎵  
    2️⃣ **AI extracts audio features** using Deep Learning 🤖  
    3️⃣ **The model predicts the genre** based on trained data 📊  
    4️⃣ **You can visualize the waveform & spectrogram** 🎨  
    """)

    # Why Choose Us?
    st.subheader("🚀 Why Choose Us?")
    st.write("""
    ✅ **Fast & Accurate Predictions**  
    ✅ **Supports 10 Popular Music Genres**  
    ✅ **Interactive Audio Visualization**  
    ✅ **AI-Powered Analysis & Recommendations**  
    """)

    # Supported Genres
    st.subheader("🎯 Supported Music Genres:")
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    st.write(", ".join(genres))

    # Footer
    st.markdown("""
    ---
    📌 **Developed by Brijesh Rakhasiya, Smit Satani & Krish Vaghani**  
    🛠 **Powered by TensorFlow & Streamlit**  
    📧 **Contact Us: support@musicgenreai.com**  
    """)

# Upload & Predict Page
elif app_mode == "📂 Upload & Predict":
    st.header("🎼 Upload an Audio File for Prediction")
    test_mp3 = st.file_uploader("Upload a file", type=["mp3", "wav"])

    if test_mp3 is not None:
        filepath = os.path.join("Test_Music", test_mp3.name)
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())

        st.session_state["test_mp3"] = filepath  
        st.audio(test_mp3)
        X_test, audio_data, sample_rate = load_and_preprocess_data(filepath)

        if X_test is not None:
            st.session_state["audio_data"] = audio_data
            st.session_state["sample_rate"] = sample_rate

            duration = librosa.get_duration(y=audio_data, sr=sample_rate)
            st.write(f"📊 **Duration:** {duration:.2f} sec | **Sample Rate:** {sample_rate} Hz")

            if st.button("🎵 Predict Genre"):
                with st.spinner("Analyzing..."):
                    genre_list = list(genre_info.keys())  
                    result_index = model_prediction(X_test)
                    predicted_genre = genre_list[result_index]
                    st.success(f"🎵 **Predicted Genre: {predicted_genre.capitalize()}**")
                    st.write(genre_info.get(predicted_genre, "No additional information available."))

# Visualization Page
elif app_mode == "📊 Visualization":
    st.header("📊 Audio Data Visualization")

    if "test_mp3" in st.session_state:
        audio_data = st.session_state["audio_data"]
        sample_rate = st.session_state["sample_rate"]

        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform")
        st.pyplot(plt)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        st.pyplot(plt)
    else:
        st.warning("⚠️ No audio file uploaded. Please go to 'Upload & Predict' to upload a file.")

# About Page
elif app_mode == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("This AI-powered classifier uses Deep Learning to identify music genres.")

    st.markdown("""
    ---
    📌 **Developed by Brijesh Rakhasiya, Smit Satani & Krish Vaghani**  
    🛠 **Powered by TensorFlow & Streamlit**  
    📧 **Contact Us: support@musicgenreai.com**  
    """)
