import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
from skimage.transform import resize

# Load pre-trained model
MODEL_PATH = "Music_Genre_Classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Only 9 classes as per training)
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'metal', 'pop', 'reggae', 'rock']

# Ensure 'Test_Music' folder exists
TEST_MUSIC_FOLDER = "Test_Music"
os.makedirs(TEST_MUSIC_FOLDER, exist_ok=True)

# Function to load and preprocess audio file
def load_and_preprocess_data(file_path, target_shape=(128, 128)):
    try:
        # Load audio file (mono-channel, first 10 seconds for better accuracy)
        audio_data, sample_rate = librosa.load(file_path, sr=22050, mono=True, duration=10)

        # Convert to Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128)

        # Convert to Decibel scale (log-Mel)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Normalize between 0 and 1
        mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min())

        # Resize to match model input shape
        mel_spectrogram_resized = resize(mel_spectrogram_db, target_shape)

        return np.expand_dims(mel_spectrogram_resized, axis=(0, -1))  # Add batch & channel dimensions
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Function to make predictions
def model_prediction(X_test, threshold=0.4):
    y_pred = model.predict(X_test, batch_size=1)
    max_prob = np.max(y_pred)

    # If confidence is too low, return "Uncertain"
    if max_prob < threshold:
        return "Uncertain Prediction. Try another sample."

    predicted_category = np.argmax(y_pred, axis=1)[0]  # Get highest probability index
    return LABELS[predicted_category]  # Return the genre

# Streamlit UI
st.set_page_config(page_title="Music Genre Classifier", layout="wide")

# Sidebar for Navigation
st.sidebar.title("🎶 Music Genre Classification 🎧")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Upload & Predict", "ℹ️ About"])

if page == "🏠 Home":
    st.title("🎵 Welcome to the Music Genre Classification System! 🎧")
    st.write("This system uses **Deep Learning** to classify music into one of 9 genres.")
    st.image("C:/Users/brije/Desktop/SGP/music_genre_classification/UI.jpg", use_column_width=True)


elif page == "📂 Upload & Predict":
    st.title("📂 Upload a WAV file and Predict the Genre")
    uploaded_file = st.file_uploader("Upload your WAV file", type=["wav"])

    if uploaded_file:
        # Define the save path
        file_path = os.path.join(TEST_MUSIC_FOLDER, uploaded_file.name)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show audio player
        st.audio(file_path, format="audio/wav")

        if st.button("Predict Genre"):
            # Preprocess and Predict
            X_test = load_and_preprocess_data(file_path)

            if X_test is not None:
                with st.spinner("⏳ Predicting..."):
                    genre = model_prediction(X_test)
                
                st.success(f"🎵 Predicted Genre: **{genre}**")

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("""
    - This AI-powered music genre classifier identifies the genre of a given song using **Deep Learning**.
    - Built using **TensorFlow, Streamlit, and Librosa**.
    - Supports **9 music genres**: Blues, Classical, Country, Disco, Hip-Hop, Metal, Pop, Reggae, and Rock.
    """)

# Footer
st.markdown("""
    ---
    📌 **Developed by Brijesh Rakhasiya , Smit Satani & Krish Vaghani**  
    🛠 **Powered by TensorFlow & Streamlit**  
""")