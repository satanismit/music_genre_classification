import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load pre-trained model
MODEL_PATH = "Music_Genre_Classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'metal', 'pop', 'reggae', 'rock']

# Ensure 'Test_Music' folder exists
TEST_MUSIC_FOLDER = "Test_Music"
os.makedirs(TEST_MUSIC_FOLDER, exist_ok=True)

# Function to load and preprocess audio file using chunking approach
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=22050)
        
        chunk_duration = 3
        overlap_duration = 1.5
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / 
                                (chunk_samples - overlap_samples))) + 1
        
        processed_chunks = []
        for i in range(min(num_chunks, 8)):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            if end <= len(audio_data):
                chunk = audio_data[start:end]
                mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate, n_mels=128)
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                mel_spectrogram_db = (mel_spectrogram_db - mel_spectrogram_db.min()) / \
                                   (mel_spectrogram_db.max() - mel_spectrogram_db.min())
                mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram_db, axis=-1), target_shape)
                processed_chunks.append(mel_spectrogram_resized)
        
        return np.array(processed_chunks) if processed_chunks else None
    
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Updated prediction function
def model_prediction(chunks, threshold=0.25):
    predictions = []
    confidence_scores = []
    all_predictions = []
    
    for i, chunk in enumerate(chunks):
        chunk_expanded = np.expand_dims(chunk, axis=0)
        y_pred = model.predict(chunk_expanded, verbose=0)[0]
        all_predictions.append(y_pred)
        
        predicted_class = np.argmax(y_pred)
        confidence = y_pred[predicted_class]
        
        predictions.append(predicted_class)
        confidence_scores.append(confidence)
    
    predictions = np.array(predictions)
    confidence_scores = np.array(confidence_scores)
    all_predictions = np.array(all_predictions)
    
    # Majority voting
    unique_classes, counts = np.unique(predictions, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    majority_count = np.max(counts)
    total_chunks = len(predictions)
    majority_mask = predictions == majority_class
    avg_confidence = np.mean(confidence_scores[majority_mask])
    
    # Average prediction
    mean_prediction = np.mean(all_predictions, axis=0)
    mean_pred_class = np.argmax(mean_prediction)
    mean_confidence = mean_prediction[mean_pred_class]
    
    # Final decision
    if majority_count / total_chunks >= 0.5 and avg_confidence >= threshold:
        final_class = majority_class
        final_confidence = avg_confidence
    else:
        final_class = mean_pred_class
        final_confidence = mean_confidence
    
    if final_confidence < threshold:
        return "Uncertain", final_confidence
    
    return LABELS[final_class], final_confidence

# Visualization functions
def plot_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
    ax.set_title("Audio Waveform")
    return fig

def plot_melspectrogram(audio_data, sample_rate):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

# Streamlit UI
st.set_page_config(page_title="Music Genre Classifier", layout="wide")

st.sidebar.title("🎶 Music Genre Classification 🎧")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Upload & Predict", "📊 Visualization", "ℹ️ About"])

if page == "🏠 Home":
    st.title("🎵 Welcome to the Music Genre Classification System! 🎧")
    st.write("This system uses Deep Learning to classify music into one of 9 genres.")
    st.write("""
    ## How it works:
    1. Upload a WAV audio file
    2. Audio is split into chunks
    3. Chunks are converted to mel spectrograms
    4. Model predicts genre using ensemble voting
    """)
    try:
        st.image("UI.jpg", use_column_width=True)
    except:
        st.write("Note: UI image not found.")

elif page == "📂 Upload & Predict":
    st.title("📂 Upload a WAV file and Predict the Genre")
    uploaded_file = st.file_uploader("Upload your WAV file", type=["wav"])

    if uploaded_file:
        file_path = os.path.join(TEST_MUSIC_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(file_path, format="audio/wav")
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        st.write(f"📊 **Audio Information:**")
        st.write(f"- **Duration:** {duration:.2f} seconds")
        st.write(f"- **Sample Rate:** {sample_rate} Hz")
        
        st.subheader("Audio Waveform")
        st.pyplot(plot_waveform(audio_data, sample_rate))

        debug_mode = st.checkbox("Enable Debug Mode")

        if st.button("Predict Genre"):
            with st.spinner("⏳ Processing audio into chunks..."):
                chunks = load_and_preprocess_data(file_path)

            if chunks is not None:
                st.success(f"✅ Processed {len(chunks)} audio chunks")
                
                if debug_mode:
                    st.subheader("Debug Information")
                    st.write(f"Chunks shape: {chunks.shape}")
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    img = ax.imshow(np.squeeze(chunks[0]), aspect='auto', origin='lower')
                    plt.colorbar(img, ax=ax)
                    st.pyplot(fig)
                
                with st.spinner("⏳ Predicting genre..."):
                    genre, confidence = model_prediction(chunks)
                
                if debug_mode:
                    st.subheader("Prediction Analysis")
                    all_preds = [model.predict(np.expand_dims(chunk, axis=0), verbose=0)[0] 
                               for chunk in chunks]
                    all_preds = np.array(all_preds)
                    
                    mean_probs = np.mean(all_preds, axis=0)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(LABELS, mean_probs)
                    ax.set_title("Average Probability Distribution")
                    ax.set_ylabel("Probability")
                    ax.set_ylim(0, 1)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                st.subheader("Prediction Result")
                if genre == "Uncertain":
                    st.warning(f"⚠️ **Uncertain Prediction** (Confidence: {confidence:.2%})")
                    st.write("Try a different sample or longer clip.")
                else:
                    st.success(f"🎵 **Predicted Genre: {genre.upper()}**")
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    genre_descriptions = {
                        "blues": "Characterized by blue notes and specific chord progressions.",
                        "classical": "Known for complex structure and orchestral instrumentation.",
                        "country": "Features string instruments and narrative lyrics.",
                        "disco": "Has a steady beat and is designed for dancing.",
                        "hiphop": "Features rhythmic vocals and beats.",
                        "metal": "Characterized by heavy guitars and aggressive vocals.",
                        "pop": "Features catchy melodies for mass appeal.",
                        "reggae": "Has distinctive rhythm and bass-heavy sound.",
                        "rock": "Features electric guitars and strong backbeat."
                    }
                    st.write(f"**Description:** {genre_descriptions.get(genre, '')}")

elif page == "📊 Visualization":
    st.title("📊 Audio Visualization")
    uploaded_file = st.file_uploader("Upload your WAV file for visualization", type=["wav"])
    
    if uploaded_file:
        file_path = os.path.join(TEST_MUSIC_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(file_path, format="audio/wav")
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        st.subheader("Audio Waveform")
        st.pyplot(plot_waveform(audio_data, sample_rate))
        
        st.subheader("Mel Spectrogram")
        st.pyplot(plot_melspectrogram(audio_data, sample_rate))
        
        st.subheader("Additional Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Chromagram**")
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma', ax=ax)
            ax.set_title("Chromagram")
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("**Spectral Contrast**")
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(contrast, sr=sample_rate, x_axis='time', ax=ax)
            ax.set_title("Spectral Contrast")
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("""
    ## Music Genre Classifier
    This AI-powered classifier uses Deep Learning to identify music genres.
    
    ### Features:
    - Advanced audio preprocessing
    - Ensemble prediction with majority voting
    - Comprehensive visualizations
    
    ### Supported Genres:
    - Blues, Classical, Country, Disco, Hip-Hop, Metal, Pop, Reggae, Rock
    
    ### Technologies:
    - TensorFlow, Librosa, Streamlit, Matplotlib
    """)

st.markdown("""
    ---
    📌 **Developed by Brijesh Rakhasiya, Smit Satani & Krish Vaghani**  
    🛠 **Powered by TensorFlow & Streamlit**  
""")