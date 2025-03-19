# 🎵 Music Genre Classification

## 📌 Project Overview
This project is a **Deep Learning-based Music Genre Classification System** that predicts the genre of a given music file using **Convolutional Neural Networks (CNNs)**. The model takes a `.wav` file as input, extracts its **Mel spectrogram**, and classifies it into one of 9 genres.

## 🏗️ Project Structure
```
├── Test_Music/                     # Directory to store test audio files
├── Music_Genre_Classifier.h5       # Pre-trained CNN model
├── README.md                       # Project documentation
├── UI.jpg                           # UI preview image
├── app.py                          # Streamlit web application
├── music-genra-classification.ipynb # Jupyter notebook for model training
├── requirements.txt                # Required dependencies
```

## 📌 Features
- 🎧 **Upload & Predict**: Upload a `.wav` file and get an instant genre prediction.
- 🔍 **Deep Learning Model**: Uses a **CNN trained on GTZAN dataset**.
- 📊 **9 Music Genres Supported**:
  - Blues, Classical, Country, Disco, Hip-Hop, Metal, Pop, Reggae, and Rock.
- 🌍 **Web UI with Streamlit**: User-friendly interface for easy predictions.

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/satanismit/music_genre_classification.git
cd music_genre_classification
```
### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 3️⃣ Run the Web App
```sh
streamlit run app.py
```

## 🚀 Usage
- Upload a **.wav** file in the Streamlit app.
- The model will process and classify it into a genre.
- View the result instantly on the web UI.

## 📌 Model Training
The model was trained using **Librosa for feature extraction** and **TensorFlow/Keras for deep learning**. The dataset used is **GTZAN Music Genre Dataset**, where each audio file is converted into a **Mel spectrogram** and resized to `128x128` before passing through a CNN model.


## ⭐ Acknowledgments
- **GTZAN Music Genre Dataset**
- **TensorFlow & Librosa Libraries**

If you like this project, don't forget to ⭐ the repository! 🚀

