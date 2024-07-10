import streamlit as st
import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Definisikan Kelas ANNModel sebelum memuat model
class ANNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_size, hidden_size2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load model terbaik
with open('best_ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

model.eval()

# Load encoder
label_encoder = LabelEncoder()
try:
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
except FileNotFoundError:
    st.error("File 'classes.npy' tidak ditemukan. Pastikan file tersebut telah dibuat dan disimpan di lokasi yang benar.")

# Load data untuk rekomendasi
try:
    mfcc_features = pd.read_csv('mfcc_feature_nama.csv')
except FileNotFoundError:
    st.error("File 'mfcc_feature_nama.csv' tidak ditemukan. Pastikan file tersebut telah dibuat dan disimpan di lokasi yang benar.")

def extract_mfcc(file):
    # Muat file audio
    y, sr = librosa.load(file, sr=None)
    # Ekstrak MFCC (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def predict_class_with_confidence(mfcc):
    inputs = torch.tensor(mfcc).float().unsqueeze(0)  # Tambahkan dimensi batch
    outputs = model(inputs)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)
    return label_encoder.inverse_transform(predicted.cpu().numpy())[0], confidence.item()

def recommend_songs(input_features, genre, num_recommendations=10):
    # Filter berdasarkan genre yang sama
    genre_filtered = mfcc_features[mfcc_features['Label'] == genre]
    # Konversi input_features menjadi NumPy array
    input_features = np.array(input_features).reshape(1, -1)
    # Pastikan hanya kolom numerik yang digunakan
    numeric_features = genre_filtered.drop(columns=['Label', 'Filename']).values
    # Hitung cosine similarity
    similarities = cosine_similarity(input_features, numeric_features)
    # Ambil indeks dari similarity tertinggi
    similar_indices = similarities[0].argsort()[-num_recommendations:][::-1]
    recommendations = genre_filtered.iloc[similar_indices]
    return recommendations['Filename'].tolist()

# Streamlit UI
st.title('Music Genre Classification and Recommendation')
st.subheader("Kelompok D1")


# Nama anggota kelompok

st.write('Anggota Kelompok:')
st.write("1. I Made Prenawa Sida Nanda (2208561017)\n"
                "2. Kadek Yuni Suratri (2208561055)\n"
               "3. Pande Komang Bhargo Anantha Yogiswara (2208561067)\n"
               "4. I Gusti Agung Ayu Gita Pradnyaswari Mantara (2208561105)\n"
               "5. I Kadek Agus Candra Widnyana (2208561129)")

uploaded_file = st.file_uploader("Upload a music file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Ekstrak fitur MFCC dari file musik
    mfcc = extract_mfcc(uploaded_file)
    
    # Memutar file musik yang diunggah
    st.audio(uploaded_file, format='audio/wav')
    
    # Prediksi genre dengan persentase kecocokan
    predicted_genre, confidence = predict_class_with_confidence(mfcc)
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #6B8A7A;">
            <h2>The predicted genre is: <em><strong>{predicted_genre}</strong></em></h2>
            <p style="background-color: #254336; padding: 10px; border-radius: 5px;">
                Confidence: {confidence * 100:.2f}%
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Rekomendasi lagu
    recommendations = recommend_songs(mfcc, predicted_genre)
    st.write('Recommended songs:')
    st.markdown('\n'.join([f"- **{song}**" for song in recommendations]))
