import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# Load pre-trained embeddings and labels
embeddings = np.load('embeddings.npy')
labels_df = pd.read_csv('labels.csv')

# Initialize FaceAnalysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Streamlit App
st.title("Face Detection and Detail Extraction Using AI")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert uploaded file bytes directly to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # 1 = color image (BGR)

    # Show uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    # Detect faces
    faces = app.get(img)
    if len(faces) == 0:
        st.error("No face detected.")
    else:
        for face in faces:
            embedding = face.normed_embedding.reshape(1, -1)
            similarities = cosine_similarity(embedding, embeddings)[0]
            max_sim = np.max(similarities)
            max_index = np.argmax(similarities)

            if max_sim > 0.6:
                usn = labels_df.iloc[max_index]['USN']
                name = labels_df.iloc[max_index]['Name']
                st.success(f"✅ Match Found: {name} ({usn}) | Similarity: {max_sim:.2f}")
            else:
                st.warning("⚠️ No Match Found.")

