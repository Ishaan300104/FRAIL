import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ghostnet import ghost_net
from ghostnet import load_pretrained_weights

# Load pre-trained FaceNet model

def extract_face_embedding(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1,3,224,224))
    model=ghost_net()
    pretrained_weights_path = (r"ghostnet.pth")  # Replace with the actual path
    load_pretrained_weights(model, pretrained_weights_path)

    # Generate embeddings using FaceNet model
    embedding = model(img)
    
    return embedding.flatten()

def save_embedding(name, embedding, embeddings_dir='embeddings'):
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, f'{name}.npy'), embedding)

def load_embeddings(embeddings_dir='embeddings'):
    embeddings = {}
    for file_name in os.listdir(embeddings_dir):
        if file_name.endswith('.npy'):
            path = os.path.join(embeddings_dir, file_name)
            embedding = np.load(path)
            name = os.path.splitext(file_name)[0]
            embeddings[name] = embedding
    return embeddings

def find_similar_faces(query_embedding, embeddings, threshold=0.7):
    similarities = {}
    for name, stored_embedding in embeddings.items():
        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        if similarity > threshold:
            similarities[name] = similarity
    return similarities

if __name__ == '__main__':
    # Extract embeddings from images
    embeddings = {}
    for file_name in os.listdir('images'):
        if file_name.endswith('.jpg'):
            path = os.path.join('images', file_name)
            name = os.path.splitext(file_name)[0]
            embedding = extract_face_embedding(path)
            embeddings[name] = embedding
            save_embedding(name, embedding)
    
    # Load embeddings from files
    embeddings = load_embeddings()
    
    # Find similar faces
    query_embedding = extract_face_embedding('images/face.jpg')
    similarities = find_similar_faces(query_embedding, embeddings)
    print(similarities)
