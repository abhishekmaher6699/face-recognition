from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import os
import cv2
import pickle

# Initialize the FaceNet model and MTCNN detector
model = FaceNet()
detector = MTCNN()
EMBEDDINGS = 'embeddings.pkl'

# Resizes the data images to a standard size
def preprocess_image(image_path, image_size=160):
    img = Image.open(image_path)
    img = img.resize((image_size, image_size))
    img = np.asarray(img)
    return img

# Extracts the face from the webcam frame
def extract_face(image, face_data, recognize=False):
    if face_data:
        if recognize:
            x1, y1, width, height = face_data
        else:
            x1, y1, width, height = face_data[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face)
        face_image = face_image.resize((160, 160))
        face_array = np.asarray(face_image)
        return face_array
    else:
        return None

# Generates embedding for a given image
def generate_embedding(img):
    face = np.expand_dims(img, axis=0)
    embeddings = model.embeddings(face)
    return embeddings[0] / np.linalg.norm(embeddings[0], ord=2)

# Creates an embeddings file of the known faces
def create_face_embeddings(known_faces_dir):
    known_faces = {}
    for person_dir in os.listdir(known_faces_dir):
        person_faces = []
        person_name = person_dir
        print(person_name)
        for filename in os.listdir(os.path.join(known_faces_dir, person_dir)):
            if filename.endswith('.jpg'):
                path = os.path.join(known_faces_dir, person_dir, filename)
                image = preprocess_image(path)
                results = detector.detect_faces(image)
                face = extract_face(image, results)
                embedding = generate_embedding(face)
                if embedding is not None:
                    person_faces.append(embedding)

        if person_faces:
            if len(person_faces) > 1:
                known_faces[person_name] = np.mean(person_faces, axis=0)
            else:
                known_faces[person_name] = person_faces[0]
            print(f"Embeddings stored for {person_name}")
        else:
            print(f"Warning: No faces found for {person_name}. Skipping this person.")

    with open(EMBEDDINGS, 'wb') as f:
        pickle.dump(known_faces, f)

    return known_faces

# Loads the known faces embedding file
def load_face_embeddings():
    if os.path.exists(EMBEDDINGS):
        with open(EMBEDDINGS, 'rb') as f:
            known_faces = pickle.load(f)
        print("Loaded embeddings from file.")
        return known_faces
    else:
        print("No embeddings file found. Please run the embedding creation process first.")
        return {}

# Recognizes the face in the frame and draws a bounding box around it 
def recognize_faces_in_frame(frame, known_faces):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    for face in faces:
        x, y, w, h = face['box']
        face_img = extract_face(img_rgb, face['box'], recognize=True)

        if face_img is not None:
            face_embedding = generate_embedding(face_img)
            name = 'Unknown'
            min_dist = float('inf')
            for person, embedding in known_faces.items():
                distance = np.linalg.norm(face_embedding - embedding)
                if distance < min_dist and distance < 0.7:
                    min_dist = distance
                    name = person

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

if __name__ == '__main__':
    known_faces_dir = r"\images"
    known_faces = load_face_embeddings()
    if not known_faces:
        known_faces = create_face_embeddings(known_faces_dir)

    cap = cv2.VideoCapture(0)  # opens webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = recognize_faces_in_frame(frame, known_faces)
        cv2.imshow('Face Recognition', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
