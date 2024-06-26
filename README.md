# Real-Time Face Recognition with FaceNet and MTCNN

This Python script demonstrates one-shot face recognition using FaceNet for face embeddings and MTCNN for face detection. It captures video frames from a webcam, detects faces, recognizes known faces based on pre-computed embeddings, and annotates the video stream with bounding boxes and labels.

# What is One-Shot Learning?
One-shot learning is a method where a few examples are used to classify new instances. In face recognition:

   - Face Embeddings: Systems use one-shot learning to create compact face representations called embeddings, capturing unique facial 
     features crucial for verification and identification.

   - Embedding Space: Faces are mapped to an embedding space where distances between embeddings indicate facial similarities, enabling recognition of new faces based on comparisons with known ones.

   - Deep Learning Model: Trained on extensive face datasets, the model generates these embeddings by extracting distinctive facial features.

This approach efficiently classifies faces with minimal training data, ideal for scenarios where large datasets are impractical.

## Features

- **Real-Time Face Detection**: Utilizes MTCNN (Multi-Task Cascaded Convolutional Neural Network) for accurate face detection in real-time video streams.
- **Face Embeddings**: Generates embeddings using FaceNet, a deep neural network that maps faces into a 128-dimensional vector space.
- **Face Recognition**: Recognizes known faces by comparing face embeddings with pre-computed embeddings stored in a file (`embeddings.pkl`).
- **Annotation**: Draws bounding boxes around detected faces and annotates with the name of recognized individuals.

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/abhishekmaher6699/face-recognition.git
   cd face-recognition
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Known Faces**:
   
  - Place images of known individuals in the images/ directory.
  - Each person's images should be in a separate subdirectory named after the person (person_1, person_2, etc.).
  ```
  root_directory/ 
  │ 
  ├── images/ 
  │ ├── person_1/ 
  │ ├── person_2/ 
  │ ├── person_3/ 
  │ └── ...
  ```

4. **Run the application**:

   Run the app by executing the following script.
   ```
   python face_recognition.py
   ```
5. **Interact**:
   - Press 'q' to exit the application.
   - The annotated video stream will display recognized faces with bounding boxes and labels.

