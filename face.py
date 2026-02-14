import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
import requests
from io import BytesIO
from PIL import Image
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
st.set_page_config(page_title="YOLOv10 Face Recognition", layout="centered")
st.title("🎥 YOLOv10 + InsightFace Real-Time Face Recognition")

# ---------------- MONGODB ----------------
MONGO_URI = "mongodb+srv://sanjay_m1512:Welcome%40123@zoe.hx1dyxw.mongodb.net/?appName=ZOE"
client = MongoClient(MONGO_URI)
db = client["face_db"]
collection = db["known_faces"]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    yolo = YOLO(r"C:\Users\HP\OneDrive\Desktop\YOLO\yolov10n.pt")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    return yolo, face_app

yolo_model, face_model = load_models()

# ---------------- HELPERS ----------------
def get_embedding_from_url(image_url):
    response = requests.get(image_url)
    img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))

    faces = face_model.get(img)
    if len(faces) == 0:
        return None

    return faces[0].embedding


def load_known_faces():
    names, embeddings = [], []
    for doc in collection.find():
        names.append(doc["name"])
        embeddings.append(np.array(doc["embedding"]))
    return names, embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------- FACE REGISTRATION ----------------
st.subheader("➕ Register New Face")

with st.form("register_face"):
    name = st.text_input("Person Name")
    image_url = st.text_input("Face Image URL")
    submit = st.form_submit_button("Add to Database")

    if submit:
        embedding = get_embedding_from_url(image_url)
        if embedding is None:
            st.error("❌ No face detected in image")
        else:
            collection.insert_one({
                "name": name,
                "image_url": image_url,
                "embedding": embedding.tolist()
            })
            st.success(f"✅ {name} added successfully")

# ---------------- LOAD KNOWN FACES ----------------
known_names, known_embeddings = load_known_faces()

# ---------------- WEBCAM UI ----------------
st.subheader("🎥 Live Webcam Detection")

start = st.button("Start Webcam")
stop = st.button("Stop Webcam")

frame_window = st.image([])

if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam")
    else:
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model.predict(frame, verbose=False)
            annotated_frame = frame.copy()

            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                name = "Unknown"
                faces = face_model.get(face_crop)

                if faces and known_embeddings:
                    emb = faces[0].embedding
                    sims = [cosine_similarity(emb, k) for k in known_embeddings]
                    best_idx = np.argmax(sims)

                    if sims[best_idx] > 0.45:
                        name = known_names[best_idx]

                cv2.rectangle(annotated_frame, (x1, y1), (0 + x2, 0 + y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_frame)

        cap.release()
        st.success("🛑 Webcam stopped")
