import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np


st.set_page_config(page_title="YOLOv10 Webcam Detection", layout="centered")
st.title("YOLOv10 Real-Time Webcam Detection")


@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\HP\OneDrive\Desktop\YOLO\yolov10n.pt")


model = load_model()


start = st.button("Start Webcam")
stop = st.button("Stop Webcam")


frame_window = st.image([])


if start:
    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        st.error("Cannot access webcam")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break


            results = model.predict(frame, verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_frame)


        cap.release()
        # st.success("Webcam stped")
        st.success("Webcam Off")



