from ultralytics import YOLO
import cv2

model = YOLO(r'C:\Users\HP\OneDrive\Desktop\YOLO\yolov10n.pt')

input_video_path = r'C:\Users\HP\OneDrive\Desktop\YOLO\example.mp4'

#video properties 
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the Output
output_video_path = r'C:\Users\HP\OneDrive\Desktop\YOLO\output.mp4'
forcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, forcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, verbose=False)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow("YOLOv10 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to {output_video_path}")